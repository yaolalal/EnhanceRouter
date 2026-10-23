"""Canonical-block frozen-view prototype and exact anchor-neighbor classifiers."""
import itertools

import torch
import torch.nn.functional as F

from eargtc.joint_cross_repair.training import _score


GLOBAL_NODE_BLOCK_SIZE = 2048


def validate_anchors(ids, gold, pseudo):
    if ids.unique().numel() != ids.numel() or not torch.isin(ids, torch.cat([gold, pseudo])).all():
        raise ValueError("anchor identity outside gold/frozen pseudo pool")


def anchors(f, s, p):
    pseudo = torch.cat([p["teach_g"], p["teach_l"], p["agreement"]]).unique(sorted=True)
    ids = torch.cat([s["gold_ids"], pseudo])
    validate_anchors(ids, s["gold_ids"], pseudo)
    if torch.isin(ids, torch.cat([s["selection_ids"], s["threshold_ids"]])).any():
        raise ValueError("validation anchor forbidden")
    labels = torch.cat([s["gold_targets"], f["selected_pred"][pseudo]])
    weights = torch.cat([torch.ones(len(s["gold_ids"])), f["trust"][pseudo].clamp(.05, 1.)])
    gold = torch.arange(len(ids)) < len(s["gold_ids"])
    return ids, labels, weights, gold


def _views(f, h, ids, device):
    def normalize(x):
        x = x.to(device=device, dtype=torch.float32)
        return F.normalize(F.layer_norm(x, (x.shape[-1],)), dim=1)

    return normalize(f["gnn_embeddings"][ids]), normalize(h[ids])


def _neighbors(sim, query_ids, anchor_ids, k):
    order = anchor_ids.to(sim.device).argsort(stable=True)
    values = sim[:, order].clone()
    values[query_ids.to(sim.device)[:, None] == anchor_ids.to(sim.device)[order][None, :]] = -torch.inf
    positions = values.argsort(dim=1, descending=True, stable=True)[:, :min(k, sim.shape[1])]
    v = values.gather(1, positions)
    idx = order[positions]
    if not torch.isfinite(v[:, 0]).all():
        raise ValueError("no non-self anchor")
    return v, idx


def _neighbor_mass(v, idx, labels, weights, classes, k, temperature):
    v, idx = v[:, :k], idx[:, :k]
    mass = ((v - v[:, 0:1]) * temperature).exp() * weights.to(v.device)[idx]
    result = torch.zeros((len(v), classes), device=v.device)
    result.scatter_add_(1, labels.to(v.device)[idx], mass)
    return (result + 1e-9) / (result.sum(1, keepdim=True) + classes * 1e-9)


def neighbor_probability(sim, query_ids, anchor_ids, labels, weights, classes, k, temperature):
    v, idx = _neighbors(sim, query_ids, anchor_ids, k)
    return _neighbor_mass(v, idx, labels, weights, classes, k, temperature)


def _prototype(ag, al, labels, weights, classes):
    target = F.one_hot(labels, classes).float() * weights[:, None]
    counts = target.sum(0)
    denom = counts[:, None].clamp_min(1e-9)
    return F.normalize(target.T @ ag / denom, dim=1), F.normalize(target.T @ al / denom, dim=1), counts / counts.sum()


def _global_similarity_blocks(f, h, query_ids, anchor_g, anchor_l, mix, device):
    """Yield requested rows after canonical full-node block dot products."""
    node_ids = f["node_ids"].detach().cpu()
    requested = query_ids.detach().cpu()
    if node_ids.ndim != 1 or requested.ndim != 1 or node_ids.unique().numel() != len(node_ids):
        raise ValueError("invalid full-node or query IDs")
    if requested.unique().numel() != len(requested) or not torch.isin(requested, node_ids).all():
        raise ValueError("query IDs must be a unique subset of full nodes")
    locations = {int(node_id): pos for pos, node_id in enumerate(requested.tolist())}
    canonical = node_ids.argsort(stable=True)
    for start in range(0, len(canonical), GLOBAL_NODE_BLOCK_SIZE):
        block_ids = node_ids[canonical[start:start + GLOBAL_NODE_BLOCK_SIZE]]
        selected = [(offset, locations[int(node_id)]) for offset, node_id in enumerate(block_ids.tolist())
                    if int(node_id) in locations]
        if not selected:
            continue
        block_g, block_l = _views(f, h, block_ids, device)
        # Both products cover the complete canonical block before requested rows are sliced.
        similarity = mix * (block_g @ anchor_g.T) + (1 - mix) * (block_l @ anchor_l.T)
        rows = torch.tensor([item[0] for item in selected], device=device)
        positions = torch.tensor([item[1] for item in selected], dtype=torch.long)
        yield positions, requested[positions], similarity[rows]


def _global_neighbors(f, h, query_ids, anchor_ids, anchor_g, anchor_l, mix, device, k):
    width = min(k, len(anchor_ids))
    values = torch.empty((len(query_ids), width), device=device)
    indices = torch.empty((len(query_ids), width), dtype=torch.long, device=device)
    for positions, block_ids, similarity in _global_similarity_blocks(
        f, h, query_ids, anchor_g, anchor_l, mix, device
    ):
        block_values, block_indices = _neighbors(similarity, block_ids, anchor_ids, k)
        device_positions = positions.to(device)
        values[device_positions] = block_values
        indices[device_positions] = block_indices
    return values, indices


def fit_retrieval(f, s, p, h, device="cuda"):
    ids, labels, weights, gold = anchors(f, s, p)
    ag, al = _views(f, h, ids, device)
    qg, ql = _views(f, h, s["selection_ids"], device)
    classes = f["logp_g"].shape[1]
    y = s["selection_targets"]
    labels = labels.to(device)
    weights = weights.to(device)
    gold = gold.to(device)
    best = {}
    history = []

    def consider(family, config, prob):
        prob = prob.detach().cpu()
        record = {"family": family, "config": config, **_score(prob, y)}
        history.append(record)
        if family not in best or (-record["accuracy"], record["nll"]) < (-best[family]["accuracy"], best[family]["nll"]):
            best[family] = {**record, "validation_probability": prob}

    for pw in [0., .25, 1., 4.]:
        w = weights * torch.where(gold, 1., pw)
        cg, cl, prior = _prototype(ag, al, labels, w, classes)
        sg, sl = qg @ cg.T, ql @ cl.T
        for mix, temp, pr in itertools.product([0., .5, 1.], [5., 20., 50.], [0., .5]):
            config = {"pseudo_weight": pw, "mix": mix, "temperature": temp, "prior": pr}
            prob = (temp * (mix * sg + (1 - mix) * sl) + pr * prior.clamp_min(1e-9).log()).softmax(1)
            consider("prototype", config, prob)
    for mix in [0., .5, 1.]:
        v, idx = _global_neighbors(f, h, s["selection_ids"], ids, ag, al, mix, device, 128)
        for k, temp, gw in itertools.product([8, 32, 128], [5., 20., 50.], [1., 4., 16.]):
            config = {"mix": mix, "k": k, "temperature": temp, "gold_weight": gw}
            w = weights * torch.where(gold, gw, 1.)
            consider("knn", config, _neighbor_mass(v, idx, labels, w, classes, k, temp))
    for family in best:
        best[family].update({"anchor_ids": ids, "anchor_labels": labels.cpu(), "anchor_weights": weights.cpu(),
                             "anchor_gold": gold.cpu(), "validation_ids": s["selection_ids"].clone(),
                             "test_labels_read": False})
    return best, history


@torch.no_grad()
def predict_retrieval(member, f, h, ids, device="cuda", batch_size=2048):
    if batch_size < 1:
        raise ValueError("invalid prediction batch size")
    anchor_ids = member["anchor_ids"]
    labels = member["anchor_labels"].to(device)
    weights = member["anchor_weights"].to(device)
    gold = member["anchor_gold"].to(device)
    ag, al = _views(f, h, anchor_ids, device)
    classes = f["logp_g"].shape[1]
    cfg = member["config"]
    family = member["family"]
    if family == "knn":
        weights = weights * torch.where(gold, cfg["gold_weight"], 1.)
        v, idx = _global_neighbors(f, h, ids, anchor_ids, ag, al, cfg["mix"], device, cfg["k"])
        return _neighbor_mass(v, idx, labels, weights, classes, cfg["k"], cfg["temperature"]).cpu()
    if family != "prototype":
        raise ValueError("unknown retrieval family")
    weights = weights * torch.where(gold, 1., cfg["pseudo_weight"])
    cg, cl, prior = _prototype(ag, al, labels, weights, classes)
    result = []
    for start in range(0, len(ids), batch_size):
        part = ids[start:start + batch_size]
        qg, ql = _views(f, h, part, device)
        score = cfg["mix"] * (qg @ cg.T) + (1 - cfg["mix"]) * (ql @ cl.T)
        result.append((cfg["temperature"] * score + cfg["prior"] * prior.clamp_min(1e-9).log()).softmax(1).cpu())
    return torch.cat(result)
