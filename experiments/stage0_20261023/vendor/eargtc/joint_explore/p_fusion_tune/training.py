"""Gold and accepted-pseudo supervision; no graph/truth file access."""
import time
import torch
from torch.nn import functional as F
from .model import Fusion
from .distillation import teacher_probability, distillation_loss
from .protocol import slurm


def pseudo_loss(logits, targets, agreement, mode, smoothing):
    if len(targets) == 0:
        return logits.sum() * 0
    values = F.cross_entropy(logits, targets, reduction='none', label_smoothing=smoothing)
    if mode == 'pooled':
        return values.mean()
    if mode != 'balanced':
        raise ValueError('Unknown pseudo pool mode')
    return torch.stack([values[mask].mean() for mask in (agreement, ~agreement) if bool(mask.any())]).mean()


def sample_pseudo(accepted, agreement, batch_size, mode, generator):
    if not len(accepted):
        return accepted
    if mode == 'pooled':
        return accepted[torch.randint(len(accepted), (batch_size,), generator=generator)]
    if mode != 'balanced':
        raise ValueError('Unknown pseudo pool mode')
    groups = [accepted[agreement[accepted]], accepted[~agreement[accepted]]]
    groups = [g for g in groups if len(g)]
    result = []
    for i, group in enumerate(groups):
        count = batch_size // len(groups) + (i < batch_size % len(groups))
        result.append(group[torch.randint(len(group), (count,), generator=generator)])
    return torch.cat(result)


@torch.no_grad()
def predict(model, data, ids, batch=2048):
    model.eval()
    out = []
    for ix in ids.split(batch):
        z = model(data['graph'][ix], data['text'][ix])
        if not bool(torch.isfinite(z).all()):
            raise ValueError('Nonfinite logits')
        out.append(z.softmax(1).cpu())
    return torch.cat(out)


def score(prob, labels):
    return {'accuracy': float((prob.argmax(1) == labels).double().mean()),
            'nll': float(-prob[torch.arange(len(labels)), labels].double().clamp_min(1e-12).log().mean())}


def new_model(data, cfg):
    return Fusion(data['graph'].shape[1], data['text'].shape[1], data['classes'],
                  cfg['graph_width'], cfg['text_width'], cfg['hidden'], cfg['dropout'])


def fit(data, checkpoint, selection, cfg):
    slurm()
    if cfg['seed'] != data['seed']:
        raise ValueError('Training seed must equal split seed')
    protected = (data['gold_ids'], data['accepted_ids'], checkpoint['ids'], selection['ids'])
    for i, ids in enumerate(protected):
        if len(ids.unique()) != len(ids) or any(bool(torch.isin(ids, other).any()) for other in protected[i+1:]):
            raise ValueError('Training/validation populations overlap')
    torch.set_num_threads(1)
    torch.manual_seed(cfg['seed'])
    torch.use_deterministic_algorithms(True)
    model = new_model(data, cfg)
    opt = torch.optim.AdamW([
        {'params':list(model.graph.parameters())+list(model.text.parameters()),'lr':cfg['lr']*cfg['projection_lr']},
        {'params':model.classifier.parameters(),'lr':cfg['lr']}], weight_decay=cfg['weight_decay'])
    # Dedicated streams pair gold draws across training recipes and keep their seed tied to outer.
    gold_rng = torch.Generator().manual_seed(cfg['seed'])
    pseudo_rng = torch.Generator().manual_seed(cfg['seed'])
    gold, targets, accepted = data['gold_ids'], data['gold_targets'], data['accepted_ids']
    start, history, best, state, stale = time.monotonic(), [], None, None, 0
    for step in range(cfg['max_steps'] + 1):
        if step:
            model.train()
            pos = torch.randint(len(gold), (cfg['batch_size'],), generator=gold_rng)
            g_ids = gold[pos]
            schedule = 1. if step > cfg['warmup'] else 0.
            if cfg['ramp']:
                schedule *= min(1., max(0., (step - cfg['warmup']) / cfg['ramp']))
            weight=cfg['pseudo_weight']*schedule
            kd_weight=cfg['kd_weight']*schedule
            p_ids = accepted[:0]
            if (weight or kd_weight) and len(accepted):
                p_ids = sample_pseudo(accepted, data['agreement'], cfg['batch_size'], cfg['pool_mode'], pseudo_rng)
            ids = torch.cat((g_ids, p_ids))
            logits = model(data['graph'][ids], data['text'][ids])
            gold_loss = F.cross_entropy(logits[:len(pos)], targets[pos])
            pseudo = pseudo_loss(logits[len(pos):], data['pseudo_labels'][p_ids],
                                 data['agreement'][p_ids], cfg['pool_mode'], cfg['smoothing'])
            kd=logits.sum()*0
            if kd_weight and len(p_ids):
                teacher=teacher_probability(data['logp_g'][p_ids],data['logp_l'][p_ids],
                    data['choose_g'][p_ids],data['agreement'][p_ids],cfg['teacher'],cfg['temperature'])
                kd=distillation_loss(logits[len(pos):],teacher,cfg['temperature'],data['agreement'][p_ids],cfg['pool_mode'])
            loss = gold_loss + weight * pseudo + kd_weight * kd
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise ValueError('Nonfinite gradient')
            opt.step()
        if step % cfg['validate_every'] == 0 or step == cfg['max_steps']:
            p = predict(model, data, checkpoint['ids'])
            row = {'step': step, **score(p, checkpoint['targets'])}
            history.append(row)
            if best is None or (-row['accuracy'], row['nll'], step) < (-best['accuracy'], best['nll'], best['step']):
                best, stale = row, 0
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if stale >= cfg['patience']:
                break
    model.load_state_dict(state)
    probability = predict(model, data, selection['ids'])
    return dict(config=cfg, state=state, checkpoint=best, history=history,
                validation_probability=probability, validation_ids=selection['ids'],
                selection=score(probability, selection['targets']), seed=data['seed'],
                test_truth_read=False, elapsed_seconds=time.monotonic()-start)
