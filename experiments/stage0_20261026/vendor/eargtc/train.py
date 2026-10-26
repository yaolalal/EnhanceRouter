from __future__ import annotations

from copy import deepcopy
from typing import Optional

import torch
import torch.nn.functional as F


def _classifier_is_better(select_by: str, val_loss: float, val_acc: float, best_val_loss: float, best_val_acc: float) -> bool:
    if select_by == "val_loss":
        return val_loss < best_val_loss
    if select_by == "val_acc":
        return val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss)
    raise ValueError(f"Unknown select_by: {select_by}")


def train_classifier(
    model: torch.nn.Module,
    features,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: Optional[torch.Tensor],
    epochs: int,
    lr: float,
    weight_decay: float,
    class_weights: Optional[torch.Tensor] = None,
    patience: Optional[int] = None,
    select_by: str = "val_loss",
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_val_acc = -1.0
    wait = 0
    train_idx = train_idx.long()
    val_idx = None if val_idx is None or val_idx.numel() == 0 else val_idx.long()
    class_weights = None if class_weights is None else class_weights.to(device=features.device, dtype=torch.float32)

    for _ in range(max(epochs, 1)):
        model.train()
        optimizer.zero_grad()
        logits = model(features)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx], weight=class_weights)
        loss.backward()
        optimizer.step()

        if val_idx is not None:
            model.eval()
            with torch.no_grad():
                val_logits = model(features)[val_idx]
                val_labels = labels[val_idx]
                val_loss = F.cross_entropy(val_logits, val_labels).item()
                val_acc = float((val_logits.argmax(dim=-1) == val_labels).float().mean().item())
            if _classifier_is_better(str(select_by), val_loss, val_acc, best_val_loss, best_val_acc):
                best_val_loss = float(val_loss)
                best_val_acc = float(val_acc)
                best_state = deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
            if patience is not None and int(patience) > 0 and wait >= int(patience):
                break

    if val_idx is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(features)




def _trainable_state_dict(model: torch.nn.Module):
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def _load_trainable_state_dict(model: torch.nn.Module, state):
    params = dict(model.named_parameters())
    for name, value in state.items():
        if name in params:
            params[name].data.copy_(value.to(device=params[name].device, dtype=params[name].dtype))


def _auto_joint_batch_size(requested: int, num_classes: int, *, eval_mode: bool = False) -> int:
    if requested > 0:
        return int(requested)
    if num_classes <= 8:
        return 8 if eval_mode else 4
    if num_classes <= 20:
        return 4 if eval_mode else 2
    return 1


def joint_llm_logits_for_indices(model, token_ids, texts, idx: torch.Tensor, batch_size: int, gnn_embeddings=None):
    rows = []
    idx = idx.long().detach().cpu()
    for start in range(0, idx.numel(), max(batch_size, 1)):
        batch_idx = idx[start : start + max(batch_size, 1)]
        batch_texts = [texts[int(i)] for i in batch_idx.tolist()]
        batch_tokens = token_ids[batch_idx.to(token_ids.device)]
        batch_embeddings = None if gnn_embeddings is None else gnn_embeddings[batch_idx.to(gnn_embeddings.device)]
        rows.append(model(batch_tokens, batch_texts, batch_embeddings).detach().cpu())
    return torch.cat(rows, dim=0) if rows else torch.empty(0)


_joint_llm_logits_for_indices = joint_llm_logits_for_indices


def _weighted_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, sample_weights: Optional[torch.Tensor] = None):
    losses = F.cross_entropy(logits, labels.to(logits.device), reduction="none")
    if sample_weights is None:
        return losses.mean()
    weights = sample_weights.to(logits.device, dtype=losses.dtype).clamp_min(0.0)
    return (losses * weights).sum() / weights.sum().clamp_min(1e-12)


def _joint_distillation_loss(student_logits, teacher_logits_list, temperature: float):
    valid_teachers = [teacher for teacher in teacher_logits_list if teacher is not None]
    if not valid_teachers:
        return student_logits.new_tensor(0.0)
    temp = max(float(temperature), 1e-6)
    teacher_probs = [F.softmax(teacher.to(student_logits.device).float() / temp, dim=-1) for teacher in valid_teachers]
    teacher_prob = torch.stack(teacher_probs, dim=0).mean(dim=0)
    student_log_prob = F.log_softmax(student_logits.float() / temp, dim=-1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (temp * temp)


def _joint_loss_and_logit_grad(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    gnn_teacher: Optional[torch.Tensor],
    llm_teacher: Optional[torch.Tensor],
    ce_weight: float,
    distill_weight: float,
    distill_temperature: float,
    sample_weights: Optional[torch.Tensor] = None,
):
    logits = student_logits.detach().float().requires_grad_(True)
    ce_loss = _weighted_cross_entropy(logits, labels, sample_weights)
    kd_loss = _joint_distillation_loss(logits, [gnn_teacher, llm_teacher], distill_temperature)
    loss = float(ce_weight) * ce_loss + float(distill_weight) * kd_loss
    (grad,) = torch.autograd.grad(loss, logits)
    return loss.detach(), grad.detach()


def _memory_efficient_joint_backward(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    texts,
    labels: torch.Tensor,
    batch_idx: torch.Tensor,
    gnn_embeddings,
    gnn_teacher_logits: Optional[torch.Tensor],
    llm_teacher_logits: Optional[torch.Tensor],
    ce_weight: float,
    distill_weight: float,
    distill_temperature: float,
    joint_sample_weights: Optional[torch.Tensor] = None,
):
    batch_cpu = batch_idx.detach().cpu()
    batch_texts = [texts[int(i)] for i in batch_cpu.tolist()]
    batch_tokens = token_ids[batch_idx.to(token_ids.device)]
    batch_embeddings = None if gnn_embeddings is None else gnn_embeddings[batch_idx.to(gnn_embeddings.device)]
    with torch.no_grad():
        logits = model(batch_tokens, batch_texts, batch_embeddings)
    gnn_teacher = None if gnn_teacher_logits is None else gnn_teacher_logits[batch_idx.to(gnn_teacher_logits.device)]
    llm_teacher = None if llm_teacher_logits is None else llm_teacher_logits[batch_idx.to(llm_teacher_logits.device)]
    batch_weights = None if joint_sample_weights is None else joint_sample_weights[batch_idx.to(joint_sample_weights.device)]
    loss, logit_grad = _joint_loss_and_logit_grad(
        logits,
        labels[batch_idx],
        gnn_teacher,
        llm_teacher,
        ce_weight,
        distill_weight,
        distill_temperature,
        batch_weights,
    )
    num_classes = logits.size(-1)
    label_batch_size = max(int(getattr(model, "label_pair_batch_size", 1)), 1)
    for row_i, node_id in enumerate(batch_cpu.tolist()):
        token = token_ids[int(node_id)].to(token_ids.device)
        text = texts[int(node_id)]
        embedding = None if gnn_embeddings is None else gnn_embeddings[int(node_id)].to(gnn_embeddings.device)
        for start in range(0, num_classes, label_batch_size):
            label_indices = torch.arange(start, min(start + label_batch_size, num_classes), dtype=torch.long, device=token_ids.device)
            scores = model.score_label_indices(token, text, label_indices, embedding, label_batch_size=label_batch_size)
            grad = logit_grad[row_i, label_indices.to(logit_grad.device)].to(scores.device)
            torch.autograd.backward(scores, grad_tensors=grad)
    return loss


def train_joint_llm_model(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    texts,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: Optional[torch.Tensor],
    eval_idx: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int = 0,
    eval_batch_size: int = 0,
    val_every: int = 1,
    gnn_embeddings=None,
    gnn_teacher_logits: Optional[torch.Tensor] = None,
    llm_teacher_logits: Optional[torch.Tensor] = None,
    ce_weight: float = 1.0,
    distill_weight: float = 0.2,
    distill_temperature: float = 2.0,
    memory_efficient_backward: bool = True,
    joint_sample_weights: Optional[torch.Tensor] = None,
):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("Joint LLM model has no trainable parameters.")
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    best_state = _trainable_state_dict(model)
    best_val = float("inf")
    train_idx = train_idx.long()
    eval_idx = eval_idx.long()
    val_idx = None if val_idx is None or val_idx.numel() == 0 else val_idx.long()
    num_classes = len(model.label_names)
    train_batch_size = _auto_joint_batch_size(batch_size, num_classes, eval_mode=False)
    val_batch_size = _auto_joint_batch_size(eval_batch_size, num_classes, eval_mode=True)
    log_every = 100
    total_epochs = max(epochs, 1)
    print(
        f"[joint] train_nodes={train_idx.numel()} val_nodes={0 if val_idx is None else val_idx.numel()} "
        f"eval_nodes={eval_idx.numel()} classes={num_classes} epochs={total_epochs} "
        f"train_batch_size={train_batch_size} eval_batch_size={val_batch_size} "
        f"memory_efficient_backward={bool(memory_efficient_backward)}",
        flush=True,
    )

    for epoch in range(total_epochs):
        model.train()
        perm = train_idx[torch.randperm(train_idx.numel(), device=train_idx.device)]
        train_batches = (perm.numel() + train_batch_size - 1) // train_batch_size
        print(f"[joint] epoch {epoch + 1}/{total_epochs} train_start batches={train_batches}", flush=True)
        for batch_no, start in enumerate(range(0, perm.numel(), train_batch_size), start=1):
            batch_idx = perm[start : start + train_batch_size]
            batch_texts = [texts[int(i)] for i in batch_idx.detach().cpu().tolist()]
            batch_embeddings = None if gnn_embeddings is None else gnn_embeddings[batch_idx.to(gnn_embeddings.device)]
            optimizer.zero_grad(set_to_none=True)
            if memory_efficient_backward and hasattr(model, "score_label_indices") and getattr(model, "scoring_mode", "full") == "full":
                if hasattr(model, "dropout"):
                    model.dropout.eval()
                loss = _memory_efficient_joint_backward(
                    model,
                    token_ids,
                    texts,
                    labels,
                    batch_idx,
                    gnn_embeddings,
                    gnn_teacher_logits,
                    llm_teacher_logits,
                    ce_weight,
                    distill_weight,
                    distill_temperature,
                    joint_sample_weights,
                )
            else:
                logits = model(token_ids[batch_idx], batch_texts, batch_embeddings)
                batch_weights = None if joint_sample_weights is None else joint_sample_weights[batch_idx.to(joint_sample_weights.device)]
                ce_loss = _weighted_cross_entropy(logits, labels[batch_idx], batch_weights)
                gnn_teacher = None if gnn_teacher_logits is None else gnn_teacher_logits[batch_idx.to(gnn_teacher_logits.device)]
                llm_teacher = None if llm_teacher_logits is None else llm_teacher_logits[batch_idx.to(llm_teacher_logits.device)]
                kd_loss = _joint_distillation_loss(logits, [gnn_teacher, llm_teacher], distill_temperature)
                loss = float(ce_weight) * ce_loss + float(distill_weight) * kd_loss
                loss.backward()
            optimizer.step()
            if batch_no == 1 or batch_no % log_every == 0 or batch_no == train_batches:
                print(f"[joint] epoch {epoch + 1}/{total_epochs} train {batch_no}/{train_batches} loss={float(loss.detach().cpu()):.6f}", flush=True)

        if val_idx is not None and ((epoch + 1) % max(val_every, 1) == 0):
            model.eval()
            losses = []
            val_batches = (val_idx.numel() + val_batch_size - 1) // val_batch_size
            print(f"[joint] epoch {epoch + 1}/{total_epochs} val_start batches={val_batches}", flush=True)
            with torch.no_grad():
                for batch_no, start in enumerate(range(0, val_idx.numel(), val_batch_size), start=1):
                    batch_idx = val_idx[start : start + val_batch_size]
                    batch_texts = [texts[int(i)] for i in batch_idx.detach().cpu().tolist()]
                    batch_embeddings = None if gnn_embeddings is None else gnn_embeddings[batch_idx.to(gnn_embeddings.device)]
                    logits = model(token_ids[batch_idx], batch_texts, batch_embeddings)
                    losses.append(F.cross_entropy(logits, labels[batch_idx].to(logits.device)).detach().cpu())
                    if batch_no == 1 or batch_no % log_every == 0 or batch_no == val_batches:
                        print(f"[joint] epoch {epoch + 1}/{total_epochs} val {batch_no}/{val_batches}", flush=True)
            val_loss = float(torch.stack(losses).mean().item()) if losses else 0.0
            print(f"[joint] epoch {epoch + 1}/{total_epochs} val_loss={val_loss:.6f}", flush=True)
            if val_loss < best_val:
                best_val = val_loss
                best_state = _trainable_state_dict(model)
                print(f"[joint] epoch {epoch + 1}/{total_epochs} best_val_updated", flush=True)

    if val_idx is not None:
        _load_trainable_state_dict(model, best_state)
    model.eval()
    all_logits = torch.zeros(labels.size(0), num_classes, dtype=torch.float32)
    print(f"[joint] eval_start nodes={eval_idx.numel()} batch_size={val_batch_size}", flush=True)
    rows = []
    eval_cpu = eval_idx.long().detach().cpu()
    eval_batches = (eval_cpu.numel() + val_batch_size - 1) // val_batch_size
    with torch.no_grad():
        for batch_no, start in enumerate(range(0, eval_cpu.numel(), max(val_batch_size, 1)), start=1):
            batch_idx = eval_cpu[start : start + max(val_batch_size, 1)]
            batch_texts = [texts[int(i)] for i in batch_idx.tolist()]
            batch_tokens = token_ids[batch_idx.to(token_ids.device)]
            batch_embeddings = None if gnn_embeddings is None else gnn_embeddings[batch_idx.to(gnn_embeddings.device)]
            rows.append(model(batch_tokens, batch_texts, batch_embeddings).detach().cpu())
            if batch_no == 1 or batch_no % log_every == 0 or batch_no == eval_batches:
                print(f"[joint] eval {batch_no}/{eval_batches}", flush=True)
    logits = torch.cat(rows, dim=0) if rows else torch.empty(0)
    all_logits[eval_idx.detach().cpu()] = logits
    print("[joint] eval_done", flush=True)
    return all_logits.to(labels.device)
