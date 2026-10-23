from __future__ import annotations

import torch


def pinned_layout_preserving_saved_tensors():
    """Offload saved CUDA tensors while retaining dense tensor strides."""

    def pack(tensor: torch.Tensor):
        device = tensor.device
        if device.type != "cuda":
            return device, tensor
        if tensor.layout != torch.strided or any(value < 0 for value in tensor.stride()):
            return device, tensor.detach().cpu()
        try:
            packed = torch.empty_strided(
                tensor.size(), tensor.stride(), dtype=tensor.dtype,
                device="cpu", pin_memory=True,
            )
            packed.copy_(tensor)
        except RuntimeError:
            packed = tensor.detach().cpu()
        return device, packed

    def unpack(value):
        device, tensor = value
        return tensor.to(device, non_blocking=bool(tensor.is_pinned()))

    return torch.autograd.graph.saved_tensors_hooks(pack, unpack)
