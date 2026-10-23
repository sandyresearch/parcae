"""LoRA adapters on the core block's cache-writing projections.

The update is `(alpha / rank) B A x` with no bias term. `B` is zero-initialised, so the wrapped model
starts exactly as the base model.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """`W x + (alpha / r) B A x`, with `W` frozen."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        d_out, d_in = base.weight.shape
        self.rank = rank
        self.scale = alpha / rank
        self.A = nn.Parameter(torch.randn(rank, d_in) / math.sqrt(d_in))
        self.B = nn.Parameter(torch.zeros(d_out, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The adapter runs outside autocast, in fp32. Autocast caches the bf16 cast of a weight for the
        # whole autocast region; with truncated backprop the first cast happens under no_grad, and later
        # recurrences would reuse a cast with no graph, so the loss would not depend on the adapter.
        out = self.base(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            delta = F.linear(F.linear(x.float(), self.A.float()), self.B.float())
        return out + (self.scale * delta).to(out.dtype)

    # Parcae reads `.weight` for shapes in a few places.
    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def merged_weight(self) -> torch.Tensor:
        """W + (alpha/r) B A, for exporting a plain checkpoint with the LoRA folded in."""
        return self.base.weight + self.scale * (self.B @ self.A)


# Projections that write the shared cache, per attention class.
DEFAULT_TARGETS = {
    "CausalSelfAttention": ("c_k", "c_v"),
    "MLACausalSelfAttention": ("kv_a_proj_with_mqa", "kv_b_proj"),
    "KDACausalSelfAttention": ("k_proj", "v_proj"),
}


def attach_lora(model, layers: Sequence[int], rank: int = 8, alpha: float = 16.0,
                targets: Sequence[str] | None = None) -> tuple[list[nn.Parameter], list[str]]:
    """Wrap `targets` on the given core layers and freeze everything else. Returns (params, names).

    `targets=None` resolves to the attention class's cache-writing projections.
    """
    core = model.transformer.core_block
    for p in model.parameters():
        p.requires_grad_(False)
    params: list[nn.Parameter] = []
    names: list[str] = []
    for layer in layers:
        if not 0 <= layer < len(core):
            raise ValueError(f"core layer {layer} outside 0..{len(core) - 1}")
        attn = core[layer].attn
        cls = type(attn).__name__
        if targets:
            layer_targets = tuple(targets)
        elif cls in DEFAULT_TARGETS:
            layer_targets = DEFAULT_TARGETS[cls]
        else:
            raise NotImplementedError(f"no default LoRA targets for {cls}; set lora.targets")
        for target in layer_targets:
            base = getattr(attn, target, None)
            if base is None:
                raise ValueError(f"core[{layer}] has no `{target}`")
            if isinstance(base, LoRALinear):
                raise RuntimeError(f"core[{layer}].{target} is already wrapped")
            wrapped = LoRALinear(base, rank, alpha).to(base.weight.device, base.weight.dtype)
            setattr(attn, target, wrapped)
            params += [wrapped.A, wrapped.B]
            names += [f"core[{layer}].{target}"]
    return params, names


def lora_modules(model) -> list[tuple[str, LoRALinear]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)]


def merged_state_dict(model) -> dict[str, torch.Tensor]:
    """The model's state dict with every adapter folded into its base weight, under plain key names."""
    state = model.state_dict()
    for name, mod in lora_modules(model):
        prefix = name + "."
        for key in (prefix + "A", prefix + "B", prefix + "base.weight", prefix + "base.bias"):
            state.pop(key, None)
        state[prefix + "weight"] = mod.merged_weight().detach()
        if mod.base.bias is not None:
            state[prefix + "bias"] = mod.base.bias.detach()
    return state


def _set_submodule(model, dotted: str, value) -> None:
    parent = model
    parts = dotted.split(".")
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    setattr(parent, parts[-1], value)


def unwrap_lora(model, merge: bool = True):
    """Restore plain `Linear` modules, optionally folding the adapter into the weight.

    A wrapped model's state dict names the base weight `...c_k.base.weight`, and `from_pretrained`
    loads non-strictly, so saving a wrapped model would reload with randomly initialised projections.
    Call this before `save_pretrained`.
    """
    for name, mod in lora_modules(model):
        base = mod.base
        if merge:
            with torch.no_grad():
                base.weight.copy_(mod.merged_weight().to(base.weight.dtype))
        base.requires_grad_(False)
        _set_submodule(model, name, base)
    return model
