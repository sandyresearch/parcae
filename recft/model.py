from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class EndpointCaptureCache:
    """Pass teacher-forced K/V through directly while the forward reports step indices."""

    def __init__(
        self,
        model: nn.Module,
        num_steps: int,
        *,
        per_sequence: bool = False,
    ) -> None:
        self.model = model
        n_prelude = len(model.transformer.prelude)
        self.n_core = len(model.transformer.core_block)
        self.core_start = n_prelude
        # Open-ended for per-sequence Poisson depths; fixed path keeps a bound.
        self.core_end = (
            None
            if per_sequence
            else n_prelude + num_steps * self.n_core
        )
        self.num_steps = num_steps
        self.per_sequence = per_sequence

    @property
    def core_step_range(self) -> tuple[int, int | None]:
        return self.core_start, self.core_end

    def sequence_totals(self) -> torch.Tensor | None:
        if not self.per_sequence:
            return None
        n = getattr(self.model, "_current_n_per_sample", None)
        k = getattr(self.model, "_current_k_per_sample", None)
        if n is None or k is None:
            return None
        return n + k

    def get_seq_length(self) -> int:
        return 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        step_idx_tensor: torch.Tensor,
        lookup_strategy: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del step_idx_tensor, lookup_strategy
        return key_states, value_states

    def update_latent_scalar(
        self,
        latent_states: torch.Tensor,
        step_idx_tensor: torch.Tensor,
        token_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pass teacher-forced MLA latents and their tokens through directly."""
        del step_idx_tensor
        if token_ids is None:
            token_ids = getattr(self.model, "_current_input_ids", None)
        if token_ids is None:
            raise ValueError("MLA endpoint capture requires current input token IDs.")
        return latent_states, token_ids

    def get_kda_state(self, step_idx_tensor: torch.Tensor) -> None:
        del step_idx_tensor
        return None

    def update_kda_state(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs


def configure_trainable_parameters(model: nn.Module, mode: str, lora=None) -> list[str]:
    if mode == "all":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    elif mode == "freeze_prelude_input":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        for parameter in model.transformer.wte.parameters():
            parameter.requires_grad_(False)
        for parameter in model.transformer.prelude.parameters():
            parameter.requires_grad_(False)
        prelude_norm = getattr(model.transformer, "ln_prelude", None)
        if prelude_norm is not None:
            for parameter in prelude_norm.parameters():
                parameter.requires_grad_(False)
        for layer_idx in range(len(model.transformer.prelude)):
            key = str(layer_idx)
            if key in model.value_embeds:
                for parameter in model.value_embeds[key].parameters():
                    parameter.requires_grad_(False)
    elif mode == "lora":
        from recft.lora import attach_lora

        if lora is None:
            raise ValueError("trainable=lora requires a LoRA config")
        attach_lora(model, layers=[0], rank=lora.rank, alpha=lora.alpha, targets=lora.targets)
    else:
        raise ValueError(f"unknown trainable mode: {mode}")
    return [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]


def block0_endpoints(
    model: nn.Module,
    input_ids: torch.Tensor,
    num_steps: int,
    *,
    want_cache_swap: bool = False,
    swap_detach_hidden: bool = True,
    grad_loops: set[int] | None = None,
    per_sequence: bool = False,
) -> tuple[
    torch.Tensor,
    dict[int, dict[int, tuple[Any, ...]]],
    torch.Tensor | None,
    torch.Tensor | None,
]:
    cache = EndpointCaptureCache(model, num_steps, per_sequence=per_sequence)
    core_start, core_end = cache.core_step_range
    n_core = cache.n_core
    endpoints = {0, num_steps - 1}
    rec_inputs: dict[int, dict[int, tuple[Any, ...]]] = {}
    core_x0_box: list[torch.Tensor | None] = [None]

    handles = []

    def loop_of(step_idx: Any, layer_idx: int = 0) -> int | None:
        if step_idx is None:
            return None
        step_idx = int(step_idx)
        if step_idx < core_start:
            return None
        if core_end is not None and step_idx >= core_end:
            return None
        relative = step_idx - core_start
        if relative % n_core == layer_idx:
            return relative // n_core
        return None

    if want_cache_swap:
        def capture_core_x0(module, args, kwargs):
            del module
            recurrence = loop_of(kwargs.get("step_idx"), 0)
            if recurrence != 0:
                return
            hidden = args[0]
            if swap_detach_hidden:
                hidden = hidden.detach()
            core_x0_box[0] = hidden

        handles.append(
            model.transformer.core_block[0].register_forward_pre_hook(
                capture_core_x0,
                with_kwargs=True,
            )
        )

        for layer_idx in range(n_core):
            layer_inputs: dict[int, tuple[Any, ...]] = {}
            rec_inputs[layer_idx] = layer_inputs

            def capture_cache_swap_inputs(
                module,
                args,
                kwargs,
                *,
                layer_idx=layer_idx,
                layer_inputs=layer_inputs,
            ):
                del module
                recurrence = loop_of(kwargs.get("step_idx"), layer_idx)
                if recurrence is None:
                    return
                if per_sequence:
                    totals = cache.sequence_totals()
                    if totals is None:
                        return
                    is_endpoint = (recurrence == 0) | (recurrence == (totals - 1))
                    if not bool(is_endpoint.any()):
                        return
                elif recurrence not in endpoints:
                    return
                hidden = args[0]
                ve = kwargs.get("ve")
                if swap_detach_hidden:
                    hidden = hidden.detach()
                    if ve is not None:
                        ve = ve.detach()
                layer_inputs[recurrence] = (
                    hidden,
                    args[1] if len(args) > 1 else None,
                    ve,
                )
            handles.append(
                model.transformer.core_block[layer_idx].attn.register_forward_pre_hook(
                    capture_cache_swap_inputs,
                    with_kwargs=True,
                )
            )

    try:
        if per_sequence:
            # Exact pretrain forward: sampler + iterate_forward + coda + head.
            output = model(
                input_ids,
                return_logits=True,
                past_key_values=cache,
            )
            logits = output["logits"]
        else:
            output = model.forward_for_generation(
                input_ids,
                num_steps=num_steps,
                past_key_values=cache,
                grad_loops=grad_loops,
            )
            logits = output["logits"]
    finally:
        for handle in handles:
            handle.remove()

    sequence_totals = cache.sequence_totals()
    core_x0 = core_x0_box[0]
    return logits, rec_inputs, core_x0, sequence_totals


def lm_ce(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        input_ids[:, 1:].reshape(-1),
    )


def _recompute_qkv(
    attention: nn.Module,
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    ve: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from parcae_lm.modules.mixer import norm
    from parcae_lm.modules.utils import apply_rotary_emb_complex_like

    batch, sequence, _ = x.shape
    if hasattr(attention, "kv_a_proj_with_mqa"):
        if attention.q_lora_rank is None:
            q = attention.c_q(x)
        else:
            q = attention.q_b_proj(
                attention.q_a_layernorm(attention.q_a_proj(x))
            )
        q = q.view(batch, sequence, attention.n_head, attention.qk_head_dim)
        q_nope, q_pe = torch.split(
            q,
            [attention.qk_nope_head_dim, attention.qk_rope_head_dim],
            dim=-1,
        )
        c_kv, k_pe = torch.split(
            attention.kv_a_proj_with_mqa(x),
            [attention.kv_lora_rank, attention.qk_rope_head_dim],
            dim=-1,
        )
        c_kv = attention.kv_a_layernorm(c_kv)
        k_pe = k_pe.view(
            batch, sequence, 1, attention.qk_rope_head_dim
        )
        if attention.config.rope_settings.use_rope:
            q_pe, k_pe = apply_rotary_emb_complex_like(
                q_pe, k_pe, freqs_cis=freqs_cis
            )
        kv = attention.kv_b_proj(c_kv).view(
            batch,
            sequence,
            attention.n_head,
            attention.qk_nope_head_dim + attention.v_head_dim,
        )
        k_nope, v = torch.split(
            kv,
            [attention.qk_nope_head_dim, attention.v_head_dim],
            dim=-1,
        )
        if ve is not None and attention.ve_gate is not None:
            ve = ve.view(
                batch, sequence, attention.n_head, attention.v_head_dim
            )
            gate = 2 * torch.sigmoid(
                attention.ve_gate(x[..., : attention.ve_gate_channels])
            )
            v = v + gate.unsqueeze(-1) * ve
        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat(
            [k_nope, k_pe.expand(-1, -1, attention.n_head, -1)],
            dim=-1,
        )
        return q, k, v

    q = attention.c_q(x).view(
        batch, sequence, attention.n_head, attention.head_dim
    )
    k = attention.c_k(x).view(
        batch, sequence, attention.n_kv_head, attention.head_dim
    )
    v = attention.c_v(x).view(
        batch, sequence, attention.n_kv_head, attention.head_dim
    )
    if ve is not None and attention.ve_gate is not None:
        ve = ve.view(
            batch, sequence, attention.n_kv_head, attention.head_dim
        )
        gate = 2 * torch.sigmoid(
            attention.ve_gate(x[..., : attention.ve_gate_channels])
        )
        v = v + gate.unsqueeze(-1) * ve
    if attention.config.clip_qkv is not None:
        clip = attention.config.clip_qkv
        q = q.clamp(min=-clip, max=clip)
        k = k.clamp(min=-clip, max=clip)
        v = v.clamp(min=-clip, max=clip)
    if attention.config.qk_bias:
        q_bias, k_bias = attention.qk_bias.split(1, dim=0)
        q = (q + q_bias).to(q.dtype)
        k = (k + k_bias).to(q.dtype)
    if attention.config.rope_settings.use_rope:
        q, k = apply_rotary_emb_complex_like(
            q,
            k,
            freqs_cis=freqs_cis,
        )
    if attention.config.qk_norm:
        q, k = norm(q), norm(k)
    return q, k, v


def _repeat_kv_heads(
    values: torch.Tensor,
    n_head: int,
) -> torch.Tensor:
    if values.shape[2] != n_head:
        values = values.repeat_interleave(
            n_head // values.shape[2],
            dim=2,
        )
    return values


def _swapped_attn_output(
    attention: nn.Module,
    x_q: torch.Tensor,
    freqs_q: torch.Tensor,
    ve_q: torch.Tensor | None,
    x_kv: torch.Tensor,
    freqs_kv: torch.Tensor,
    ve_kv: torch.Tensor | None,
) -> torch.Tensor:
    if hasattr(attention, "_chunk_kda"):
        return attention(
            x_q, freqs_q, ve=ve_q, state_source=x_kv, state_ve=ve_kv
        )

    q, _, _ = _recompute_qkv(attention, x_q, freqs_q, ve_q)
    _, k, v = _recompute_qkv(attention, x_kv, freqs_kv, ve_kv)
    k = _repeat_kv_heads(k, attention.n_head)
    v = _repeat_kv_heads(v, attention.n_head)
    y = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=0.0,
        is_causal=True,
    )
    batch, sequence = q.shape[0], q.shape[1]
    y = y.transpose(1, 2).reshape(batch, sequence, -1)
    return attention.c_proj(y)


def cache_swap_ce(
    model: nn.Module,
    input_ids: torch.Tensor,
    rec_inputs: dict[int, dict[int, tuple[Any, ...]]],
    core_x0: torch.Tensor | None,
    detach_hidden: bool,
) -> torch.Tensor:
    """CE of one core pass with Q/residual @ R=0 and KV @ R=T (all layers)."""
    if core_x0 is None:
        raise ValueError("cache_swap_ce requires captured core_x0")
    if not rec_inputs:
        raise ValueError("cache_swap_ce requires per-layer rec_inputs")

    x = core_x0.detach() if detach_hidden else core_x0
    for layer_idx in sorted(rec_inputs):
        block = model.transformer.core_block[layer_idx]
        layer_inputs = rec_inputs[layer_idx]
        first = min(layer_inputs)
        final = max(layer_inputs)
        _x0, freqs0, ve0 = layer_inputs[first]
        xT, freqsT, veT = layer_inputs[final]
        del _x0
        x_q = block.norm_1(x)
        freqs_q = freqs0 if freqs0 is not None else freqsT
        ve_q = ve0.detach() if (detach_hidden and ve0 is not None) else ve0
        x_kv = xT.detach()
        freqs_kv = freqsT.detach()
        ve_kv = veT.detach() if veT is not None else None
        attn_out = _swapped_attn_output(
            block.attn,
            x_q,
            freqs_q.detach(),
            ve_q,
            x_kv,
            freqs_kv,
            ve_kv,
        )
        x = attn_out + x
        x = block.mlp(block.norm_2(x)) + x

    model._current_input_ids = input_ids
    x = model.transformer.C(x)
    n_prelude = len(model.transformer.prelude)
    n_core = len(model.transformer.core_block)
    coda_base = n_prelude + n_core  # readout as if recurrence depth 1
    coda_ve_offset = n_prelude + n_core
    freqs_cis = model.freqs_cis[:, : input_ids.shape[1]]
    for i, block in enumerate(model.transformer.coda):
        key = str(coda_ve_offset + i)
        ve_embed = model.value_embeds[key] if key in model.value_embeds else None
        ve = ve_embed(input_ids) if ve_embed is not None else None
        x = block(
            x,
            freqs_cis,
            None,
            step_idx=torch.tensor(coda_base + i, dtype=torch.long),
            ve=ve,
        )
    x = model.transformer.ln_f(x)
    if model.config.use_fused_head == "full-triton":
        weight = model.lm_head.weight.T if model.config.tie_embeddings else model.lm_head.weight
        logits = torch.matmul(x, weight).float() * model.config.init.logit_scale
    else:
        logits = model.lm_head(x).float() * model.config.init.logit_scale
    softcap = getattr(model.config, "logit_softcap", None)
    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)
    return lm_ce(logits, input_ids)


class RecFTModule(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        *,
        num_steps: int,
        aux: str,
        gamma: float,
        grad_loops: set[int] | None,
        precision: str,
        swap_detach_hidden: bool = True,
        recurrence_sampling: str = "fixed",
        lockstep_n: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.num_steps = num_steps
        self.aux = aux
        self.gamma = gamma
        self.grad_loops = grad_loops
        self.precision = precision
        self.swap_detach_hidden = swap_detach_hidden
        self.recurrence_sampling = recurrence_sampling
        self.last_num_steps = num_steps
        if recurrence_sampling == "poisson":
            scheme = getattr(model.config, "sampling_scheme", "")
            method = getattr(model, "recurrent_iteration_method", None)
            if "poisson-truncated-full" not in scheme:
                raise ValueError(
                    "recurrence_sampling=poisson requires model "
                    "sampling_scheme containing poisson-truncated-full, "
                    f"got {scheme!r}"
                )
            if method != "per-sequence":
                raise ValueError(
                    "recurrence_sampling=poisson requires "
                    f"recurrent_iteration_method='per-sequence', got {method!r}"
                )
            # Pretraining uses independent rank-local depth vectors by default.
            # Lockstep remains available for explicit reproducibility diagnostics.
            model.config.lockstep_n = lockstep_n

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        per_sequence = self.recurrence_sampling == "poisson"
        grad_loops = self.grad_loops
        num_steps = self.num_steps

        if per_sequence:
            # Pretrain path owns sampling + truncated BPTT via iterate_forward.
            self.model.step = int(getattr(self.model, "step", 0)) + 1
            grad_loops = None

        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.precision == "bf16"
            else torch.autocast(device_type="cuda", enabled=False)
        )
        with context:
            logits, rec_inputs, core_x0, sequence_totals = block0_endpoints(
                self.model,
                input_ids,
                num_steps,
                want_cache_swap=self.aux == "swap_ce",
                swap_detach_hidden=self.swap_detach_hidden,
                grad_loops=grad_loops,
                per_sequence=per_sequence,
            )
            if per_sequence and sequence_totals is not None:
                self.last_num_steps = float(sequence_totals.float().mean().item())
            ce = lm_ce(logits, input_ids)
            if self.aux == "none" or (not per_sequence and num_steps < 2):
                aux = torch.zeros(
                    (),
                    device=logits.device,
                    dtype=torch.float32,
                )
            elif self.aux == "swap_ce":
                aux = cache_swap_ce(
                    self.model,
                    input_ids,
                    rec_inputs,
                    core_x0,
                    self.swap_detach_hidden,
                )
            else:
                raise ValueError(f"unknown aux: {self.aux}")
            total = ce + self.gamma * aux
        return total, ce, aux
