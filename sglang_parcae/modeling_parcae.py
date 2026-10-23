"""Parcae for SGLang."""

from __future__ import annotations

import functools
import logging
import math
import os
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.layers.activation import ReLU2 as SGLReLU2
from sglang.srt.layers.layernorm import RMSNorm as SGLRMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader

from sglang_parcae.planes import plane, plane_count, validate_planes, ve_key

logger = logging.getLogger(__name__)

_FOLD_RELU2_SCALE = os.environ.get("SGLANG_PARCAE_FOLD_RELU2_SCALE", "1") == "1"
_FOLD_VE_GATE_SCALE = os.environ.get("SGLANG_PARCAE_FOLD_VE_GATE_SCALE", "1") == "1"
_FUSE_VE_ADD = os.environ.get("SGLANG_PARCAE_FUSE_VE_ADD", "1") == "1"
_HOIST_ADAPTER_INJECTION = os.environ.get("SGLANG_PARCAE_HOIST_ADAPTER_INJECTION", "1") == "1"


class ParcaeRMSNorm(SGLRMSNorm):
    """SGLang RMSNorm with an fp32 fallback. The fused flashinfer kernel behind forward_cuda is bf16/fp16 only."""

    _FUSED_DTYPES = (torch.bfloat16, torch.float16)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if x.dtype not in self._FUSED_DTYPES:
            return self.forward_native(x, *args, **kwargs)
        return super().forward(x, *args, **kwargs)


class Relu2(nn.Module):
    """relu(x)^2 * 0.5; the 0.5 is folded into the following weights when SGLANG_PARCAE_FOLD_RELU2_SCALE=1."""

    def __init__(self) -> None:
        super().__init__()
        self._relu2 = SGLReLU2()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._relu2(x)
        return out if _FOLD_RELU2_SCALE else out * 0.5


@triton.jit
def _mul_rn_bf16(x, y):
    return tl.inline_asm_elementwise(
        asm="mul.rn.bf16 $0, $1, $2;",
        constraints="=h,h,h",
        args=[x, y],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _add_rn_bf16(x, y):
    return tl.inline_asm_elementwise(
        asm="add.rn.bf16 $0, $1, $2;",
        constraints="=h,h,h",
        args=[x, y],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fused_ve_add_kernel(
    v_ptr,
    gate_ptr,
    ve_ptr,
    out_ptr,
    n_elements,
    v_row_stride: tl.constexpr,
    gate_row_stride: tl.constexpr,
    ve_row_stride: tl.constexpr,
    kv_size: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    rows = offsets // kv_size
    inner = offsets - rows * kv_size
    heads = inner // head_dim
    v = tl.load(v_ptr + rows * v_row_stride + inner, mask=mask)
    gate = tl.load(gate_ptr + rows * gate_row_stride + heads, mask=mask)
    ve = tl.load(ve_ptr + rows * ve_row_stride + inner, mask=mask)
    # Match eager bf16's separately rounded multiply followed by add.
    gated_ve = _mul_rn_bf16(gate, ve)
    tl.store(out_ptr + offsets, _add_rn_bf16(v, gated_ve), mask=mask)


@functools.lru_cache(maxsize=None)
def _has_bf16_ptx(device_index: int) -> bool:
    """The fused ve-add kernel uses bf16 PTX (mul/add.rn.bf16) that needs sm_90+."""
    ok = torch.cuda.get_device_capability(device_index) >= (9, 0)
    if not ok:
        logger.warning(
            "SGLANG_PARCAE_FUSE_VE_ADD: the fused ve-add kernel needs sm_90+ (cuda:%d is sm_%d%d); "
            "using the unfused path. Set SGLANG_PARCAE_FUSE_VE_ADD=0 to silence this.",
            device_index, *torch.cuda.get_device_capability(device_index),
        )
    return ok


def fused_ve_add(
    v: torch.Tensor,
    gate: torch.Tensor,
    ve: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """v + gate * ve per KV head, rounded like eager bf16. The bf16 path uses sm_90 PTX; other GPUs take the unfused path."""
    if not v.is_cuda or v.dtype != torch.bfloat16 or not _has_bf16_ptx(v.device.index):
        n_kv_head = gate.shape[-1]
        return (
            v.unflatten(-1, (n_kv_head, head_dim))
            + gate.unsqueeze(-1) * ve.unflatten(-1, (n_kv_head, head_dim))
        ).flatten(-2)
    out = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    n_elements = v.numel()
    _fused_ve_add_kernel[(triton.cdiv(n_elements, 256),)](
        v,
        gate,
        ve,
        out,
        n_elements,
        v_row_stride=v.stride(0),
        gate_row_stride=gate.stride(0),
        ve_row_stride=ve.stride(0),
        kv_size=v.shape[-1],
        head_dim=head_dim,
        BLOCK_SIZE=256,
    )
    return out


class ParcaeMLP(nn.Module):
    def __init__(self, dim: int, hidden: int, bias: bool, prefix: str = "") -> None:
        super().__init__()

        self.fc = ColumnParallelLinear(dim, hidden, bias=bias, quant_config=None,
                                       prefix=prefix + "fc")
        self.proj = RowParallelLinear(hidden, dim, bias=bias, quant_config=None,
                                      prefix=prefix + "proj")
        self.nonlin = Relu2()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.fc(x)
        out, _ = self.proj(self.nonlin(h))
        return out


class DiagonalInjection(nn.Module):
    """h_{t+1} = h_t * exp(-dt * exp(A_log)) + dt * (e @ B.T)"""

    def __init__(self, state_dim: int, input_dim: int) -> None:
        super().__init__()
        self.A_log = nn.Parameter(torch.empty(state_dim))
        self.dt_bias = nn.Parameter(torch.empty(state_dim))
        self.B = nn.Parameter(torch.empty(state_dim, input_dim))

        # Cache the computed dt and decay values, as they do not depend on h or e. Saves on kernels.
        self._dt: Optional[torch.Tensor] = None
        self._decay: Optional[torch.Tensor] = None

    def invalidate_cache(self) -> None:
        """Call after any write to A_log / dt_bias (i.e. after load_weights)."""
        self._dt = self._decay = None

    def _coefficients(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._dt is None:
            dt = F.softplus(self.dt_bias)
            self._dt = dt
            self._decay = torch.exp(-dt * torch.exp(self.A_log))
        return self._dt, self._decay

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        dt, decay = self._coefficients()
        return h * decay + dt * (e @ self.B.T)

    def prepare_injection(self, e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the depth-invariant decay and input injection once per token."""
        dt, decay = self._coefficients()
        return decay, dt * (e @ self.B.T)

    @staticmethod
    def apply_prepared(
        h: torch.Tensor,
        decay: torch.Tensor,
        injection: torch.Tensor,
    ) -> torch.Tensor:
        return h * decay + injection


class ParcaeAttention(nn.Module):
    """MHA with an optional value-embedding gate."""

    def __init__(self, config: PretrainedConfig, dim: int, n_head: int, n_kv_head: int,
                 has_ve_gate: bool, prefix: str = "") -> None:
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = dim // n_head
        self.qk_norm = bool(config.qk_norm)
        bias = bool(config.parcae.get("bias", False))

        self.q_size = n_head * self.head_dim
        self.kv_size = n_kv_head * self.head_dim
        self.ve_start = 0
        self.qkv_proj = QKVParallelLinear(
            hidden_size=dim, head_size=self.head_dim,
            total_num_heads=n_head, total_num_kv_heads=n_kv_head,
            bias=bias, quant_config=None, prefix=prefix + "qkv_proj",
        )
        self.n_head = self.qkv_proj.num_heads
        self.n_kv_head = self.qkv_proj.num_kv_heads
        self.q_size = self.qkv_proj.q_proj_shard_size
        self.kv_size = self.qkv_proj.kv_proj_shard_size
        self.ve_start = (
            self.qkv_proj.tp_rank // self.qkv_proj.num_kv_head_replicas
        ) * self.kv_size
        self.c_proj = RowParallelLinear(dim, dim, bias=bias, quant_config=None,
                                        prefix=prefix + "c_proj")

        self.qk_norm_op = (
            ParcaeRMSNorm(self.head_dim, eps=torch.finfo(torch.float32).eps) if self.qk_norm else None
        )
        self.ve_gate_channels = 32

        self.ve_gate = (
            ColumnParallelLinear(self.ve_gate_channels, n_kv_head, bias=False, quant_config=None,
                                 prefix=prefix + "ve_gate")
            if has_ve_gate else None
        )

    def forward(self, x: torch.Tensor, positions: torch.Tensor, forward_batch: ForwardBatch,
                attn: RadixAttention, rotary, ve: Optional[torch.Tensor]) -> torch.Tensor:
        qkv, _ = self.qkv_proj(x)

        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        if ve is not None and self.ve_gate is not None:
            ve = ve[..., self.ve_start:self.ve_start + self.kv_size]
            gate_raw, _ = self.ve_gate(x[..., : self.ve_gate_channels])
            gate = torch.sigmoid(gate_raw)
            if not _FOLD_VE_GATE_SCALE:
                gate = 2 * gate
            if _FUSE_VE_ADD:
                v = fused_ve_add(v, gate, ve, self.head_dim)
            else:
                v = (v.unflatten(-1, (self.n_kv_head, self.head_dim))
                     + gate.unsqueeze(-1) * ve.unflatten(-1, (self.n_kv_head, self.head_dim))
                     ).flatten(-2)

        q, k = rotary(positions, q, k)
        if self.qk_norm_op is not None:
            q = self.qk_norm_op(q.unflatten(-1, (self.n_head, self.head_dim))).flatten(-2)
            k = self.qk_norm_op(k.unflatten(-1, (self.n_kv_head, self.head_dim))).flatten(-2)

        v = v.contiguous()
        y = attn(q, k, v, forward_batch)
        out, _ = self.c_proj(y)
        return out


class ParcaeMLAAttention(nn.Module):
    """MLA with an optional value-embedding gate; K/V are expanded into a standard MHA KV cache."""

    def __init__(self, config: PretrainedConfig, dim: int, n_head: int,
                 has_ve_gate: bool, prefix: str = "") -> None:
        super().__init__()
        self.n_head = n_head
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        bias = bool(config.parcae.get("bias", False))
        assert not config.qk_norm, "MLA requires qk_norm=False"

        if self.q_lora_rank is None:
            self.c_q = ColumnParallelLinear(dim, n_head * self.qk_head_dim, bias=bias,
                                            quant_config=None, prefix=prefix + "c_q")
            self.q_a_proj = self.q_a_layernorm = self.q_b_proj = None
        else:
            self.c_q = None
            self.q_a_proj = ColumnParallelLinear(dim, self.q_lora_rank, bias=bias,
                                                 quant_config=None, prefix=prefix + "q_a_proj")
            self.q_a_layernorm = ParcaeRMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(self.q_lora_rank, n_head * self.qk_head_dim,
                                                 bias=False, quant_config=None,
                                                 prefix=prefix + "q_b_proj")

        self.kv_a_proj_with_mqa = ColumnParallelLinear(
            dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=bias, quant_config=None,
            prefix=prefix + "kv_a_proj_with_mqa")
        self.kv_a_layernorm = ParcaeRMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank, n_head * (self.qk_nope_head_dim + self.v_head_dim), bias=False,
            quant_config=None, prefix=prefix + "kv_b_proj")
        self.c_proj = RowParallelLinear(n_head * self.v_head_dim, dim, bias=bias,
                                        quant_config=None, prefix=prefix + "c_proj")

        self.ve_gate_channels = 32
        self.ve_gate = (
            ColumnParallelLinear(self.ve_gate_channels, n_head, bias=False, quant_config=None,
                                 prefix=prefix + "ve_gate")
            if has_ve_gate else None
        )

    def forward(self, x: torch.Tensor, positions: torch.Tensor, forward_batch: ForwardBatch,
                attn: RadixAttention, rotary, ve: Optional[torch.Tensor]) -> torch.Tensor:
        n_tok = x.shape[0]
        if self.c_q is not None:
            q, _ = self.c_q(x)
        else:
            a, _ = self.q_a_proj(x)
            q, _ = self.q_b_proj(self.q_a_layernorm(a))
        q = q.view(n_tok, self.n_head, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        kv_a, _ = self.kv_a_proj_with_mqa(x)
        c_kv, k_pe = kv_a.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        c_kv = self.kv_a_layernorm(c_kv)

        q_pe, k_pe = rotary(positions, q_pe.reshape(n_tok, -1), k_pe.reshape(n_tok, -1))
        q_pe = q_pe.view(n_tok, self.n_head, self.qk_rope_head_dim)
        k_pe = k_pe.view(n_tok, 1, self.qk_rope_head_dim)

        gate = None
        if self.ve_gate is not None:
            gate_raw, _ = self.ve_gate(x[..., : self.ve_gate_channels])
            gate = torch.sigmoid(gate_raw)
            if not _FOLD_VE_GATE_SCALE:
                gate = 2 * gate

        kv, _ = self.kv_b_proj(c_kv)
        kv = kv.view(n_tok, self.n_head, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        if gate is not None and ve is not None:
            v = v + gate.unsqueeze(-1) * ve.view(n_tok, self.n_head, self.v_head_dim)

        k = torch.cat([k_nope, k_pe.expand(-1, self.n_head, -1)], dim=-1)

        q = torch.cat([q_nope, q_pe], dim=-1)
        y = attn(q.reshape(n_tok, -1), k.reshape(n_tok, -1), v.reshape(n_tok, -1), forward_batch)
        out, _ = self.c_proj(y.reshape(n_tok, self.n_head * self.v_head_dim))
        return out


class ParcaeBlock(nn.Module):
    def __init__(self, config: PretrainedConfig, dim: int, hidden: int, n_head: int, n_kv_head: int,
                 has_ve_gate: bool) -> None:
        super().__init__()
        eps = config.rms_norm_eps
        bias = bool(config.parcae.get("bias", False))

        self.norm_1 = ParcaeRMSNorm(dim, eps=eps)
        self.attn = (
            ParcaeMLAAttention(config, dim, n_head, has_ve_gate)
            if getattr(config, "is_mla", False)
            else ParcaeAttention(config, dim, n_head, n_kv_head, has_ve_gate)
        )
        self.norm_2 = ParcaeRMSNorm(dim, eps=eps)
        self.mlp = ParcaeMLP(dim, hidden, bias)

    def forward(self, x, residual, positions, forward_batch, attn_slot, rotary, ve):
        if residual is None:
            residual = x
            x = self.norm_1(x)
        else:
            x, residual = self.norm_1(x, residual)
        x = self.attn(x, positions, forward_batch, attn_slot, rotary, ve)
        x, residual = self.norm_2(x, residual)
        x = self.mlp(x)
        return x, residual


class ParcaeForCausalLM(nn.Module):
    """Parcae causal LM for SGLang."""

    def __init__(self, config: PretrainedConfig, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        self.n_prelude = config.n_layers_in_prelude
        self.n_core = config.n_layers_in_recurrent_block
        self.n_coda = config.n_layers_in_coda
        self.strategy = config.lookup_strategy
        self.t_max = config.recurrence_depth_max
        self.t_default = config.recurrence_depth_default
        self.hoist_core_ve = os.environ.get("SGLANG_PARCAE_HOIST_CORE_VE", "1") == "1"

        if config.attention_class_name == "KDACausalSelfAttention":
            raise NotImplementedError("KDA is not supported by the SGLang backend yet.")
        self._assert_supported_parallelism(quant_config)
        self._assert_supported_state_init(config.state_init)
        validate_planes(self.strategy, self.t_max, self.n_prelude, self.n_core, self.n_coda)
        declared = plane_count(self.strategy, self.t_max, self.n_prelude, self.n_core, self.n_coda)
        assert declared == config.num_hidden_layers, (
            f"num_hidden_layers={config.num_hidden_layers} disagrees with the plane map ({declared}). "
            f"The KV pool is sized from the config, so a mismatch mis-addresses KV silently."
        )

        dim = config.hidden_size
        rec_dim = config.recurrent_embedding_dimension
        n_head = config.num_attention_heads
        n_kv_head = config.num_key_value_heads

        self.ve_keys = set(self._expected_ve_keys())

        if config.is_mla:
            ve_dim = n_head * config.v_head_dim
        else:
            ve_dim = n_kv_head * (dim // n_head)
        self.ve_dim = ve_dim

        self.wte = VocabParallelEmbedding(config.vocab_size, dim, quant_config=None)
        self.value_embeds = nn.ModuleDict(
            {k: VocabParallelEmbedding(config.vocab_size, ve_dim, quant_config=None)
             for k in sorted(self.ve_keys)}
        )

        self.prelude = nn.ModuleList([
            ParcaeBlock(config, dim, config.intermediate_size, n_head, n_kv_head, self._has_ve_gate("prelude", i))
            for i in range(self.n_prelude)
        ])
        self.ln_prelude = ParcaeRMSNorm(dim, eps=config.rms_norm_eps) if config.prelude_norm else None

        self.adapter = DiagonalInjection(rec_dim, dim)
        rec_hidden = config.recurrent_intermediation_embedding_dimension
        self.core_block = nn.ModuleList([
            ParcaeBlock(config, rec_dim, rec_hidden, n_head, n_kv_head, self._has_ve_gate("core", i))
            for i in range(self.n_core)
        ])
        self.C = ColumnParallelLinear(
            rec_dim, dim, bias=False, gather_output=True, quant_config=None, prefix="C"
        )
        self.coda = nn.ModuleList([
            ParcaeBlock(config, dim, config.intermediate_size, n_head, n_kv_head, self._has_ve_gate("coda", i))
            for i in range(self.n_coda)
        ])
        self.ln_f = ParcaeRMSNorm(dim, eps=config.rms_norm_eps)
        self.lm_head = ParallelLMHead(config.vocab_size, dim, quant_config=None)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.wte.weight

        self.logits_processor = LogitsProcessor(config, logit_scale=config.logit_scale)

        rope_dim = (config.qk_rope_head_dim if getattr(config, "is_mla", False)
                    else dim // n_head)
        self.rotary = get_rope(
            head_size=rope_dim,
            rotary_dim=rope_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            is_neox_style=False,
        )
        local_attn = self.core_block[0].attn
        self._build_attention_planes(
            local_attn.n_head,
            getattr(local_attn, "n_kv_head", local_attn.n_head),
            dim // n_head,
        )

    # Set SGLANG_PARCAE_ALLOW_UNVERIFIED_TP=1 to run TP>1, which has not been tested.
    _TP_OVERRIDE_ENV = "SGLANG_PARCAE_ALLOW_UNVERIFIED_TP"

    @classmethod
    def _assert_supported_parallelism(cls, quant_config) -> None:
        """Raise for parallelism and quantization modes that have not been tested."""

        def size(attr: str) -> int:
            """tp_size / pp_size, or 1 when there is no distributed context."""
            from sglang.srt.runtime_context import get_parallel

            try:
                return int(getattr(get_parallel(), attr))
            except Exception:
                return 1

        tp, pp = size("tp_size"), size("pp_size")

        if tp > 1 and not os.environ.get(cls._TP_OVERRIDE_ENV):
            raise NotImplementedError(
                f"tensor parallelism (tp_size={tp}) has not been tested. Set "
                f"{cls._TP_OVERRIDE_ENV}=1 to run it anyway, and check parity against TP=1."
            )
        if pp > 1:
            raise NotImplementedError(
                f"pipeline parallelism (pp_size={pp}) is not supported: SGLang partitions by layer "
                f"index, but this model's layers are recurrence planes over shared core weights."
            )
        if quant_config is not None:
            raise NotImplementedError(
                f"quantization ({type(quant_config).__name__}) has not been tested with Parcae."
            )

    def _expected_ve_keys(self) -> list[str]:
        from sglang_parcae.planes import has_ve

        n_eff = self.config.n_effective_layers
        return [str(i) for i in range(n_eff) if has_ve(i, n_eff)]

    def _has_ve_gate(self, kind: str, layer: int) -> bool:
        """Whether this layer has a ve_gate.

        The reference decides ve tables with has_ve(effective_idx, n_effective_layers) and ve gates
        with has_ve(layer_id, cfg.n_layer). The two agree for every released checkpoint but can differ
        for odd layer counts, so the gate set is read from `ve_gate_layers`, which build_model_dir
        computes with the reference's own rule.
        """
        gates = getattr(self.config, "ve_gate_layers", None)
        if gates is None:
            raise RuntimeError(
                "config has no `ve_gate_layers`. Rebuild the shim with build_model_dir -- guessing "
                "it from the ve tables is wrong whenever n_prelude/n_core/n_coda are odd."
            )
        return f"{kind}.{layer}" in set(gates)

    def _ve_for(self, kind: str, layer: int, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        key = ve_key(kind, layer, n_prelude=self.n_prelude, n_core=self.n_core)
        return self.value_embeds[key](input_ids) if key in self.value_embeds else None

    def _build_attention_planes(self, n_head: int, n_kv_head: int, head_dim: int) -> None:
        addr = dict(strategy=self.strategy, t_max=self.t_max,
                    n_prelude=self.n_prelude, n_core=self.n_core)
        cfg = self.config
        v_head_dim = -1
        if getattr(cfg, "is_mla", False):
            head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
            v_head_dim = cfg.v_head_dim
            n_kv_head = n_head

        scaling = head_dim ** -0.5

        def slot(layer_id: int, name: str) -> RadixAttention:
            return RadixAttention(num_heads=n_head, head_dim=head_dim, scaling=scaling,
                                  num_kv_heads=n_kv_head, layer_id=layer_id,
                                  v_head_dim=v_head_dim,
                                  quant_config=self.quant_config, prefix=name)

        self.prelude_attn = nn.ModuleList(
            [slot(plane("prelude", i, **addr), f"prelude.{i}") for i in range(self.n_prelude)]
        )
        self.core_attn = nn.ModuleDict()
        n_depths = self.t_max if self.strategy == "full" else 1
        for d in range(n_depths):
            for l in range(self.n_core):
                self.core_attn[f"d{d}_l{l}"] = slot(plane("core", l, depth=d, **addr), f"core.d{d}.l{l}")
        self.coda_attn = nn.ModuleList(
            [slot(plane("coda", i, **addr), f"coda.{i}") for i in range(self.n_coda)]
        )

    def _core_slot(self, depth: int, layer: int) -> RadixAttention:
        return self.core_attn[f"d{depth if self.strategy == 'full' else 0}_l{layer}"]

    SUPPORTED_STATE_INIT = frozenset({"normal", "embed", "like-init", "zero", "unit"})

    @classmethod
    def _assert_supported_state_init(cls, mode: str) -> None:
        if mode in cls.SUPPORTED_STATE_INIT:
            return
        raise NotImplementedError(
            f"unknown state_init={mode!r}; supported: {sorted(cls.SUPPORTED_STATE_INIT)}"
        )

    def _draw_h0(self, shape, device, dtype) -> torch.Tensor:
        """One h_0 draw."""
        cfg = self.config
        mode = cfg.state_init

        if mode == "zero":
            return torch.zeros(shape, device=device, dtype=dtype)

        x = torch.randn(shape, device=device, dtype=dtype)

        if mode == "normal":
            return x
        if mode == "embed":
            return x.mul(1 / math.sqrt(cfg.hidden_size))
        if mode == "unit":
            std, mean = torch.std_mean(x, dim=-1, keepdim=True)
            return (x - mean) / (std + 1e-8)
        if mode == "like-init":
            std = cfg.state_init_std
            nn.init.trunc_normal_(x, mean=0.0, std=std, a=-3 * std, b=3 * std)
            if cfg.embedding_scale != 1:
                x = x * cfg.embedding_scale
            return x
        raise AssertionError(f"unhandled state_init {mode!r}")  # validated in __init__

    def initialize_state(self, e: torch.Tensor) -> torch.Tensor:
        """Initialize the recurrent state h_0 with one draw from the global RNG, matching forward_for_generation."""
        rec_dim = self.config.recurrent_embedding_dimension
        return self._draw_h0((e.shape[0], rec_dim), e.device, e.dtype)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                forward_batch: ForwardBatch, input_embeds: Optional[torch.Tensor] = None,
                num_steps: Optional[int] = None):
        hidden = self.forward_trunk(input_ids, positions, forward_batch, input_embeds, num_steps)
        return self.logits_processor(input_ids, hidden, self.lm_head, forward_batch)

    def forward_trunk(self, input_ids: torch.Tensor, positions: torch.Tensor,
                      forward_batch: ForwardBatch, input_embeds: Optional[torch.Tensor] = None,
                      num_steps: Optional[int] = None):
        """Everything up to and including ln_f; mirrors forward_for_generation in receval/models/parcae.py but stops before the head."""
        cfg = self.config
        t = num_steps if num_steps is not None else self.t_default
        if not 1 <= t <= self.t_max:
            raise ValueError(f"num_steps={t} outside [1, t_max={self.t_max}]")

        x = input_embeds if input_embeds is not None else self.wte(input_ids)
        if cfg.embedding_scale != 1.0:
            x = x * cfg.embedding_scale

        residual = None
        for i, block in enumerate(self.prelude):
            x, residual = block(x, residual, positions, forward_batch, self.prelude_attn[i],
                                self.rotary, self._ve_for("prelude", i, input_ids))
        x = self.ln_prelude(x, residual)[0] if self.ln_prelude is not None else x + residual

        e = x                                   # Injection, re-fed at every recurrence step.
        h = self.initialize_state(e)
        core_ve = (
            [self._ve_for("core", l, input_ids) for l in range(self.n_core)]
            if self.hoist_core_ve
            else None
        )
        prepared_injection = (
            self.adapter.prepare_injection(e) if _HOIST_ADAPTER_INJECTION else None
        )
        for depth in range(t):
            if prepared_injection is None:
                h = self.adapter(h, e)
            else:
                h = self.adapter.apply_prepared(h, *prepared_injection)
            h_residual = None
            for l, block in enumerate(self.core_block):
                ve = core_ve[l] if core_ve is not None else self._ve_for("core", l, input_ids)
                h, h_residual = block(h, h_residual, positions, forward_batch,
                                      self._core_slot(depth, l), self.rotary, ve)
            h = h + h_residual

        x, _ = self.C(h)
        residual = None
        for i, block in enumerate(self.coda):
            x, residual = block(x, residual, positions, forward_batch, self.coda_attn[i],
                                self.rotary, self._ve_for("coda", i, input_ids))
        return self.ln_f(x, residual)[0]         # ln_f absorbs the last residual add

    _RENAMES = (
        ("transformer.wte.", "wte."),
        ("transformer.ln_prelude.", "ln_prelude."),
        ("transformer.ln_f.", "ln_f."),
        ("transformer.adapter.", "adapter."),
        ("transformer.C.", "C."),
        ("transformer.prelude.", "prelude."),
        ("transformer.core_block.", "core_block."),
        ("transformer.coda.", "coda."),
    )

    # In the checkpoint but deliberately not loaded: a precomputed RoPE table of shape
    # (1, block_size, 1, head_dim/2, 2). We regenerate rotary embeddings via SGLang's get_rope, so
    # this is expected-absent rather than a mismatch.
    _IGNORED_CHECKPOINT_TENSORS = frozenset({"freqs_cis"})

    # Parameters this model owns that the checkpoint does not contain. qk-norm is parameterless in
    # Parcae; we implement it with a ones-weight RMSNorm purely to use the fused kernel, so its
    # weight must stay at its ones initialisation and must not trip the strict load check.
    _NON_CHECKPOINT_PARAM_SUFFIXES = (".qk_norm_op.weight",)

    # Parcae stores q/k/v separately; QKVParallelLinear holds one fused parameter. Map each source
    # tensor onto the fused param with a shard id, the way llama does.
    _STACKED_PARAMS = (
        (".qkv_proj", ".c_q", "q"),
        (".qkv_proj", ".c_k", "k"),
        (".qkv_proj", ".c_v", "v"),
    )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights from a checkpoint."""
        params = dict(self.named_parameters(remove_duplicate=False))
        loaded, skipped = set(), []
        qkv_shards: dict[str, set] = {}
        for name, w in weights:
            if name in self._IGNORED_CHECKPOINT_TENSORS:
                continue
            for src, dst in self._RENAMES:
                if name.startswith(src):
                    name = dst + name[len(src):]
                    break
            # Fused q/k/v: rewrite the name onto the fused param and pass the shard id. The
            # fused param is only "loaded" once all three shards have landed.
            for fused, src, shard in self._STACKED_PARAMS:
                if src in name:
                    target = name.replace(src, fused)
                    if target in params:
                        params[target].weight_loader(params[target], w, shard)
                        qkv_shards.setdefault(target, set()).add(shard)
                        break
            else:
                if name not in params:
                    skipped.append(name)
                    continue
                p = params[name]
                # Every Relu2 feeds exactly one row-parallel projection. Since 0.5 is a
                # power-of-two scale, folding it into that projection's bf16 weight preserves
                # the products and accumulation order while removing an elementwise kernel.
                if _FOLD_RELU2_SCALE and name.endswith(".mlp.proj.weight"):
                    w = w * 0.5
                # Value embeddings have one consumer, where they are multiplied by
                # 2*sigmoid(gate). Moving the exact power-of-two factor into the embedding
                # table removes the separate gate scaling kernel.
                if _FOLD_VE_GATE_SCALE and (
                    name.startswith("value_embeds.") and name.endswith(".weight")
                ):
                    w = w * 2.0
                getattr(p, "weight_loader", default_weight_loader)(p, w)
                loaded.add(name)

        for target, shards in qkv_shards.items():
            if shards != {"q", "k", "v"}:
                raise RuntimeError(f"{target}: got shards {sorted(shards)}, expected ['k', 'q', 'v']")
            loaded.add(target)

        expected = {n for n in params
                    if not n.endswith(self._NON_CHECKPOINT_PARAM_SUFFIXES)}
        if self.config.tie_word_embeddings:
            expected -= {"wte.weight", "lm_head.weight"}
            if not ({"wte.weight", "lm_head.weight"} & loaded):
                raise RuntimeError("neither wte.weight nor lm_head.weight was loaded")

        self.adapter.invalidate_cache()

        missing = sorted(expected - loaded)
        if missing or skipped:
            raise RuntimeError(
                f"weight load mismatch -- loading with strict=False would have hidden this.\n"
                f"  in model but not in checkpoint ({len(missing)}): {missing[:8]}\n"
                f"  in checkpoint but not in model ({len(skipped)}): {skipped[:8]}"
            )


class ParcaeMLAForCausalLM(ParcaeForCausalLM):
    """MLA Parcae. K/V are expanded into an ordinary MHA cache.

    Not declared as AttentionArch.MLA to SGLang: the shim emits MHA-shaped dims
    (num_key_value_heads = n_head, head_dim = qk_nope + qk_rope, v_head_dim = v_head_dim), so it
    needs neither an entry in SGLang's MLA architecture list nor a custom KV pool.
    """


EntryClass = [ParcaeForCausalLM, ParcaeMLAForCausalLM]
