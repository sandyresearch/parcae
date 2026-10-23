import math
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from parcae_lm.models.config import AnyConfig

from parcae_lm.modules.utils import apply_rotary_emb_complex_like
from parcae_lm.attention_backends import flash_attn


def norm(x: Tensor) -> Tensor:
    """Purely functional rmsnorm with no learnable params."""
    return torch.nn.functional.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx: int, n_layer: int) -> bool:
    """Returns True if layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


class CausalSelfAttention(torch.nn.Module):
    def __init__(self, config: "AnyConfig", layer_id: int) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.n_head = config.num_attention_heads
        self.n_kv_head = config.num_key_value_heads
        self.head_dim = config.n_embd // self.n_head
        self.n_rep = self.n_head // self.n_kv_head

        assert config.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        self.c_q = config.Linear(config.n_embd, self.n_head * self.head_dim, bias=config.bias, init_method=config.init.fn("q", layer_id))
        self.c_k = config.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias, init_method=config.init.fn("k", layer_id))
        self.c_v = config.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias, init_method=config.init.fn("v", layer_id))
        self.c_proj = config.Linear(config.n_embd, config.n_embd, bias=config.bias, init_method=config.init.fn("out_attn", layer_id))

        if config.qk_bias:
            self.qk_bias = torch.nn.Parameter(torch.zeros(2, 1, self.n_head, self.head_dim))
        self.ve_gate_channels = 32
        self.ve_gate = (
            torch.nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_id, config.n_layer) else None
        )
        self.monitoring = False
        self.latest_metrics = {}

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        ve = kwargs.get("ve")
        if ve is not None and self.ve_gate is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 2)
            v = v + gate.unsqueeze(-1) * ve
        if self.config.clip_qkv is not None:
            q = q.clamp(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            k = k.clamp(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            v = v.clamp(min=-self.config.clip_qkv, max=self.config.clip_qkv)
        if self.config.qk_bias:
            q_bias, k_bias = self.qk_bias.split(1, dim=0)
            q = (q + q_bias).to(q.dtype)
            k = (k + k_bias).to(q.dtype)
        if self.config.rope_settings.use_rope:
            q, k = apply_rotary_emb_complex_like(q, k, freqs_cis=freqs_cis)
        if self.config.qk_norm:
            q, k = norm(q), norm(k)

        window_size = kwargs.get("window_size", (-1, -1))
        kv_cache = kwargs.get("kv_cache")
        past_key_values = kwargs.get("past_key_values")
        step_idx = kwargs.get("step_idx")

        if kv_cache is not None:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_id)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            if self.layer_id == kv_cache.n_layers - 1:
                kv_cache.advance(T)
        elif past_key_values is not None and step_idx is not None:
            # ParcaeDynamicCache: cache expects (B, H, S, D) layout
            k_for_cache = k.transpose(1, 2)
            v_for_cache = v.transpose(1, 2)
            k_cached, v_cached = past_key_values.update(k_for_cache, v_for_cache, step_idx)
            k_full = k_cached.transpose(1, 2)
            v_full = v_cached.transpose(1, 2)
            y = flash_attn.flash_attn_func(q, k_full, v_full, causal=True, window_size=window_size)
        else:
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

        y = y.contiguous().view(B, T, C)
        return self.c_proj(y)

    @staticmethod
    def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
        """Repeat K/V heads for GQA: torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
        if n_rep == 1:
            return x
        B, T, H, D = x.shape
        return x.unsqueeze(3).expand(B, T, H, n_rep, D).reshape(B, T, H * n_rep, D)

    @torch.no_grad()
    def monitor_layer(self, q, k, mask):
        """Casting metric computations into low precision because fusion op can be unreliable"""
        S = q.shape[1]
        if mask is None:
            attn_mask_tril = torch.ones([S, S], dtype=torch.bool, device=q.device).tril()
            attn_mask = torch.zeros_like(attn_mask_tril).to(q)
            attn_mask = attn_mask_tril.masked_fill(~attn_mask_tril, -10000)
        else:
            attn_mask = mask
        q = q.half().transpose(1, 2)  # (B, nh, S, hs)
        k = k.half().transpose(1, 2)
        A = ((q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)) + attn_mask).half()
        max_attn_logit = A.max()
        A = torch.softmax(A, dim=-1)  # overwrite A immediately
        if self.config.center_attention:
            A = A + torch.eye(S, device=A.device, dtype=A.dtype)[None, None, :, :]
        if self.config.debias_attention:
            mask_matrix = torch.ones([S, S], dtype=torch.bool, device=q.device).tril()
            A = A - mask_matrix / mask_matrix.sum(dim=1, keepdim=True)
        attn_entropy = 1 / S * torch.where(A > 0, -A.float() * A.float().log(), 0).sum(dim=-1).sum(dim=-1).mean()
        metrics = {f"attn_entropy_{self.layer_id}": attn_entropy, f"max_attn_logit_{self.layer_id}": max_attn_logit}
        self.latest_metrics = metrics  # will be picked up from monitoring caller


class MLACausalSelfAttention(torch.nn.Module):
    """Multi-head Latent Attention (DeepSeek-V3 style) with causal masking.

    Head geometry follows DeepSeek-V3: `qk_head_dim = qk_nope + qk_rope` != `v_head_dim`.
    """
    def __init__(self, config: "AnyConfig", layer_id: int) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.n_head = config.num_attention_heads

        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        assert config.mup_model_scaling_factor == 1, "MLA head dims do not scale with mup"
        assert not config.qk_bias, "qk_bias is not supported by MLA"
        assert config.clip_qkv is None, "clip_qkv is not supported by MLA"
        assert not config.qk_norm, (
            "qk_norm RMS-norms whole q/k heads post-RoPE.  MLA takes that stability from the latent "
            "norms (kv_a_layernorm / q_a_layernorm) instead, and a post-RoPE head norm would also "
            "block folding W_UK into W_Q for absorbed decode.  Set qk_norm=False on MLA configs."
        )

        if self.q_lora_rank is None:
            self.c_q = config.Linear(config.n_embd, self.n_head * self.qk_head_dim, bias=config.bias, init_method=config.init.fn("q", layer_id))
        else:
            self.q_a_proj = config.Linear(config.n_embd, self.q_lora_rank, bias=config.bias, init_method=config.init.fn("q_a", layer_id))
            self.q_a_layernorm = config.Norm(self.q_lora_rank, eps=config.norm_eps)
            self.q_b_proj = config.Linear(self.q_lora_rank, self.n_head * self.qk_head_dim, bias=False, init_method=config.init.fn("q_b", layer_id))

        self.kv_a_proj_with_mqa = config.Linear(config.n_embd, self.kv_lora_rank + self.qk_rope_head_dim, bias=config.bias, init_method=config.init.fn("kv_a", layer_id))
        self.kv_a_layernorm = config.Norm(self.kv_lora_rank, eps=config.norm_eps)
        self.kv_b_proj = config.Linear(self.kv_lora_rank, self.n_head * (self.qk_nope_head_dim + self.v_head_dim), bias=False, init_method=config.init.fn("kv_b", layer_id))

        self.c_proj = config.Linear(self.n_head * self.v_head_dim, config.n_embd, bias=config.bias, init_method=config.init.fn("out_attn", layer_id))

        # Value embeddings on alternating layers, as in MHA: V += 2 * sigmoid(ve_gate(x[..., :32])) * ve.
        self.ve_gate_channels = 32
        self.ve_gate = (
            torch.nn.Linear(self.ve_gate_channels, self.n_head, bias=False)
            if has_ve(layer_id, config.n_layer) else None
        )
        self.monitoring = False
        self.latest_metrics = {}

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        B, T, C = x.size()

        if self.q_lora_rank is None:
            q = self.c_q(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.view(B, T, self.n_head, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        c_kv, k_pe = torch.split(self.kv_a_proj_with_mqa(x), [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        c_kv = self.kv_a_layernorm(c_kv)
        k_pe = k_pe.view(B, T, 1, self.qk_rope_head_dim)

        if self.config.rope_settings.use_rope:
            q_pe, k_pe = apply_rotary_emb_complex_like(q_pe, k_pe, freqs_cis=freqs_cis)

        q = torch.cat([q_nope, q_pe], dim=-1)                     # (B,T,n_head,qk_head_dim)

        # Per-head value-embedding gate, (B, T, n_head) or None.
        gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels])) \
            if self.ve_gate is not None else None

        window_size = kwargs.get("window_size", (-1, -1))
        past_key_values = kwargs.get("past_key_values")
        step_idx = kwargs.get("step_idx")
        if kwargs.get("kv_cache") is not None:
            raise NotImplementedError("MLACausalSelfAttention does not support kv_cache; use past_key_values.")

        if past_key_values is not None and step_idx is not None:
            # Scalar eval/generation path: retain MLA's compact latent cache and
            # reuse ParcaeDynamicCache's standard full/compress-s1 slot mapping.
            g_slot = gate if gate is not None else x.new_zeros(B, T, self.n_head)
            latent = torch.cat([c_kv, k_pe[:, :, 0, :], g_slot], dim=-1)
            latent_full, token_ids = past_key_values.update_latent_scalar(
                latent,
                step_idx,
                getattr(past_key_values, "_current_input_ids", None),
            )
            sequence_length = latent_full.shape[1]
            rope_end = self.kv_lora_rank + self.qk_rope_head_dim
            c_kv_full = latent_full[..., : self.kv_lora_rank]
            k_pe_full = latent_full[..., self.kv_lora_rank : rope_end]
            g_full = latent_full[..., rope_end :]
            kv_full = self.kv_b_proj(c_kv_full).view(
                B, sequence_length, self.n_head, self.qk_nope_head_dim + self.v_head_dim
            )
            k_nope_full, v_full = torch.split(
                kv_full, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
            )
            ve_table = kwargs.get("ve_table")
            if ve_table is not None and self.ve_gate is not None:
                ve_full = ve_table(token_ids).view(
                    B, sequence_length, self.n_head, self.v_head_dim
                )
                v_full = v_full + g_full.unsqueeze(-1) * ve_full
            k_full = torch.cat(
                [k_nope_full, k_pe_full.unsqueeze(2).expand(-1, -1, self.n_head, -1)],
                dim=-1,
            )
            y = flash_attn.flash_attn_func(
                q, k_full, v_full, causal=True, window_size=window_size
            )
        else:
            kv = self.kv_b_proj(c_kv).view(B, T, self.n_head, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            ve = kwargs.get("ve")
            if gate is not None and ve is not None:
                v = v + gate.unsqueeze(-1) * ve.view(B, T, self.n_head, self.v_head_dim)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_head, -1)], dim=-1)
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

        y = y.reshape(B, T, self.n_head * self.v_head_dim)
        return self.c_proj(y)

    @torch.no_grad()
    def monitor_layer(self, q, k, mask):
        """Casting metric computations into low precision because fusion op can be unreliable"""
        S = q.shape[1]
        if mask is None:
            attn_mask_tril = torch.ones([S, S], dtype=torch.bool, device=q.device).tril()
            attn_mask = torch.zeros_like(attn_mask_tril).to(q)
            attn_mask = attn_mask_tril.masked_fill(~attn_mask_tril, -10000)
        else:
            attn_mask = mask
        q = q.half().transpose(1, 2)  # (B, nh, S, hs)
        k = k.half().transpose(1, 2)
        A = ((q @ k.transpose(-2, -1) / math.sqrt(self.qk_head_dim)) + attn_mask).half()
        max_attn_logit = A.max()
        A = torch.softmax(A, dim=-1)  # overwrite A immediately
        if self.config.center_attention:
            A = A + torch.eye(S, device=A.device, dtype=A.dtype)[None, None, :, :]
        if self.config.debias_attention:
            mask_matrix = torch.ones([S, S], dtype=torch.bool, device=q.device).tril()
            A = A - mask_matrix / mask_matrix.sum(dim=1, keepdim=True)
        attn_entropy = 1 / S * torch.where(A > 0, -A.float() * A.float().log(), 0).sum(dim=-1).sum(dim=-1).mean()
        metrics = {f"attn_entropy_{self.layer_id}": attn_entropy, f"max_attn_logit_{self.layer_id}": max_attn_logit}
        self.latest_metrics = metrics  # will be picked up from monitoring caller


class KDACausalSelfAttention(torch.nn.Module):
    """Parcae adapter for FLA's Kimi Delta Attention."""

    def __init__(self, config: "AnyConfig", layer_id: int) -> None:
        super().__init__()
        try:
            from fla.modules import FusedRMSNormGated, ShortConvolution  # type: ignore[import-not-found]
            from parcae_lm.kda_backends import select_kda_implementation
        except ImportError as exc:
            raise ImportError(
                "KDACausalSelfAttention requires flash-linear-attention (`fla`) with KDA support."
            ) from exc

        self.config = config
        self.layer_id = layer_id
        self.mode = config.kda_mode
        self.disable_recompute = bool(config.kda_disable_recompute)
        self.hidden_size = config.n_embd
        self.head_dim = int(config.kda_head_dim)
        self.num_heads = config.num_attention_heads
        self.num_v_heads = int(config.kda_num_v_heads or self.num_heads)
        self.expand_v = float(config.kda_expand_v)
        self.use_short_conv = bool(config.kda_use_short_conv)
        self.allow_neg_eigval = bool(config.kda_allow_neg_eigval)
        self.safe_gate = bool(config.kda_safe_gate)
        self.lower_bound = config.kda_lower_bound
        self.conv_size = int(config.kda_conv_size)
        self.conv_bias = bool(config.kda_conv_bias)
        self._chunk_kda = select_kda_implementation(config.kda_impl)

        if self.mode != "chunk":
            raise ValueError(f"Unsupported KDA mode {self.mode!r}; only 'chunk' is supported.")
        if self.num_v_heads < self.num_heads or self.num_v_heads % self.num_heads != 0:
            raise ValueError(
                "KDA num_v_heads must equal or be a multiple of num_attention_heads."
            )

        self.head_k_dim = self.head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        if not math.isclose(self.head_dim * self.expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(
                f"kda_expand_v={self.expand_v} does not produce an integer value-head dimension."
            )
        self.key_dim = self.num_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.gate_dim = self.num_v_heads * self.head_k_dim

        linear = config.Linear
        init = config.init
        self.q_proj = linear(
            self.hidden_size, self.key_dim, bias=False,
            init_method=init.fn("q", layer_id),
        )
        self.k_proj = linear(
            self.hidden_size, self.key_dim, bias=False,
            init_method=init.fn("k", layer_id),
        )
        self.v_proj = linear(
            self.hidden_size, self.value_dim, bias=False,
            init_method=init.fn("v", layer_id),
        )

        if self.use_short_conv:
            self.q_conv1d = ShortConvolution(
                self.key_dim, self.conv_size, bias=self.conv_bias, activation="silu",
            )
            self.k_conv1d = ShortConvolution(
                self.key_dim, self.conv_size, bias=self.conv_bias, activation="silu",
            )
            self.v_conv1d = ShortConvolution(
                self.value_dim, self.conv_size, bias=self.conv_bias, activation="silu",
            )

        self.f_proj = torch.nn.Sequential(
            linear(
                self.hidden_size, self.head_v_dim, bias=False,
                init_method=init.fn("in_proj", layer_id),
            ),
            linear(
                self.head_v_dim, self.gate_dim, bias=False,
                init_method=init.fn("in_proj", layer_id),
            ),
        )
        self.b_proj = linear(
            self.hidden_size, self.num_v_heads, bias=False,
            init_method=init.fn("in_proj", layer_id),
        )

        if self.safe_gate:
            A = torch.zeros(self.num_v_heads, dtype=torch.float32)
        else:
            A = torch.log(torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(1, 16))
        self.A_log = torch.nn.Parameter(A)
        setattr(self.A_log, "_no_weight_decay", True)
        dt = torch.exp(
            torch.rand(self.gate_dim, dtype=torch.float32)
            * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp_min_(1e-4)
        self.dt_bias = torch.nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        setattr(self.dt_bias, "_no_weight_decay", True)

        self.g_proj = torch.nn.Sequential(
            linear(
                self.hidden_size, self.head_v_dim, bias=False,
                init_method=init.fn("in_proj", layer_id),
            ),
            linear(
                self.head_v_dim, self.value_dim, bias=True,
                init_method=init.fn("in_proj", layer_id),
            ),
        )
        self.o_norm = FusedRMSNormGated(
            self.head_v_dim, activation="sigmoid", eps=config.norm_eps,
        )
        self.o_proj = linear(
            self.value_dim, self.hidden_size, bias=False,
            init_method=init.fn("out_attn", layer_id),
        )

        # Value embeddings on alternating layers, as in MHA.
        self.ve_head_dim = self.hidden_size // self.num_heads
        self.ve_gate_channels = 32
        self.ve_proj = self.ve_gate = None
        if has_ve(layer_id, config.n_layer):
            self.ve_proj = linear(
                self.ve_head_dim, self.head_v_dim, bias=False,
                init_method=init.fn("v", layer_id),
            )
            self.ve_gate = torch.nn.Linear(
                self.ve_gate_channels, self.num_v_heads, bias=False,
            )

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """By default the state is built from `x`. If `state_source` is given, k, v, the decay and beta
        come from it instead (with `state_ve` as its value embedding), while q and the output gate still use `x`."""
        del freqs_cis  # KDA encodes position through its recurrent decay.
        attention_mask = mask if mask is not None and mask.ndim == 2 else None
        if kwargs.get("kv_cache") is not None:
            raise NotImplementedError("KDA does not support GPTKVCache, use past_key_values")
        past_key_values = kwargs.get("past_key_values")
        step_idx = kwargs.get("step_idx")
        state_source = kwargs.get("state_source", x)
        if "state_source" in kwargs and past_key_values is not None:
            raise ValueError("state_source cannot be combined with past_key_values.")
        last_state = None
        if past_key_values is not None:
            if step_idx is None:
                raise ValueError("KDA caching requires step_idx.")
            last_state = past_key_values.get_kda_state(step_idx)

        B, T, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(state_source)
        v = self.v_proj(state_source)

        ve = kwargs.get("state_ve", kwargs.get("ve"))
        if ve is not None:
            if self.ve_proj is None or self.ve_gate is None:
                raise ValueError(
                    f"KDA layer {self.layer_id} received a value embedding but has no value-embedding parameters."
                )
            if self.num_v_heads != self.config.num_key_value_heads:
                raise ValueError(
                    "KDA value embeddings require kda_num_v_heads == num_key_value_heads."
                )
            ve = ve.view(B, T, self.num_v_heads, self.ve_head_dim)
            ve = self.ve_proj(ve)
            ve_gate = 2 * torch.sigmoid(
                self.ve_gate(x[..., :self.ve_gate_channels]),
            )
            v = v + (ve_gate.unsqueeze(-1) * ve).flatten(2)

        conv_state = (None, None, None) if last_state is None else last_state["conv_state"]
        recurrent_state = None if last_state is None else last_state["recurrent_state"]
        if self.use_short_conv:
            q, q_conv_state = self.q_conv1d(
                q,
                mask=attention_mask,
                cache=conv_state[0],
                output_final_state=past_key_values is not None,
            )
            k, k_conv_state = self.k_conv1d(
                k,
                mask=attention_mask,
                cache=conv_state[1],
                output_final_state=past_key_values is not None,
            )
            v, v_conv_state = self.v_conv1d(
                v,
                mask=attention_mask,
                cache=conv_state[2],
                output_final_state=past_key_values is not None,
            )
        else:
            q, k, v = F.silu(q), F.silu(k), F.silu(v)
            q_conv_state = k_conv_state = v_conv_state = None

        q = q.view(B, T, self.num_heads, self.head_k_dim)
        k = k.view(B, T, self.num_heads, self.head_k_dim)
        v = v.view(B, T, self.num_v_heads, self.head_v_dim)
        g = self.f_proj(state_source).view(B, T, self.num_v_heads, self.head_k_dim)
        beta = self.b_proj(state_source)

        o, recurrent_state = self._chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            initial_state=recurrent_state,
            output_final_state=past_key_values is not None,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            allow_neg_eigval=self.allow_neg_eigval,
            safe_gate=self.safe_gate,
            lower_bound=self.lower_bound,
            disable_recompute=self.disable_recompute,
            state_v_first=True,
        )
        if past_key_values is not None:
            past_key_values.update_kda_state(
                {
                    "conv_state": (q_conv_state, k_conv_state, v_conv_state),
                    "recurrent_state": recurrent_state,
                },
                step_idx,
                T,
            )
        gate = self.g_proj(x).view(B, T, self.num_v_heads, self.head_v_dim)
        o = self.o_norm(o, gate)
        return self.o_proj(o.reshape(B, T, self.value_dim))

    def init_state(self, batch_size: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        param = next(self.parameters())
        return (
            param.new_zeros(batch_size, self.key_dim, self.conv_size),
            param.new_zeros(batch_size, self.key_dim, self.conv_size),
            param.new_zeros(batch_size, self.value_dim, self.conv_size),
            param.new_zeros(
                batch_size, self.num_v_heads, self.head_v_dim, self.head_k_dim,
            ),
        )

    def state_size(self, **kwargs) -> int:
        del kwargs
        conv_state_size = 0
        if self.use_short_conv:
            conv_state_size = (
                2 * self.key_dim * self.conv_size
                + self.value_dim * self.conv_size
            )
        return (
            conv_state_size
            + self.num_v_heads * self.head_v_dim * self.head_k_dim
        )
