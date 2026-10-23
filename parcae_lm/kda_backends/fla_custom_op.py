"""Compile-friendly training wrapper for FLA's KDA Triton kernels.

The public FLA ``chunk_kda`` function is decorated with
``torch.compiler.disable``.  This backend bypasses that Python entry point and
registers the low-level forward and backward kernel pipelines as opaque
``torch.library`` operators.  The default FLA backend remains available as a
control and as the fallback for stateful inference and unsupported options.
"""

from typing import Any, Optional

import torch

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd  # type: ignore[import-not-found]
from fla.ops.common.gate import (  # type: ignore[import-not-found]
    fused_beta_sigmoid,
    fused_beta_sigmoid_bwd,
)
from fla.ops.kda.chunk_bwd import chunk_kda_bwd  # type: ignore[import-not-found]
from fla.ops.kda.chunk_fwd import chunk_kda_fwd  # type: ignore[import-not-found]


_ForwardOutputs = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]
_BackwardOutputs = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


@torch.library.custom_op(
    "parcae::fla_kda_forward",
    mutates_args=(),
    device_types="cuda",
)
def _fla_kda_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    allow_neg_eigval: bool,
    safe_gate: bool,
    lower_bound: Optional[float],
    state_v_first: bool,
    chunk_size: int,
) -> _ForwardOutputs:
    q_norm, q_rstd = l2norm_fwd(q)
    k_norm, k_rstd = l2norm_fwd(k)
    beta_activated = fused_beta_sigmoid(
        beta,
        scale=2.0 if allow_neg_eigval else 1.0,
    )
    (
        output,
        _final_state,
        _g_cumsum,
        Aqk,
        Akk,
        _w,
        _u,
        _qg,
        _kg,
        _v_new,
        _h,
        _initial_state,
    ) = chunk_kda_fwd(
        q=q_norm,
        k=k_norm,
        v=v,
        g=g,
        beta=beta_activated,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        state_v_first=state_v_first,
        chunk_size=chunk_size,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        use_gate_in_kernel=True,
        A_log=A_log,
        dt_bias=dt_bias,
        disable_recompute=False,
    )
    return (
        output.to(q.dtype),
        q_norm,
        q_rstd,
        k_norm,
        k_rstd,
        beta_activated,
        Aqk,
        Akk,
    )


@_fla_kda_forward.register_fake
def _fla_kda_forward_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    allow_neg_eigval: bool,
    safe_gate: bool,
    lower_bound: Optional[float],
    state_v_first: bool,
    chunk_size: int,
) -> _ForwardOutputs:
    del g, A_log, dt_bias, scale, allow_neg_eigval, safe_gate, lower_bound, state_v_first
    batch, sequence, value_heads, value_dim = v.shape
    output = q.new_empty((batch, sequence, value_heads, value_dim))
    q_rstd = q.new_empty(q.shape[:-1], dtype=torch.float32)
    k_rstd = k.new_empty(k.shape[:-1], dtype=torch.float32)
    Aqk = k.new_empty((batch, sequence, value_heads, chunk_size))
    Akk = k.new_empty((batch, sequence, value_heads, chunk_size))
    return (
        output,
        torch.empty_like(q),
        q_rstd,
        torch.empty_like(k),
        k_rstd,
        torch.empty_like(beta, dtype=torch.float32),
        Aqk,
        Akk,
    )


@torch.library.custom_op(
    "parcae::fla_kda_backward",
    mutates_args=(),
    device_types="cuda",
)
def _fla_kda_backward(
    q_norm: torch.Tensor,
    q_rstd: torch.Tensor,
    k_norm: torch.Tensor,
    k_rstd: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    beta_activated: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    grad_output: torch.Tensor,
    scale: float,
    allow_neg_eigval: bool,
    safe_gate: bool,
    lower_bound: Optional[float],
    state_v_first: bool,
    chunk_size: int,
) -> _BackwardOutputs:
    dq, dk, dv, dbeta, dg, _dh0, dA, dbias = chunk_kda_bwd(
        q=q_norm,
        k=k_norm,
        v=v,
        beta=beta_activated,
        Aqk=Aqk,
        Akk=Akk,
        scale=scale,
        initial_state=None,
        do=grad_output.contiguous(),
        dht=None,
        g=None,
        g_org=g,
        state_v_first=state_v_first,
        chunk_size=chunk_size,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        use_gate_in_kernel=True,
        A_log=A_log,
        dt_bias=dt_bias,
        disable_recompute=False,
    )
    dq = l2norm_bwd(q_norm, q_rstd, dq)
    dk = l2norm_bwd(k_norm, k_rstd, dk)
    dbeta = fused_beta_sigmoid_bwd(
        beta,
        dbeta,
        scale=2.0 if allow_neg_eigval else 1.0,
    )
    return (
        dq.to(q_norm.dtype),
        dk.to(k_norm.dtype),
        dv.to(v.dtype),
        dg.to(g.dtype),
        dbeta.to(beta.dtype),
        dA,
        dbias,
    )


@_fla_kda_backward.register_fake
def _fla_kda_backward_fake(
    q_norm: torch.Tensor,
    q_rstd: torch.Tensor,
    k_norm: torch.Tensor,
    k_rstd: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    beta_activated: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    grad_output: torch.Tensor,
    scale: float,
    allow_neg_eigval: bool,
    safe_gate: bool,
    lower_bound: Optional[float],
    state_v_first: bool,
    chunk_size: int,
) -> _BackwardOutputs:
    del (
        q_rstd,
        k_rstd,
        beta_activated,
        Aqk,
        Akk,
        grad_output,
        scale,
        allow_neg_eigval,
        safe_gate,
        lower_bound,
        state_v_first,
        chunk_size,
    )
    return (
        torch.empty_like(q_norm),
        torch.empty_like(k_norm),
        torch.empty_like(v),
        torch.empty_like(g),
        torch.empty_like(beta),
        torch.empty_like(A_log),
        torch.empty_like(dt_bias),
    )


def _setup_context(ctx: Any, inputs: tuple[Any, ...], output: _ForwardOutputs) -> None:
    (
        _q,
        _k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        scale,
        allow_neg_eigval,
        safe_gate,
        lower_bound,
        state_v_first,
        chunk_size,
    ) = inputs
    output_tensor, q_norm, q_rstd, k_norm, k_rstd, beta_activated, Aqk, Akk = output
    ctx.mark_non_differentiable(q_norm, q_rstd, k_norm, k_rstd, beta_activated, Aqk, Akk)
    ctx.save_for_backward(
        q_norm,
        q_rstd,
        k_norm,
        k_rstd,
        v,
        g,
        beta,
        beta_activated,
        A_log,
        dt_bias,
        Aqk,
        Akk,
    )
    ctx.scale = scale
    ctx.allow_neg_eigval = allow_neg_eigval
    ctx.safe_gate = safe_gate
    ctx.lower_bound = lower_bound
    ctx.state_v_first = state_v_first
    ctx.chunk_size = chunk_size
    del output_tensor


def _backward(
    ctx: Any,
    grad_output: torch.Tensor,
    _grad_q_norm: Optional[torch.Tensor],
    _grad_q_rstd: Optional[torch.Tensor],
    _grad_k_norm: Optional[torch.Tensor],
    _grad_k_rstd: Optional[torch.Tensor],
    _grad_beta_activated: Optional[torch.Tensor],
    _grad_Aqk: Optional[torch.Tensor],
    _grad_Akk: Optional[torch.Tensor],
) -> tuple[Optional[torch.Tensor], ...]:
    (
        q_norm,
        q_rstd,
        k_norm,
        k_rstd,
        v,
        g,
        beta,
        beta_activated,
        A_log,
        dt_bias,
        Aqk,
        Akk,
    ) = ctx.saved_tensors
    dq, dk, dv, dg, dbeta, dA, dbias = _fla_kda_backward(
        q_norm,
        q_rstd,
        k_norm,
        k_rstd,
        v,
        g,
        beta,
        beta_activated,
        A_log,
        dt_bias,
        Aqk,
        Akk,
        grad_output,
        ctx.scale,
        ctx.allow_neg_eigval,
        ctx.safe_gate,
        ctx.lower_bound,
        ctx.state_v_first,
        ctx.chunk_size,
    )
    return dq, dk, dv, dg, dbeta, dA, dbias, None, None, None, None, None, None


_fla_kda_forward.register_autograd(_backward, setup_context=_setup_context)


def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    lower_bound: Optional[float] = None,
    disable_recompute: bool = False,
    return_intermediate_states: bool = False,
    state_v_first: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_cpu: Optional[torch.Tensor] = None,
    cp_context: Any = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    **kwargs: Any,
):
    """Run graph-preserving KDA for Parcae training, falling back when needed."""
    supported_training_call = (
        initial_state is None
        and not output_final_state
        and use_qk_l2norm_in_kernel
        and use_gate_in_kernel
        and use_beta_sigmoid_in_kernel
        and not disable_recompute
        and not return_intermediate_states
        and cu_seqlens is None
        and cu_seqlens_cpu is None
        and cp_context is None
        and A_log is not None
        and dt_bias is not None
        and not kwargs
    )
    if not supported_training_call:
        from .fla_kda import chunk_kda as fla_chunk_kda

        return fla_chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            allow_neg_eigval=allow_neg_eigval,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            disable_recompute=disable_recompute,
            return_intermediate_states=return_intermediate_states,
            state_v_first=state_v_first,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            cp_context=cp_context,
            A_log=A_log,
            dt_bias=dt_bias,
            chunk_size=chunk_size,
            **kwargs,
        )

    if scale is None:
        scale = q.shape[-1] ** -0.5
    if safe_gate and (lower_bound is None or not -5 <= lower_bound < 0):
        raise ValueError("safe_gate requires lower_bound in [-5, 0).")
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError("allow_neg_eigval requires beta sigmoid in the kernel.")
    assert A_log is not None and dt_bias is not None

    q, k, v, g, beta, A_log, dt_bias = (
        tensor.contiguous() for tensor in (q, k, v, g, beta, A_log, dt_bias)
    )
    output, *_saved = _fla_kda_forward(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        scale,
        allow_neg_eigval,
        safe_gate,
        lower_bound,
        state_v_first,
        chunk_size,
    )
    return output, None
