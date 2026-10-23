"""Unmodified FLA KDA backend used as the correctness and performance control."""

from typing import Any

import torch


def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    **kwargs: Any,
):
    from fla.ops.kda import chunk_kda as fla_chunk_kda  # type: ignore[import-not-found]

    return fla_chunk_kda(q=q, k=k, v=v, g=g, beta=beta, **kwargs)
