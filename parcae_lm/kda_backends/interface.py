"""Select a KDA implementation without importing FLA for non-KDA models."""

from collections.abc import Callable


def select_kda_implementation(provider: str = "fla-chunk") -> Callable:
    """Return the requested KDA implementation."""
    if provider in ("fla-chunk", "fla", "chunk"):
        from .fla_kda import chunk_kda

        return chunk_kda
    if provider in ("fla-custom-op", "custom-op"):
        from .fla_custom_op import chunk_kda

        return chunk_kda
    raise ValueError(
        f"KDA provider {provider!r} not supported. "
        "Use 'fla-chunk' or 'fla-custom-op'."
    )
