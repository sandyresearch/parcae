from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch


def configure_compile(
    local_rank: int,
    *,
    optimize_ddp: str,
    compiled_autograd: bool,
    shape_padding: bool = True,
) -> None:
    cache_dir = (
        Path(tempfile.gettempdir())
        / f"recft-{os.getuid()}"
        / f"rank-{local_rank}"
    )
    inductor_dir = cache_dir / "inductor"
    triton_dir = cache_dir / "triton"
    inductor_dir.mkdir(parents=True, exist_ok=True)
    triton_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_dir)
    os.environ["TRITON_CACHE_DIR"] = str(triton_dir)

    import torch._dynamo.config as dynamo_config
    import torch._functorch.config as functorch_config
    import torch._inductor.config as inductor_config

    dynamo_config.cache_size_limit = 256
    if hasattr(dynamo_config, "recompile_limit"):
        dynamo_config.recompile_limit = 256
    if hasattr(dynamo_config, "accumulated_recompile_limit"):
        dynamo_config.accumulated_recompile_limit = 256
    dynamo_config.capture_scalar_outputs = True
    dynamo_config.optimize_ddp = optimize_ddp
    dynamo_config.compiled_autograd = compiled_autograd
    dynamo_config.suppress_errors = False
    inductor_config.emulate_precision_casts = True
    inductor_config.triton.cudagraphs = False
    inductor_config.coordinate_descent_tuning = True
    inductor_config.shape_padding = shape_padding
    inductor_config.fx_graph_cache = True
    functorch_config.donated_buffer = False


def compile_transformer_blocks(model, mode: str) -> None:
    def compile_fn(module):
        return torch.compile(
            module,
            backend="inductor",
            mode=mode,
            dynamic=False,
            fullgraph=False,
        )

    for name in ("prelude", "core_block", "coda"):
        blocks = getattr(model.transformer, name)
        for index, block in enumerate(blocks):
            blocks[index] = compile_fn(block)


def unwrap_transformer_blocks(model) -> None:
    for name in ("prelude", "core_block", "coda"):
        blocks = getattr(model.transformer, name)
        for index, block in enumerate(blocks):
            if hasattr(block, "_orig_mod"):
                blocks[index] = block._orig_mod
