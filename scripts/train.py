import time
import sys
import signal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from parcae_lm.models.parcae import ParcaeConfig
from parcae_lm.tokenizer import Tokenizer

global_start_time = time.time()
import math
import os
import socket

from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Tuple, Optional
import json

import torch
import torch.nn as nn

nvml_count = torch.cuda._device_count_amdsmi() if torch.version.hip else torch.cuda._device_count_nvml()
if nvml_count < 1:
    raise ValueError(f"Node failure! Device manager init failed on {socket.gethostname()}")


if TYPE_CHECKING:
    import torch.distributed
    import torch.version
    import torch._dynamo.config
from lightning.fabric.strategies import FSDPStrategy, DDPStrategy, SingleDeviceStrategy
from lightning.pytorch.loggers import WandbLogger
from torchdata.stateful_dataloader import StatefulDataLoader
from torch.utils.data import DataLoader
from torchmetrics.aggregation import RunningMean
from torch.distributed.checkpoint import state_dict as state_dict_helpers

import warnings

warnings.filterwarnings("ignore", message="The config.capture_autograd_function flag is deprecated")  # pytorch nightly
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`.*")  # our weights

from recpre.settings import CLISettings
from recpre.schedulers import get_lr_scheduler

from recpre.huggingface_dataset import HuggingfaceDataset, ParquetStream, ParquetStreamPure, RandomTokensDataset
from recpre.data_loading_utils import (
    generic_collate_fn,
    BestFitPackingCollator,
    PackedIterableDataset,
    unwrap_singleton_collate,
)
import recpre.utils
from recpre.data_scheduler_utils import DataSchedulerTracker, DataScheduler
from recpre.monitor import (
    enable_monitoring_on_step,
    disable_monitoring_and_retrieve_metrics,
    track_gradient_metrics,
    get_MFU_metrics,
)
from cost import estimate_flops_recurrent, estimate_flops_gpt, FLOPBreakdown


def compute_flops_per_token_at_recurrence(flop_breakdown: FLOPBreakdown, forward_only: int, backprop_depth: int) -> int:
    if not flop_breakdown.is_recurrent:
        return 6 * flop_breakdown.total_params + flop_breakdown.total_attn_flops
    
    total_recurrence = forward_only + backprop_depth
    
    non_core_flops = 6 * flop_breakdown.non_core_params + flop_breakdown.non_core_attn_flops
    core_fwd = 2 * flop_breakdown.core_block_params * total_recurrence
    core_bwd = 4 * flop_breakdown.core_block_params * backprop_depth
    core_attn_fwd = flop_breakdown.core_attn_fwd_per_step * total_recurrence
    core_attn_bwd = 2 * flop_breakdown.core_attn_fwd_per_step * backprop_depth
    
    return int(non_core_flops + core_fwd + core_bwd + core_attn_fwd + core_attn_bwd)

from dataclasses import asdict, is_dataclass
from jsonargparse import CLI
import re

RETRY_CACHE_INDUCTOR = False

if RETRY_CACHE_INDUCTOR:
    import torch._inductor.codecache

    torch._inductor.codecache.PyCodeCache.load_by_key_path = classmethod(recpre.utils.load_by_key_path_with_retry)

end_time = time.time()
if int(os.getenv("SLURM_PROCID", "0")) == 0:
    print(f"{time.ctime()[:-5]}: Time to load libraries: {end_time - global_start_time:.02f} seconds.")

Fabric = recpre.utils.LightningFabric | recpre.utils.SimpleFabric


def set_torch_flags(cfg):
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    try:
        torch.backends.cuda.enable_cudnn_sdp(False)
    except AttributeError:
        pass

    torch._dynamo.config.optimize_ddp = cfg.dynamo_ddp_config
    torch._dynamo.config.compiled_autograd = cfg.compiled_autograd
    
    if cfg.compile_model:
        import torch._inductor.config as inductor_config
        import torch._functorch.config as functorch_config
        inductor_config.emulate_precision_casts = True
        inductor_config.triton.cudagraphs = False
        inductor_config.coordinate_descent_tuning = False
        inductor_config.fx_graph_cache = False
        functorch_config.donated_buffer = False
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.suppress_errors = False

    if cfg.fail_instead_of_recompile:
        torch._dynamo.config.error_on_recompile = True


def setup_fabric(cfg: CLISettings, wandb_run_id: Optional[str] = None) -> Fabric:
    if cfg.logger_name == "wandb":
        logger_kwargs = {
            "project": cfg.logger_project,
            "name": cfg.run_name,
            "save_dir": cfg.out_dir
        }
        if cfg.logger_entity:
            logger_kwargs["entity"] = cfg.logger_entity
        if cfg.logger_group is not None:
            logger_kwargs["group"] = cfg.logger_group
        if wandb_run_id is not None:
            logger_kwargs["id"] = wandb_run_id
            logger_kwargs["resume"] = "allow"
        logger = WandbLogger(**logger_kwargs)
    else:
        raise ValueError(f"`logger={cfg.logger_name}` is not a valid option.")

    if cfg.fabric_strategy == "simple-ddp":
        fabric = recpre.utils.SimpleFabric(precision=cfg.fabric_precision, loggers=[logger])
        fabric.print("Using simple fabric.")
    else:
        if "fsdp" in cfg.fabric_strategy:
            if "grad" in cfg.fabric_strategy:
                sharding_strategy = "SHARD_GRAD_OP"
            elif "full" in cfg.fabric_strategy:
                sharding_strategy = "FULL_SHARD"
            elif "hybrid2" in cfg.fabric_strategy:
                sharding_strategy = "_HYBRID_SHARD_ZERO2"
            else:
                sharding_strategy = "HYBRID_SHARD"
            precision_strategy = derive_precision(cfg.fabric_precision, cfg.fabric)
            strategy = FSDPStrategy(
                auto_wrap_policy={cfg.model_config.Block},
                mixed_precision=precision_strategy,
                activation_checkpointing_policy={cfg.model_config.Block} if cfg.gradient_checkpointing else None,
                state_dict_type="full",
                sharding_strategy=sharding_strategy,  # type: ignore
                param_init_fn=((lambda x: x.to_empty(recurse=False)) if cfg.model_impl == "huggingface" else None),
            )
        elif cfg.fabric_strategy == "ddp":
            strategy = DDPStrategy(find_unused_parameters=True)
        elif cfg.fabric_strategy == "single":
            strategy = SingleDeviceStrategy(device=torch.device("cuda:0") if torch.cuda.is_available() else "cpu")
        elif cfg.fabric_strategy == "axonn_tp":
            from axonn.lightning import AxonnStrategy
            strategy = AxonnStrategy(
                G_intra_r=cfg.fabric.row_tensor_parallel_size,
                G_intra_c=cfg.fabric.col_tensor_parallel_size,  # this needs more integration!
                G_intra_d=cfg.fabric.depth_tensor_parallel_size,
                overlap_communication=cfg.fabric.optimize_communication,
            )
        else:
            raise ValueError(f"`fabric_strategy={cfg.fabric_strategy}` is not a valid option.")

        fabric = recpre.utils.LightningFabric(
            devices=cfg.devices,
            strategy=strategy,
            precision=cfg.fabric_precision,
            loggers=[logger],
            num_nodes=cfg.num_nodes,
        )
        fabric.print(f"Using Lightning Fabric with strategy {cfg.fabric_strategy} ")
        fabric.launch()

    fabric.print(f"> gradient_accumulation_steps = {cfg.gradient_accumulation_steps}")
    fabric.print(f"> micro_batch_size = {cfg.micro_batch_size}")
    fabric.print(f"> global_batch_size = {cfg.world_batch_size}")

    return fabric


def startup(fabric: Fabric, cfg: CLISettings):
    start_time = time.time()

    if cfg.save_n_min_before_job_done is not None:
        if fabric.global_rank == 0:
            global_total_time = _get_time_from_slurm()
            fabric.print(f"Total job time: {global_total_time:.02f} seconds.")
        else:
            global_total_time = 0

        global_total_time = fabric.broadcast(global_total_time, 0)
        cfg.global_total_time = global_total_time

    if fabric.global_rank == 0:
        if cfg.resume_checkpoint_path is not None:
            checkpoint_path = Path(cfg.resume_checkpoint_path)
            if "checkpoints-" in str(checkpoint_path):
                cfg.out_dir = str(checkpoint_path.parent.parent)
                fabric.print(f"Using output directory from checkpoint path: {cfg.out_dir}")
    if torch.distributed.is_initialized():
        out_dir_list = [cfg.out_dir]
        torch.distributed.broadcast_object_list(out_dir_list, src=0)
        cfg.out_dir = out_dir_list[0]

    checkpoint_error = [None]
    if fabric.global_rank == 0 and not cfg.resume and Path(cfg.out_dir).exists():
        for subdir in ["checkpoints-DDPStrategy", "checkpoints-FSDPStrategy", "checkpoints-SingleDeviceStrategy"]:
            checkpoint_dir = Path(cfg.out_dir) / subdir
            if checkpoint_dir.exists():
                # Check for checkpoint files/directories (with or without .pth extension)
                checkpoints = list(checkpoint_dir.glob("step-*"))
                if checkpoints:
                    checkpoint_error[0] = f"Output directory {cfg.out_dir} already contains checkpoints. Use --resume=True to resume."
                    break
    if torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(checkpoint_error, src=0)
    if checkpoint_error[0] is not None:
        raise ValueError(checkpoint_error[0])

    if fabric.global_rank == 0:
        Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
        (Path(cfg.out_dir) / fabric.get_prefix_for_checkpoint()).mkdir(parents=True, exist_ok=True)
        with open(f"{cfg.out_dir}/run_config.json", "w") as f:
            json.dump(asdict(cfg), f, indent=4)
        with open(f"{cfg.out_dir}/model_config.json", "w") as f:
            json.dump(asdict(cfg.model_config) if is_dataclass(cfg.model_config) else cfg.model_config, f, indent=4)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    tokenizer = Tokenizer(cfg.tokenizer_path)
    if tokenizer.pad_id is None:
        # Use 0 as pad token (common convention) - -1 is invalid for embedding lookup
        tokenizer.pad_id = 0
    if cfg.cache_attn:
        assert tokenizer.cache_token_id is not None
    if cfg.doc_block_attn:
        assert tokenizer.eod_token_id is not None
    tokenizer_vocab_size = len(tokenizer)
    if tokenizer_vocab_size > cfg.model_config.vocab_size:
        cfg.model_config.vocab_size = tokenizer_vocab_size

    t0 = time.time()
    if not cfg.ignore_block_size_mismatch:
        assert cfg.block_size == cfg.model_config.block_size, "cfg.block_size must match config.block_size"
    train_dataloader, val_dataloader, data_scheduler_tracker, packing_collator = create_dataloaders(
        batch_size=cfg.micro_batch_size,
        block_size=cfg.loader_block_size,
        fabric=fabric,
        seed=(cfg.seed + fabric.global_rank),
        cfg=cfg,
        tokenizer=tokenizer,
    )
    if data_scheduler_tracker is not None:
        data_scheduler = DataScheduler(data_scheduler_tracker, cfg.data_config["train_data"], cfg)
        data_scheduler.step(0)
    else:
        data_scheduler = None
    fabric.print(f"{time.ctime()[:-5]}: Time to instantiate and setup dataloaders: {time.time() - t0:.02f} seconds.")

    fabric.seed_everything(cfg.seed)  # same seed for every process to init model (FSDP)
    fabric.print(f"Loading model with {cfg.model_config.__dict__}")

    objective = {
        "op": recpre.utils.chunked_cross_entropy,
        "label_smoothing": cfg.label_smoothing,
        "ignore_index": tokenizer.pad_id,
        "z_regularization": cfg.z_regularization,
    }

    t0 = time.time()
    with fabric.init_module(empty_init="fsdp" in cfg.fabric_strategy):
        model = cfg.model_config.construct_model(
            objective=objective,
            gradient_checkpointing=cfg.gradient_checkpointing and "fsdp" not in cfg.fabric_strategy,
        )
    fabric.print(f"{time.ctime()[:-5]}: Time to instantiate model: {time.time() - t0:.02f} seconds.")
    num_params = recpre.utils.num_parameters(model)
    rec_params = recpre.utils.num_recurrent_parameters(model)
    fabric.log_to_summary({"num_parameters": num_params, "device": torch.cuda.get_device_name()})
    fabric.log_to_summary({"num_recurrent_parameters": rec_params})

    is_recurrent_model = hasattr(model, 'transformer') and hasattr(model.transformer, 'core_block') and model.transformer.core_block is not None
    if is_recurrent_model:
        flop_breakdown = estimate_flops_recurrent(model, cfg.model_config)
    else:
        flop_breakdown = estimate_flops_gpt(model, cfg.model_config)
    flops_per_token = flop_breakdown.flops_per_token(use_curriculum_adjusted=False)

    if cfg.max_flops is not None and cfg.max_tokens == 0:
        cfg.max_tokens = int(cfg.max_flops / flops_per_token)
        fabric.print(f"Derived max_tokens={cfg.max_tokens:,} from max_flops={cfg.max_flops:.2e}")
    if cfg.token_to_param_ratio is not None and cfg.max_tokens == 0:
        cfg.max_tokens = int(cfg.token_to_param_ratio * num_params)
        fabric.print(f"Derived max_tokens={cfg.max_tokens:,} from {cfg.token_to_param_ratio}:1 token-to-param ratio ({num_params:,} params)")

    if cfg.max_steps is None:
        cfg.max_tokens_per_device = cfg.max_tokens // fabric.world_size
        cfg.tokens_per_step = cfg.micro_batch_size * cfg.block_size * cfg.gradient_accumulation_steps
        cfg.max_steps = cfg.max_tokens_per_device // cfg.tokens_per_step
        fabric.print(
            f"Based on block size {cfg.block_size}, expecting to take {cfg.max_steps} optimizer steps to reach "
            f"{cfg.max_tokens / 1e12:4.2f}T tokens.\nRunning {cfg.tokens_per_step * fabric.world_size} tok/step "
            f"for {cfg.max_tokens_per_device / 1e9:4.2f}B in total per device."
        )

    t0 = time.time()
    if cfg.compile_model and cfg.compile_backend == "selective_pre_ddp":
        fabric.print(f"{time.ctime()[:-5]}: Selectively compiling transformer blocks BEFORE DDP wrapping...")
        compile_fn = lambda m: torch.compile(m, backend="inductor", mode=cfg.compile_mode, dynamic=False)
        
        if hasattr(model, "transformer"):
            if hasattr(model.transformer, "prelude"):
                model.transformer.prelude = torch.nn.ModuleList([compile_fn(block) for block in model.transformer.prelude])
            if hasattr(model.transformer, "core_block"):
                model.transformer.core_block = torch.nn.ModuleList([compile_fn(block) for block in model.transformer.core_block])
            if hasattr(model.transformer, "coda"):
                model.transformer.coda = torch.nn.ModuleList([compile_fn(block) for block in model.transformer.coda])
        fabric.print(f"{time.ctime()[:-5]}: Pre-DDP selective compilation complete!")
    model = fabric.setup(model, compile=False)
    if cfg.compile_model and cfg.compile_backend != "selective_pre_ddp":
        if cfg.compile_backend == "selective":
            fabric.print(f"{time.ctime()[:-5]}: Selectively compiling transformer blocks with inductor...")
            compile_fn = lambda m: torch.compile(m, backend="inductor", mode=cfg.compile_mode, dynamic=False)
            inner_model = model.module if hasattr(model, "module") else model
            if hasattr(inner_model, "transformer"):
                if hasattr(inner_model.transformer, "prelude"):
                    inner_model.transformer.prelude = torch.nn.ModuleList([compile_fn(block) for block in inner_model.transformer.prelude])
                if hasattr(inner_model.transformer, "core_block"):
                    inner_model.transformer.core_block = torch.nn.ModuleList([compile_fn(block) for block in inner_model.transformer.core_block])
                if hasattr(inner_model.transformer, "coda"):
                    inner_model.transformer.coda = torch.nn.ModuleList([compile_fn(block) for block in inner_model.transformer.coda])
            fabric.print(f"{time.ctime()[:-5]}: Selective compilation complete!")
        elif cfg.compile_backend == "selective_eager":
            fabric.print(f"{time.ctime()[:-5]}: Selectively compiling transformer blocks with EAGER (no AOT autograd)...")
            compile_fn = lambda m: torch.compile(m, backend="eager", dynamic=False)
            
            inner_model = model.module if hasattr(model, "module") else model
            if hasattr(inner_model, "transformer"):
                if hasattr(inner_model.transformer, "prelude"):
                    inner_model.transformer.prelude = torch.nn.ModuleList([compile_fn(block) for block in inner_model.transformer.prelude])
                if hasattr(inner_model.transformer, "core_block"):
                    inner_model.transformer.core_block = torch.nn.ModuleList([compile_fn(block) for block in inner_model.transformer.core_block])
                if hasattr(inner_model.transformer, "coda"):
                    inner_model.transformer.coda = torch.nn.ModuleList([compile_fn(block) for block in inner_model.transformer.coda])
            fabric.print(f"{time.ctime()[:-5]}: Selective eager compilation complete!")
        elif cfg.compile_backend == "torchscript":
            fabric.print(f"{time.ctime()[:-5]}: Compiling transformer blocks with TorchScript...")
            
            inner_model = model.module if hasattr(model, "module") else model
            if hasattr(inner_model, "transformer"):
                if hasattr(inner_model.transformer, "prelude"):
                    for i, block in enumerate(inner_model.transformer.prelude):
                        try:
                            inner_model.transformer.prelude[i] = torch.jit.script(block)
                        except Exception as e:
                            fabric.print(f"Warning: Could not script prelude block {i}: {e}")
                if hasattr(inner_model.transformer, "core_block"):
                    for i, block in enumerate(inner_model.transformer.core_block):
                        try:
                            inner_model.transformer.core_block[i] = torch.jit.script(block)
                        except Exception as e:
                            fabric.print(f"Warning: Could not script core_block {i}: {e}")
                if hasattr(inner_model.transformer, "coda"):
                    for i, block in enumerate(inner_model.transformer.coda):
                        try:
                            inner_model.transformer.coda[i] = torch.jit.script(block)
                        except Exception as e:
                            fabric.print(f"Warning: Could not script coda block {i}: {e}")
            fabric.print(f"{time.ctime()[:-5]}: TorchScript compilation complete!")
        elif cfg.compile_backend == "post_ddp":
            fabric.print(f"{time.ctime()[:-5]}: Compiling model AFTER DDP wrapping...")
            model = torch.compile(
                model,
                backend="inductor",
                mode=cfg.compile_mode,
                fullgraph=cfg.compile_fullgraph,
                dynamic=cfg.compile_dynamic,
            )
            fabric.print(f"{time.ctime()[:-5]}: Post-DDP compilation complete!")
        else:
            fabric.print(f"{time.ctime()[:-5]}: Compiling with backend={cfg.compile_backend}...")
            model = torch.compile(
                model,
                backend=cfg.compile_backend,
                mode=cfg.compile_mode,
                fullgraph=cfg.compile_fullgraph,
                dynamic=cfg.compile_dynamic,
            )
            fabric.print(f"{time.ctime()[:-5]}: Model compiled!")
    if cfg.compile_model:
        fabric.print(f"{time.ctime()[:-5]}: Model compilation complete!")
    fabric.print(f"Model with full setup is {model}")
    fabric.print(f"Total parameters: {num_params:,}")
    if hasattr(model, "transformer") and hasattr(model.transformer, "core_block") and model.transformer.core_block is not None:
        rec_params = recpre.utils.num_recurrent_parameters(model)
        static_params = num_params - rec_params
        if "fsdp" in cfg.fabric_strategy:
            rec_params, static_params = rec_params * cfg.devices, static_params * cfg.devices
        r = model.config.mean_recurrence
        unfolded_params_mean = static_params + rec_params * r
        unfolded_params_max = static_params + rec_params * 2 * r
        fabric.print(f"Model initialized with {int(rec_params / 1e6):,}m parameters in recurrent block.")
        fabric.print(f"Will unfold to {int(unfolded_params_mean // 1e6):,}m mean parameters at test time ({r} rec).")
        fabric.print(f"Could unfold to {int(unfolded_params_max // 1e6):,}m parameters at test time ({2 * r} rec).")
    fabric.print(f"{time.ctime()[:-5]}: Time to setup model: {time.time() - t0:.02f} seconds.")
    t0 = time.time()
    fabric.print(f"{time.ctime()[:-5]}: Setting up optimizer...")
    verbose_param_groups = fabric.global_rank == 0
    if cfg.optimizer == "MuonAdamW":
        param_groups = recpre.optim.get_muon_param_groups_from_config(
            model.named_parameters(),
            cfg.optim_config,
            cfg.no_weight_decay_for_bias_and_norm_params,
            verbose=verbose_param_groups,
        )
    else:
        param_groups = recpre.optim.get_param_groups(
            model.named_parameters(),
            cfg.no_weight_decay_for_bias_and_norm_params,
            weight_lr_scale=1 / getattr(model.config, "mup_model_scaling_factor", 1.0),
            verbose=verbose_param_groups,
        )
    if fabric.global_rank == 0:
        if cfg.optimizer == "MuonAdamW":
            adamw_params = sum(p.numel() for g in param_groups if g.get('kind') == 'adamw' for p in g['params'])
            muon_params = sum(p.numel() for g in param_groups if g.get('kind') == 'muon' for p in g['params'])
            total = adamw_params + muon_params
            fabric.print(f"Optimizer param groups:")
            fabric.print(f"  AdamW: {adamw_params:,} params ({adamw_params/total*100:.1f}%)")
            fabric.print(f"  Muon: {muon_params:,} params ({muon_params/total*100:.1f}%)")
        elif len(param_groups) >= 3:
            fabric.print(f"Param group sizes: weights={len(param_groups[0]['params'])}, "
                         f"embeddings={len(param_groups[1]['params'])}, "
                         f"scalers={len(param_groups[2]['params'])}")
    optimizer = recpre.optim.get_optimizer(
        cfg.optimizer,
        model,
        cfg.fabric.optim_sharding,
        allow_fusion=not torch.version.hip and "bf16" in cfg.fabric_precision,
        use_apex_adamw=cfg.fabric.use_apex_adamw,
    )(param_groups, **cfg.optim_config)
    optimizer = fabric.setup_optimizers(optimizer)
    fabric.print(f"{time.ctime()[:-5]}: Time to instantiate and setup optimizers: {time.time() - t0:.02f} seconds.")

    assert cfg.max_steps is not None, "max_steps must be set before creating LR scheduler"
    warmup_steps = int(cfg.warmup_steps * cfg.max_steps) if cfg.warmup_steps < 1 else cfg.warmup_steps
    cooldown_steps = int(cfg.cooldown_steps * cfg.max_steps) if cfg.cooldown_steps < 1 else cfg.cooldown_steps
    lr_scheduler = get_lr_scheduler(
        schedule_type=cfg.lr_schedule,
        base_lr=cfg.optim_config["lr"],
        min_lr=cfg.min_lr,
        warmup_steps=warmup_steps,
        cooldown_steps=cooldown_steps,
        max_steps=cfg.max_steps,
    )

    max_flops = cfg.max_tokens * flops_per_token
    fabric.log_to_summary({"max_flops": max_flops, "flops_per_token": flops_per_token})
    fabric.print(f"Estimated max FLOPs: {max_flops/1e18:.5f} exaFLOPs ({flops_per_token:,} FLOPs/token)")

    base_muon_wd = cfg.optim_config.get('muon_wd', 0.0)
    muon_momentum = cfg.optim_config.get('muon_momentum', 0.95)
    muon_momentum_warmup_start = cfg.optim_config.get('muon_momentum_warmup_start', 0.85)
    muon_momentum_warmup_steps = cfg.optim_config.get('muon_momentum_warmup_steps', 300)

    state = {
        "model": model,
        "optimizer": optimizer,
        "tokenizer": tokenizer,
        "data_scheduler": data_scheduler,
        "lr_scheduler": lr_scheduler,
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
        "packing_collator": packing_collator,
        "microbatch_step": 0,
        "optimizer_step": 0,
        "is_accumulating": False,
        "metrics": {"lr": 0.0, "grad_norm": 0.0, "current_batch_size": 0},
        "should_exit_training": False,
        "model_config": asdict(cfg.model_config) if is_dataclass(cfg.model_config) else cfg.model_config,
        "wandb_run_id": None,
        "flops_per_token": flops_per_token,
        "flop_breakdown": flop_breakdown,
        "cumulative_estimated_flops": 0,
        "_cached_recurrence_split": None,
        "_current_recurrence_split": None,
        "base_muon_wd": base_muon_wd,
        "muon_momentum": muon_momentum,
        "muon_momentum_warmup_start": muon_momentum_warmup_start,
        "muon_momentum_warmup_steps": muon_momentum_warmup_steps,
    }

    if cfg.recurrence_schedule == "constant" and is_recurrent_model:
        t, s = get_expected_recurrence_split(0, cfg)  # Constant means same at all steps
        state["_cached_recurrence_split"] = (t, s)
        fabric.print(f"Cached constant recurrence split: t={t}, s={s}")

    fabric.print(f"{time.ctime()[:-5]}: Checking for checkpoints...")
    load_checkpoint(fabric, state, cfg.out_dir, cfg.run_name, cfg.model_checkpoint, cfg.resume, cfg.resume_checkpoint_path)
    
    if state.get("wandb_run_id") and fabric.global_rank == 0 and cfg.logger_name == "wandb" and hasattr(fabric.logger, "experiment"):
        if fabric.logger.experiment.id != state["wandb_run_id"]:
            import wandb
            fabric.logger.experiment.finish()
            init_kwargs = dict(id=state["wandb_run_id"], resume="allow", project=cfg.logger_project, name=cfg.run_name, dir=cfg.out_dir)
            if cfg.logger_entity:
                init_kwargs["entity"] = cfg.logger_entity
            fabric.logger._experiment = wandb.init(**init_kwargs)
    
    if asdict(cfg.model_config) != state["model_config"]:
        fabric.print("-------------Warning, model config difference between checkpoint and model config!-------------")

    fabric.print(f"{time.ctime()[:-5]}: Logging hyperparameters...")
    fabric.print(f"cmdline + derived cfg:\n{json.dumps(cfg.__dict__, default=lambda x: x.__dict__, indent=4)}")
    if fabric.global_rank == 0:
        try:
            fabric.logger.log_hyperparams(cfg.__dict__)
        except Exception as e:
            fabric.print(f"Warning: WandB logging failed: {e}")

    fabric.print(f"{time.ctime()[:-5]}: Waiting at barrier before starting training...")
    fabric.barrier()
    end_time = time.time()
    fabric.print(f"{time.ctime()[:-5]}: Total time to run main func setups: {end_time - start_time:.02f} seconds.")

    return state


@torch.no_grad()
def validate(
    fabric: Fabric, model: nn.Module, val_dataloader: DataLoader, tokenizer: Tokenizer, cfg
) -> dict[str, torch.Tensor]:
    fabric.print(f"Validating for {cfg.eval_iters} steps ...")
    model.eval()
    
    eval_dtype = get_eval_precision_dtype(cfg.eval_precision, cfg.fabric_precision)
    autocast_dtype = get_eval_autocast_dtype(cfg.eval_precision)
    saved_state_dict = None
    
    if eval_dtype is not None and autocast_dtype is None:
        first_param = next(model.parameters())
        original_dtype = first_param.dtype
        if original_dtype != eval_dtype:
            fabric.print(f"Saving state dict and casting model from {original_dtype} to {eval_dtype} for evaluation")
            saved_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            model.to(eval_dtype)
    
    if autocast_dtype is not None:
        autocast_ctx = torch.autocast(device_type="cuda", dtype=autocast_dtype)
        fabric.print(f"Using autocast with dtype {autocast_dtype} for evaluation")
    else:
        autocast_ctx = nullcontext()

    def loss_fn(logits, labels):
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.view(-1), ignore_index=model.objective["ignore_index"]
        )

    metrics = {}

    mean_recurrence = getattr(model.config, 'mean_recurrence', None)
    if mean_recurrence is None:
        losses = torch.zeros(cfg.eval_iters, device=fabric.device)
        for k, (input_ids, labels, _) in enumerate(val_dataloader):
            if k >= cfg.eval_iters:
                break

            input_ids = input_ids.to(fabric.device, non_blocking=True)
            labels = labels.to(fabric.device, non_blocking=True)

            mask, positions = get_attention_mask(input_ids, tokenizer, cfg.cache_attn, cfg.doc_block_attn)
            with autocast_ctx:
                outputs = model(
                    input_ids, position_ids=positions, attention_mask=mask, return_logits=True
                )
            losses[k] = loss_fn(outputs["logits"], labels)

        global_val_loss = fabric.all_reduce(losses.mean())
        metrics["val_loss"] = global_val_loss
        metrics["val_ppl"] = global_val_loss.exp()
    else:
        losses = torch.zeros(cfg.eval_iters, len(cfg.partial_depth_eval) + 1, device=fabric.device)
        for idx, depth in enumerate(cfg.partial_depth_eval + [mean_recurrence]):
            for k, (input_ids, labels, _) in enumerate(val_dataloader):
                if k >= cfg.eval_iters:
                    break

                input_ids = input_ids.to(fabric.device, non_blocking=True)
                labels = labels.to(fabric.device, non_blocking=True)

                mask, positions = get_attention_mask(input_ids, tokenizer, cfg.cache_attn, cfg.doc_block_attn)
                rec_steps = torch.as_tensor([depth, 0])
                with autocast_ctx:
                    outputs = model(
                        input_ids, position_ids=positions, attention_mask=mask, return_logits=True, num_steps_pair=rec_steps
                    )
                losses[k, idx] = loss_fn(outputs["logits"], labels)

        global_val_loss = fabric.all_reduce(losses.mean(dim=0))  # dim-0 is the mbs dimension, dim-1 is kept after comms
        metrics["val_loss"] = global_val_loss[-1]
        metrics["val_ppl"] = global_val_loss[-1].exp()
        for idx, depth in enumerate(cfg.partial_depth_eval + [mean_recurrence]):
            metrics[f"val_loss_{depth}"] = global_val_loss[idx]
            metrics[f"val_ppl_{depth}"] = global_val_loss[idx].exp()

    if saved_state_dict is not None:
        fabric.print(f"Restoring original model weights after evaluation")
        model.load_state_dict(saved_state_dict)
        del saved_state_dict
    
    model.train()
    return metrics


def train_step(input_ids, labels, fabric, state, running_loss, running_ppl, cfg):
    model = state["model"]
    optimizer = state["optimizer"]
    data_scheduler = state["data_scheduler"]
    tokenizer = state["tokenizer"]
    metrics = state["metrics"]

    state["microbatch_step"] += 1
    model.step = state["microbatch_step"]
    if cfg.goldfish.strategy is not None:
        labels, _ = recpre.utils.apply_tld(labels=labels, settings=cfg.goldfish, ignore_index=tokenizer.pad_id)

    input_ids = input_ids.to(fabric.device, non_blocking=True)
    labels = labels.to(fabric.device, non_blocking=True)
    mask, positions = get_attention_mask(input_ids, tokenizer, cfg.cache_attn, cfg.doc_block_attn)

    if state["microbatch_step"] < cfg.shape_watching_steps:
        bsz, seq_len = input_ids.shape
        fabric.print(f"bsz: {bsz} | seq_len: {seq_len}")
        fabric.print(f"input_ids.shape: {input_ids.shape} | labels.shape: {labels.shape}")
    elif state["microbatch_step"] == cfg.shape_watching_steps and cfg.shape_watching_steps > 0:
        fabric.print("Silencing shape watching ...")
    state["is_accumulating"] = state["microbatch_step"] % cfg.gradient_accumulation_steps != 0
    monitor_step = cfg.model_telemetry and state["microbatch_step"] % cfg.log_step_interval == 0
    if monitor_step and not state["is_accumulating"]:
        model.module.apply(partial(enable_monitoring_on_step, extreme=cfg.extreme_telemetry))

    def tightly_scoped_fwd_bwd(model, input_ids, positions, labels, mask):
        if state["microbatch_step"] == 1:
            fabric.print(f"{time.ctime()[:-5]}: About to run first forward pass (this may take 10-30 min if compiling)...")
        with fabric.no_backward_sync(model, enabled=state["is_accumulating"]):
            outputs = model(input_ids, position_ids=positions, labels=labels, attention_mask=mask)
            if state["microbatch_step"] == 1:
                fabric.print(f"{time.ctime()[:-5]}: First forward pass complete, starting backward...")
            fabric.backward(outputs["loss"] / cfg.gradient_accumulation_steps, model=model)
            if state["microbatch_step"] == 1:
                fabric.print(f"{time.ctime()[:-5]}: First backward pass complete!")
            return outputs["loss"].detach(), outputs["log_ppl"].detach()

    loss, log_ppl = tightly_scoped_fwd_bwd(model, input_ids, positions, labels, mask)
    metrics["mbs_loss"] = loss
    running_loss.update(loss)
    running_ppl.update(log_ppl)

    if not cfg.allow_nonfinite_loss and not torch.isfinite(loss):
        fabric.print(f"Loss is {loss} on {socket.gethostname()}. Terminating ...")
        state["should_exit_training"] = True

    if not state["is_accumulating"]:
        current_step_lr = state["lr_scheduler"].get_lr(state["optimizer_step"])
        for param_group in optimizer.param_groups:
            # Skip groups with fixed learning rates (e.g., memory value parameters)
            if param_group.get("fixed_lr", False):
                continue
            param_group["lr"] = torch.as_tensor(current_step_lr * param_group["base_lr"])

        if state["base_muon_wd"] > 0 and cfg.max_steps > 0:
            current_wd = state["base_muon_wd"] * (1 - state["optimizer_step"] / cfg.max_steps)
            for param_group in optimizer.param_groups:
                if param_group.get("kind") == "muon":
                    param_group["weight_decay"] = max(0.0, current_wd)

        warmup_steps = state["muon_momentum_warmup_steps"]
        if warmup_steps > 0:
            frac = min(state["optimizer_step"] / warmup_steps, 1.0)
            current_momentum = state["muon_momentum_warmup_start"] + frac * (state["muon_momentum"] - state["muon_momentum_warmup_start"])
            for param_group in optimizer.param_groups:
                if param_group.get("kind") == "muon":
                    param_group["momentum"] = current_momentum

        metrics["grad_norm"] = fabric.clip_gradients(model, optimizer, max_norm=cfg.grad_clip, error_if_nonfinite=False)
        if torch.isfinite(metrics["grad_norm"]):
            if state["optimizer_step"] > 0:  # Skip first step if compiling or autotuning
                if cfg.compile_optimizer:
                    for param_group in optimizer.param_groups:
                        for param in param_group["params"]:
                            if param.grad is not None:
                                torch._dynamo.decorators.mark_static_address(param.grad)
                    torch.compile(optimizer.step, mode="max-autotune-no-cudagraphs")()
                else:
                    optimizer.step()
        else:
            if cfg.skip_nonfinite_grads:
                fabric.print(f"Grad norm non-finite! Optimizer step {state['optimizer_step'] + 1} skipped.")
            else:
                fabric.print(f"Grad norm non-finite! Optimizer step {state['optimizer_step'] + 1}. Terminating ...")
                state["should_exit_training"] = True

        if monitor_step:
            track_gradient_metrics(model, optimizer, metrics)
            model.module.apply(partial(disable_monitoring_and_retrieve_metrics, metrics=metrics))
        optimizer.zero_grad(set_to_none=not (cfg.fabric.use_apex_adamw or cfg.compile_optimizer))
        state["optimizer_step"] += 1
        if data_scheduler is not None:
            data_scheduler.step(state["microbatch_step"])
        cfg.gradient_accumulation_steps = get_batch_size(state["microbatch_step"], cfg)

        metrics["lr"] = current_step_lr
        metrics["current_batch_size"] = cfg.gradient_accumulation_steps * cfg.micro_batch_size * cfg.replicas
        if state["base_muon_wd"] > 0 and cfg.max_steps > 0:
            metrics["muon_wd"] = state["base_muon_wd"] * max(0.0, 1 - state["optimizer_step"] / cfg.max_steps)
        if state["muon_momentum_warmup_steps"] > 0:
            frac = min(state["optimizer_step"] / state["muon_momentum_warmup_steps"], 1.0)
            metrics["muon_momentum"] = state["muon_momentum_warmup_start"] + frac * (state["muon_momentum"] - state["muon_momentum_warmup_start"])

    if "flop_breakdown" in state and state["flop_breakdown"] is not None:
        tokens_this_step = cfg.micro_batch_size * cfg.block_size * fabric.world_size
        if state.get("_cached_recurrence_split") is not None:
            t, s = state["_cached_recurrence_split"]
        else:
            t, s = get_expected_recurrence_split(state["microbatch_step"], cfg)
        state["_current_recurrence_split"] = (t, s)
        flops_this_step = compute_flops_per_token_at_recurrence(state["flop_breakdown"], t, s) * tokens_this_step
        state["cumulative_estimated_flops"] += flops_this_step


def train(fabric, state, cfg):
    fabric.print(f"{time.ctime()[:-5]}: Starting warmup...")
    warmup_or_early_fail_allreduce(fabric)
    fabric.print(f"{time.ctime()[:-5]}: Warmup complete, creating iterator...")
    state["initial_step"] = state["last_logged_step"] = state["microbatch_step"]
    if state["microbatch_step"] > 0:
        fabric.print(f"=== CHECKPOINT RESUME DIAGNOSTICS ===")
        fabric.print(f"  Resuming from microbatch_step: {state['microbatch_step']}")
        fabric.print(f"  optimizer_step: {state.get('optimizer_step', 'N/A')}")
        # Check LR at resume point (LR scheduler uses optimizer_step)
        expected_lr = state["lr_scheduler"].get_lr(state.get("optimizer_step", 0))
        fabric.print(f"  Expected LR at optimizer_step {state.get('optimizer_step', 0)}: {expected_lr:.6e}")
        # Check optimizer has state
        if hasattr(state["optimizer"], "state"):
            params_with_state = sum(1 for p in state["optimizer"].state if state["optimizer"].state[p])
            fabric.print(f"  Optimizer params with state: {params_with_state}")
        fabric.print(f"=====================================")
    
    train_iterator = iter(state["train_dataloader"])
    # Skip one batch on resume to fix StatefulDataLoader off-by-one
    if state["microbatch_step"] > 0:
        next(train_iterator, None)
        fabric.print(f"Skipped one batch to fix StatefulDataLoader off-by-one on resume")
    fabric.print(f"{time.ctime()[:-5]}: Iterator created, setting up metrics...")

    running_loss = RunningMean(window=cfg.log_step_interval, sync_on_compute=False).to(fabric.device)
    running_log_ppl = RunningMean(window=cfg.log_step_interval, sync_on_compute=False).to(fabric.device)
    first_validation_passed = False
    fabric.barrier()
    state["total_t0"] = time.time()
    fabric.print(f"{time.ctime()[:-5]}: Training preparations finished, starting to iterate train data now.")

    step_time = 0
    first_batch = True
    resumed_from_checkpoint = state["microbatch_step"] > 0
    pre_resume_loss = None
    for input_ids, labels, _ in train_iterator:
        if first_batch:
            fabric.print(f"{time.ctime()[:-5]}: Got first batch, shape: {input_ids.shape}, starting train_step...")
            first_batch = False
        
        step_before_train = state["microbatch_step"]
        is_save_boundary = (step_before_train > 0 and step_before_train % cfg.save_step_interval == 0) or \
                           (step_before_train > 0 and (step_before_train - 1) % cfg.save_step_interval == 0) or \
                           (step_before_train > 0 and (step_before_train + 1) % cfg.save_step_interval == 0)
        if is_save_boundary or (resumed_from_checkpoint and step_before_train <= state["initial_step"] + 2):
            batch_hash = hash(tuple(input_ids.flatten()[:100].tolist())) % (10**8)
            fabric.print(f"[BATCH_HASH] Step {step_before_train}: hash={batch_hash}, input_ids[0,:5]={input_ids[0, :5].tolist()}")
        t0 = time.time()
        train_step(input_ids, labels, fabric, state, running_loss, running_log_ppl, cfg=cfg)
        step_time += time.time() - t0
        step = state["microbatch_step"]
        if resumed_from_checkpoint and pre_resume_loss is None:
            pre_resume_loss = state["metrics"].get("mbs_loss", None)
            if pre_resume_loss is not None:
                # Compute batch hash for comparison (to verify data matches)
                batch_hash = hash(tuple(input_ids.flatten()[:100].tolist())) % (10**8)
                fabric.print(f"=== FIRST STEP AFTER RESUME ===")
                fabric.print(f"  Step {step}, Loss: {pre_resume_loss.item():.4f}")
                fabric.print(f"  LR: {state['metrics'].get('lr', 'N/A')}")
                fabric.print(f"  Batch hash (first 100 tokens): {batch_hash}")
                fabric.print(f"================================")

        validate_regular = not state["is_accumulating"] and step % cfg.eval_step_interval == 0
        validate_at_the_end = state["optimizer_step"] >= cfg.max_steps - 1
        if validate_regular or validate_at_the_end:
            t0 = time.time()
            val_metrics = validate(fabric, state["model"], state["val_dataloader"], state["tokenizer"], cfg=cfg)
            td = time.time() - t0
            val_metrics["val_time"] = torch.as_tensor(td)

            fabric.print(f"Step {step}: Val loss {val_metrics['val_loss'].item():.4f}, Val time: {td:.2f}s")
            state["metrics"] |= val_metrics
            if not first_validation_passed:
                fabric.log_to_summary({"first_validation_passed": time.time() - global_start_time})
                first_validation_passed = True
            fabric.barrier()

            if torch.distributed.is_initialized() and (state["microbatch_step"] % 32) == 0:
                exit_tensor = torch.as_tensor([int(state["should_exit_training"])], device=fabric.device)
                torch.distributed.all_reduce(exit_tensor, torch.distributed.ReduceOp.MIN, async_op=False)
                state["should_exit_training"] = bool(exit_tensor.item())  # always cast back to bool

        # Log at an interval.
        if step % cfg.log_step_interval == 0 or (state["should_exit_training"] and (step % 32) == 0):
            log_step(fabric, state, running_loss, running_log_ppl, step_time, state["data_scheduler"], cfg)
            step_time = 0

        if state["should_exit_training"] and (state["microbatch_step"] % 32) == 0:
            fabric.print(f"{time.ctime()[:-5]}: Exiting training early in step {step} due to error signal received.")
            break

        maybe_save_checkpoint(fabric, state, cfg, is_accumulating=state["is_accumulating"])

        exit_step = cfg.exit_at_step if cfg.exit_at_step is not None else cfg.max_steps
        if state["optimizer_step"] >= exit_step - 1:
            fabric.print(f"{time.ctime()[:-5]}: Exiting training orderly after completion of {step + 1} steps.")
            break


def log_step(
    fabric: Fabric,
    state: dict,
    running_loss: RunningMean,
    running_log_ppl: RunningMean,
    accumulated_step_time: float,
    data_scheduler: Optional[DataScheduler],
    cfg: CLISettings,
):
    loss = running_loss.compute()
    log_ppl = running_log_ppl.compute()
    t1 = time.time()

    metrics = state["metrics"]

    avg_time_per_step = accumulated_step_time / (state["microbatch_step"] - state["last_logged_step"])
    tokens_per_step = cfg.micro_batch_size * cfg.block_size * fabric.world_size
    tokens_per_second = tokens_per_step / avg_time_per_step

    metrics |= {
        "local_loss": loss,
        "local_ppl": log_ppl.exp(),
        "microbatch_step": state["microbatch_step"],
        "optimizer_step": state["optimizer_step"],
        "steps/second": 1 / avg_time_per_step,
        "seconds/step": avg_time_per_step,
        "tokens/second": tokens_per_second,
        "remaining_time": (
            (t1 - state["total_t0"])
            / (state["microbatch_step"] - state["initial_step"])
            * (cfg.max_steps * cfg.gradient_accumulation_steps - state["microbatch_step"])
        ),
        "total_tokens": state["microbatch_step"] * tokens_per_step,
        "total_time": t1 - state["total_t0"],
    }
    if cfg.measure_utilization:
        max_memory_allocated_per_gpu = torch.cuda.max_memory_allocated(fabric.device) / 1024**3
        max_mem_reserved_per_gpu = torch.cuda.max_memory_reserved(fabric.device) / 1024**3
        torch.cuda.reset_peak_memory_stats(fabric.device)
        model_flops, tflops, mfu = get_MFU_metrics(tokens_per_second, fabric, state["model"], cfg.fabric_precision)
        metrics |= {
            "total_FLOPs": state["microbatch_step"] * tokens_per_step * model_flops,
            "FLOP/S": tflops,
            "model_flop_utilization": mfu,
            "max_mem_per_gpu": max_memory_allocated_per_gpu,
            "max_mem_reserved_per_gpu": max_mem_reserved_per_gpu,
        }

    if "flops_per_token" in state:
        # Simple estimate using target (post-ramp) flops_per_token
        metrics["total_estimated_FLOPs"] = metrics["total_tokens"] * state["flops_per_token"]
    if "cumulative_estimated_flops" in state:
        metrics["total_estimated_FLOPs_curriculum"] = state["cumulative_estimated_flops"]
        if "_current_recurrence_split" in state:
            t, s = state["_current_recurrence_split"]
            metrics["expected_forward_only_steps"] = t
            metrics["expected_backprop_depth"] = s
            metrics["expected_total_recurrence"] = t + s

    if "grad_norm" in metrics and metrics["grad_norm"] is not None:
        grad_norm = fabric.all_reduce(metrics["grad_norm"])
        metrics["global_grad_norm"] = grad_norm
    else:
        metrics["global_grad_norm"] = None
    metrics["global_loss"] = fabric.all_reduce(loss)

    if cfg.loss_guardrail_active:
        total_tokens = state["microbatch_step"] * cfg.micro_batch_size * cfg.block_size * fabric.world_size
        if total_tokens > 10_000_000_000 and metrics["global_loss"] > 6:  # after 10b tokens we're in slow descent
            fabric.print(
                f"Loss guard activated with loss {metrics['global_loss']} in step {state['microbatch_step']}. "
                f"Terminating ..."
            )
            state["should_exit_training"] = True

    metrics["global_train_ppl"] = fabric.all_reduce(log_ppl).exp()

    if data_scheduler is not None:
        curr_data_weights = data_scheduler.get_data_weights()
        curr_data_weights = dict(zip(cfg.dataset_names, curr_data_weights))

        curr_sample_count = data_scheduler.get_sample_count()
        curr_sample_count = fabric.all_reduce(curr_sample_count, reduce_op="sum")

        curr_epoch_count = data_scheduler.get_epoch_count()
        curr_epoch_count = fabric.all_reduce(curr_epoch_count, reduce_op="mean")

        for i, x in enumerate(curr_data_weights.keys()):
            metrics["data_scheduler_weight/" + x] = curr_data_weights[x]
            metrics["data_scheduler_norm_weight/" + x] = curr_data_weights[x] / sum(list(curr_data_weights.values()))
            metrics["data_scheduler_sample_count/" + x] = curr_sample_count[i]
            metrics["data_scheduler_epoch_count/" + x] = curr_epoch_count[i]

            state["data_scheduler_weight/" + x] = metrics["data_scheduler_weight/" + x]
            state["data_scheduler_norm_weight/" + x] = metrics["data_scheduler_norm_weight/" + x]
            state["data_scheduler_sample_count/" + x] = metrics["data_scheduler_sample_count/" + x]
            state["data_scheduler_epoch_count/" + x] = metrics["data_scheduler_epoch_count/" + x]

    fabric.log_dict(metrics, step=state["microbatch_step"])
    state["last_logged_step"] = state["microbatch_step"]

    step_timing = (
        f" steps/sec: {metrics['steps/second']:4.2f}  |"
        if metrics["steps/second"] >= 1.0
        else f" secs/step: {metrics['seconds/step']:4.2f}  |"
    )
    lr_str = f"{metrics['lr']:2.4e}" if "lr" in metrics and metrics["lr"] is not None else ""
    grad_norm_str = f"{metrics['global_grad_norm']:6.4e}" if metrics["global_grad_norm"] is not None else ""

    fabric.print(
        f"{time.ctime()[:-5]}\n"
        f"Step {metrics['microbatch_step']:>8}    | Loss: {metrics['global_loss']:7.4f} | {metrics['global_train_ppl']:9.2f} PPL     |"
        f" Update {metrics['optimizer_step']:>8}     |\n"
        f"{'(optimizer.step)' if not state['is_accumulating'] else ' ' * 16}"
        f" | LR: {lr_str:>10}| Grad norm: {grad_norm_str:>11} |{' ' * 19}|\n"
        f"                 | MFU : {metrics.get('model_flop_utilization', 0):6.2%}  | TFLOP/S : {metrics.get('FLOP/S', 0):5.2f}  |"
        f" tok/sec: {metrics['tokens/second']:8.1f} | {step_timing}\n"
        f"                 | Max mem allocated: {metrics.get('max_mem_per_gpu', 0):4.2f} GB       "
        f"| Max mem reserved: {metrics.get('max_mem_reserved_per_gpu', 0):4.2f} GB            |\n"
        f"                 | Tokens: {metrics['total_tokens'] / 1e9: 4.1f}B | exaFLOP: {metrics.get('total_FLOPs', 0) / 1e18:8.5f} |"
        f" Remaining time: {metrics['remaining_time'] / 3600 / 24:.2f} days             |"
    )
    state["metrics"] = {}


def create_dataloader(
    data_config: list[recpre.settings.DataEntry],
    batch_size: int,
    block_size: int,
    n_chunks: int,
    data_dir: str,
    fabric: Fabric,
    seed: int = 1337,
    *,
    cfg: CLISettings,
    tokenizer: Tokenizer,
    stateful: bool = True,
) -> tuple[StatefulDataLoader | DataLoader, Optional[DataSchedulerTracker], Optional[BestFitPackingCollator]]:
    global_data_dir = data_dir
    datasets = []
    for curr_config in data_config:
        if curr_config.type == "hfds":
            assert tokenizer is not None, "tokenizer must be provided for HuggingfaceDataset"
            assert curr_config.data_dir is not None, "data_dir must be provided for HuggingfaceDataset"
            dataset = HuggingfaceDataset(
                ds_name_or_path=curr_config.data_dir,  # this is a path to a previously save_to_disk'd hfds
                seed=seed,
                num_processes=fabric.world_size,
                process_rank=fabric.global_rank,
                data_id=curr_config.prefix,  # this is provided for logging, and schedule purposes
                return_data_id=curr_config.return_data_id,
                data_signature=curr_config.data_signature or cfg.data_signature,  # specification of the data fmt
                repetitions=curr_config.repetitions,  # repeat the dataset a number of times
            )
        elif "pqds" in curr_config.type:
            ParquetImpl = ParquetStreamPure if curr_config.type == "pqds-pure" else ParquetStream
            dataset = ParquetImpl(
                dataset_folder_path=curr_config.data_dir if curr_config.data_dir is not None else global_data_dir,
                seed=seed,
                shuffle=cfg.shuffle_blocks,
                shuffle_filenames=cfg.shuffle_filenames,
                num_processes=fabric.world_size,
                process_rank=fabric.global_rank,
                data_id=curr_config.prefix,
                data_signature=curr_config.data_signature or cfg.data_signature,
                repetitions=None,
                return_data_id=curr_config.return_data_id,
                prefix=curr_config.prefix,
                stateful=stateful,
                tokenizer=tokenizer,  # Pass tokenizer for raw text tokenization
            )
        elif curr_config.type == "rngds":  # debug option
            dataset = RandomTokensDataset(seed=seed, vocab_size=tokenizer.vocab_size, block_size=block_size)
        else:
            raise ValueError(f"Unsupported dataset type: {curr_config.type}")

        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(f"No data found at {data_dir}.")

    if len(datasets) > 1:
        raise ValueError("Not exported")
    else:
        combined_dataset = datasets[0]
        data_scheduler_tracker = None

    packing_collator = None
    loader_batch_size = batch_size
    if cfg.pack_sequences:
        # Pull enough documents to keep the best-fit buffer full while emitting
        # exactly one configured microbatch per dataloader item.
        packing_collator = BestFitPackingCollator(
            tokenizer=tokenizer,
            block_size=cfg.loader_block_size,
            add_bos=cfg.add_bos,
            add_eos=cfg.add_eos,
            buffer_size=cfg.pack_buffer_size,
        )
        combined_dataset = PackedIterableDataset(combined_dataset, packing_collator, n_rows=batch_size)
        parametrized_collate_fn = unwrap_singleton_collate
        loader_batch_size = 1
    else:
        parametrized_collate_fn = partial(
            generic_collate_fn,
            tokenizer=tokenizer,
            block_size=cfg.loader_block_size,
            pad_to_block_size=cfg.pad_to_block_size,
            add_bos=cfg.add_bos,
            add_eos=cfg.add_eos,
            collate_checks_enabled=cfg.collate_checks_enabled,
            all_block_size_tensors=cfg.all_block_size_tensors,
        )

    loader_class = StatefulDataLoader if stateful else DataLoader
    return (
        loader_class(
            combined_dataset,
            batch_size=loader_batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=parametrized_collate_fn,
            num_workers=cfg.dataloader_num_workers,
            prefetch_factor=4 if cfg.dataloader_num_workers > 0 else None,
        ),
        data_scheduler_tracker,
        packing_collator,
    )


def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric: Fabric,
    seed: int = 1337,
    *,
    cfg: CLISettings,
    tokenizer: Tokenizer,
) -> Tuple[StatefulDataLoader, Optional[DataLoader], DataSchedulerTracker, Optional[BestFitPackingCollator]]:
    fabric.print(f"Creating dataloaders with seed: {seed}")
    train_dataloader, data_scheduler_tracker, packing_collator = create_dataloader(
        cfg.data_config["train_data"],
        batch_size=batch_size,
        block_size=block_size,
        n_chunks=cfg.n_chunks,
        fabric=fabric,
        data_dir=cfg.train_data_dir,
        seed=seed,
        cfg=cfg,
        tokenizer=tokenizer,
    )
    val_dataloader, _, _ = (
        create_dataloader(
            cfg.data_config["val_data"],
            batch_size=batch_size,
            block_size=block_size,
            n_chunks=cfg.n_chunks,
            fabric=fabric,
            data_dir=cfg.val_data_dir,
            seed=seed,
            cfg=cfg,
            tokenizer=tokenizer,
            stateful=False,
        )
        if "val_data" in cfg.data_config
        else (None, None, None)
    )
    return train_dataloader, val_dataloader, data_scheduler_tracker, packing_collator  # type: ignore


def derive_precision(precision, strategy_details):
    import torch.distributed.fsdp

    param_dtype = torch.bfloat16 if "bf16" in precision else torch.float16 if "16" in precision else torch.float32
    reduce_dtype = torch.float32 if "mixed" in precision else param_dtype
    if r := strategy_details.all_reduce_dtype is not None:
        reduce_dtype = (
            torch.float16
            if r in ["16", "fp16", "fp16-mixed"]
            else torch.bfloat16
            if r in ["bf16", "bf16-mixed"]
            else torch.float32
        )
    return torch.distributed.fsdp.MixedPrecision(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        buffer_dtype=torch.float32,
        keep_low_precision_grads=False,
        # cast_forward_inputs=False,
    )


def get_eval_precision_dtype(eval_precision: str, train_precision: str) -> Optional[torch.dtype]:
    if eval_precision == "same":
        return None
    
    precision_map = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "bf16-mixed": torch.bfloat16,
        "fp16-mixed": torch.float16,
    }
    
    return precision_map.get(eval_precision, None)


def get_eval_autocast_dtype(eval_precision: str) -> Optional[torch.dtype]:
    if eval_precision in ["bf16-mixed", "fp16-mixed"]:
        return torch.bfloat16 if "bf16" in eval_precision else torch.float16
    return None


def get_attention_mask(input_ids, tokenizer, cache_attn=True, doc_block_attn=True):
    mask, position_ids = None, None
    return mask, position_ids


def get_lr(step: int, max_steps: int, cfg: CLISettings) -> float:
    base_lr = cfg.optim_config["lr"]
    if step < cfg.warmup_steps:
        return base_lr * step / cfg.warmup_steps
    if step > (max_steps - cfg.cooldown_steps):
        return max(base_lr * (max_steps - step) / cfg.cooldown_steps, cfg.min_lr)
    if step > max_steps:
        return cfg.min_lr
    decay_ratio = (step - cfg.warmup_steps) / (max_steps - cfg.warmup_steps)
    assert 0 <= decay_ratio <= 1
    if cfg.lr_schedule == "linear":
        return base_lr - decay_ratio * (base_lr - cfg.min_lr)
    elif cfg.lr_schedule in ["constant", "trapezoid"]:
        return base_lr
    elif cfg.lr_schedule == "cosine":
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return cfg.min_lr + coeff * (base_lr - cfg.min_lr)
    else:
        raise ValueError(f"Unsupported lr_schedule: {cfg.lr_schedule}")


def get_expected_recurrence(step: int, cfg: CLISettings) -> int:
    assert isinstance(cfg.model_config, RecurrentConfig), "Only recurrent models are supported"

    tgt_depth = cfg.model_config.mean_recurrence
    num_warmup_steps = cfg.cirriculum_steps

    if cfg.recurrence_schedule == "constant":
        return tgt_depth
    elif cfg.recurrence_schedule == "linear":
        # linear(tgt_depth, current_step) = ceil(tgt_depth * (current_step/num_warmup_steps))
        return math.ceil(tgt_depth * min(1.0, step / num_warmup_steps))
    elif cfg.recurrence_schedule == "1-sqrt":
        # f1-sqrt(tgt_depth, current_step) = ceil(tgt_depth * (1 - sqrt(1 - current_step/num_warmup_steps)))
        if step >= num_warmup_steps:
            return tgt_depth
        progress = step / num_warmup_steps
        return math.ceil(tgt_depth * (1.0 - math.sqrt(1.0 - progress)))
    else:
        raise ValueError(f"Unsupported recurrence_schedule: {cfg.recurrence_schedule}")


def get_expected_recurrence_split(step: int, cfg: CLISettings) -> Tuple[int, int]:
    if not isinstance(cfg.model_config, ParcaeConfig):
        return (0, 0)  # Not a recurrent model
    
    tgt_total = cfg.model_config.mean_recurrence
    tgt_backprop = cfg.model_config.mean_backprop_depth
    tgt_forward_only = max(tgt_total - tgt_backprop, 0)
    num_warmup_steps = cfg.cirriculum_steps
    curriculum_target = getattr(cfg.model_config, 'curriculum_target', 'forward')
    if cfg.recurrence_schedule == "constant" or step >= num_warmup_steps:
        progress = 1.0
    elif cfg.recurrence_schedule == "linear":
        progress = min(1.0, step / num_warmup_steps)
    elif cfg.recurrence_schedule == "1-sqrt":
        progress = 1.0 - math.sqrt(1.0 - min(1.0, step / num_warmup_steps))
    else:
        progress = 1.0
    if curriculum_target == "forward":
        t = max(1, math.ceil(progress * tgt_forward_only))
        s = tgt_backprop
    elif curriculum_target == "backward":
        t = tgt_forward_only
        s = max(1, math.ceil(progress * tgt_backprop))
    else:  # "both"
        t = max(1, math.ceil(progress * tgt_forward_only)) if tgt_forward_only > 0 else 0
        s = max(1, math.ceil(progress * tgt_backprop))
    
    return (t, s)


def get_batch_size(step: int, cfg: CLISettings) -> int:
    if step > cfg.batch_size_ramp:
        gradient_accumulation_steps = cfg.batch_size // cfg.micro_batch_size
    else:
        slope = step / cfg.batch_size_ramp
        gradient_accumulation_steps = math.ceil(slope * cfg.batch_size / cfg.micro_batch_size)
    return gradient_accumulation_steps


def load_checkpoint(fabric, state, out_dir, run_name, model_checkpoint, resume=True, resume_checkpoint_path=None):
    resume_ckpt = None
    t0 = time.time()
    if resume:
        fabric.print("-------------------- Model Load triggered ------------------------------")
        
        if resume_checkpoint_path:
            p = Path(resume_checkpoint_path)
            resume_ckpt = p.parent / p.stem.rsplit("_", 1)[0] if "_" in p.stem else p.with_suffix("")
            fabric.print(f"Using checkpoint: {resume_ckpt}")
        else:
            base_for_glob = Path(out_dir) / fabric.get_prefix_for_checkpoint()
            fabric.print(f"Globbing for checkpoint files in {base_for_glob}")
            # DDPStrategy saves without extension, others may add rank suffix like {name}_0.pth
            if fabric.strategy_name == "axonn_tp":
                ckpt_pattern = f"*/*-{run_name}_*.pth"
            elif fabric.strategy_name == "DDPStrategy":
                # Try without extension first (Lightning Fabric default), then with .pth
                ckpt_pattern = f"step-*-{run_name}"
            else:
                ckpt_pattern = f"*-{run_name}_*.pth"
            ckpt_paths = list(base_for_glob.glob(ckpt_pattern))
            # Also try with .pth extension for DDPStrategy
            if len(ckpt_paths) == 0 and fabric.strategy_name == "DDPStrategy":
                ckpt_paths = list(base_for_glob.glob(f"step-*-{run_name}.pth"))
            fabric.print(f"Found {len(ckpt_paths)} checkpoint(s) matching pattern")
            if len(ckpt_paths) == 0:
                fabric.print(f"No checkpoint found in {out_dir} to resume from.")
            else:
                # Extract step number from filename like "step-00005000-{run_name}"
                def extract_step(p):
                    name = p.stem if p.suffix else p.name  # handle with/without extension
                    # Format: step-XXXXXXXX-{run_name}
                    parts = name.split("-")
                    if len(parts) >= 2:
                        try:
                            return int(parts[1])
                        except ValueError:
                            return 0
                    return 0
                resume_ckpt = max(ckpt_paths, key=extract_step)
                fabric.print(f"Selected checkpoint: {resume_ckpt}")
                # For DDPStrategy, use the path as-is (fabric.load handles both with/without extension)
                if fabric.strategy_name != "DDPStrategy":
                    filename, directory = str(resume_ckpt.name), resume_ckpt.parents[0]
                    filename = filename[filename.find("step") :]
                    filename = filename.split(f"-{run_name}_")[0] + f"-{run_name}"  # split off rank info and .pth
                    if fabric.strategy_name == "axonn_tp":
                        directory = Path(out_dir) / fabric.get_prefix_for_checkpoint()
                    resume_ckpt = directory / filename
        
        if resume_ckpt is not None:
            fabric.print(f"Resuming training from {resume_ckpt}")
            pre_load_step = state["microbatch_step"]
            
            train_dataloader = state["train_dataloader"]
            val_dataloader = state.get("val_dataloader")
            packing_collator = state.get("packing_collator")
            # Don't restore lr_scheduler from checkpoint — it depends on current config's max_steps
            lr_scheduler = state.pop("lr_scheduler", None)
            
            ckpt_path = Path(resume_ckpt)
            if ckpt_path.is_dir():
                fabric.print(f"WARNING: resume_ckpt is a directory: {resume_ckpt}")
                possible_files = list(ckpt_path.glob("*.pt")) + list(ckpt_path.glob("*.pth")) + list(ckpt_path.glob("*.ckpt"))
                if possible_files:
                    ckpt_path = possible_files[0]
                    fabric.print(f"Using checkpoint file: {ckpt_path}")
            
            try:
                checkpoint_raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                fabric.print(f"Loaded raw checkpoint, keys: {list(checkpoint_raw.keys())}")
                if "train_dataloader" in checkpoint_raw:
                    _raw_dl = checkpoint_raw['train_dataloader']
                    fabric.print(f"  train_dataloader type: {type(_raw_dl)}")
                    if isinstance(_raw_dl, dict):
                        fabric.print(f"  train_dataloader keys: {list(_raw_dl.keys())}")
                    else:
                        _raw_ds = getattr(_raw_dl, 'dataset', None)
                        _raw_ds_state = getattr(_raw_ds, '_state', None) if _raw_ds is not None else None
                        if _raw_ds_state is not None:
                            fabric.print(f"  pickled dataset _state: file_idx={_raw_ds_state.get('file_idx')}, "
                                         f"row_group_idx={_raw_ds_state.get('row_group_idx')}, "
                                         f"buffer_len={len(_raw_ds_state.get('buffer', []))}")
                        else:
                            fabric.print(f"  pickled object has no dataset._state")
            except Exception as e:
                fabric.print(f"WARNING: Failed to load raw checkpoint: {e}")
                checkpoint_raw = {}
            
            _original_torch_load = torch.load
            torch.load = lambda *args, **kwargs: _original_torch_load(*args, weights_only=False, **{k: v for k, v in kwargs.items() if k != 'weights_only'})
            try:
                fabric.load(resume_ckpt, state, strict=False)
            finally:
                torch.load = _original_torch_load
            
            if lr_scheduler is not None:
                state["lr_scheduler"] = lr_scheduler
                fabric.print(f"LR scheduler NOT restored from checkpoint (using current config's max_steps)")
            
            # Restore dataset state only (not full dataloader state, which causes double-positioning)
            state["train_dataloader"] = train_dataloader
            state["val_dataloader"] = val_dataloader
            if "train_dataloader" in checkpoint_raw:
                saved_dl_state = checkpoint_raw["train_dataloader"]
                if isinstance(saved_dl_state, dict) and "dataset_state" in saved_dl_state:
                    ds_state = saved_dl_state["dataset_state"]
                    fabric.print(f"Restoring dataset state from checkpoint (dict path):")
                    fabric.print(f"  file_idx={ds_state.get('file_idx')}, "
                                 f"row_group_idx={ds_state.get('row_group_idx')}, "
                                 f"row_idx={ds_state.get('row_idx')}")
                    train_dataloader.dataset.load_state_dict(ds_state)
                elif not isinstance(saved_dl_state, dict) and hasattr(saved_dl_state, 'dataset'):
                    saved_ds = saved_dl_state.dataset
                    if hasattr(saved_ds, '_state') and saved_ds._state:
                        train_dataloader.dataset._state = saved_ds._state
                        fabric.print(f"Restoring dataset state from checkpoint (pickled path): "
                                     f"file_idx={saved_ds._state.get('file_idx')}, "
                                     f"row_group_idx={saved_ds._state.get('row_group_idx')}")
                    else:
                        fabric.print("WARNING: Pickled dataloader has no dataset state")
                else:
                    fabric.print(f"WARNING: Cannot extract dataset state from train_dataloader "
                                 f"(type={type(saved_dl_state).__name__}), data restarts from beginning")
            else:
                fabric.print("WARNING: No train_dataloader in checkpoint, data restarts from beginning")

            if packing_collator is not None and "packing_collator" in checkpoint_raw:
                pc_saved = checkpoint_raw["packing_collator"]
                if hasattr(pc_saved, 'doc_buffer'):
                    # Pickled BestFitPackingCollator object — use it directly
                    packing_collator = pc_saved
                    fabric.print(f"Packing collator restored (pickled object): "
                                 f"buffer_size={len(packing_collator.doc_buffer)}")
                elif isinstance(pc_saved, dict):
                    # State dict format — load into the fresh collator
                    packing_collator.load_state_dict(pc_saved)
                    fabric.print(f"Packing collator restored (state dict): "
                                 f"buffer_size={len(packing_collator.doc_buffer)}")
                else:
                    fabric.print(f"WARNING: Unknown packing_collator format: {type(pc_saved)}")
                dataset = getattr(state["train_dataloader"], "dataset", None)
                if isinstance(dataset, PackedIterableDataset):
                    dataset.collator = packing_collator
                    fabric.print("  Updated dataset collator to restored collator")
                elif hasattr(state["train_dataloader"], 'collate_fn'):
                    state["train_dataloader"].collate_fn = packing_collator
                    fabric.print(f"  Updated dataloader collate_fn to restored collator")
            if packing_collator is not None:
                state["packing_collator"] = packing_collator
            
            if "rng_state_torch" in checkpoint_raw and checkpoint_raw["rng_state_torch"] is not None:
                torch.set_rng_state(checkpoint_raw["rng_state_torch"])
                fabric.print(f"Restored PyTorch CPU RNG state")
            if "rng_state_cuda" in checkpoint_raw and checkpoint_raw["rng_state_cuda"] is not None:
                torch.cuda.set_rng_state(checkpoint_raw["rng_state_cuda"], device=fabric.device)
                fabric.print(f"Restored PyTorch CUDA RNG state")
            if "rng_state_numpy" in checkpoint_raw and checkpoint_raw["rng_state_numpy"] is not None:
                import numpy as np
                np.random.set_state(checkpoint_raw["rng_state_numpy"])
                fabric.print(f"Restored NumPy RNG state")
            if "rng_state_python" in checkpoint_raw and checkpoint_raw["rng_state_python"] is not None:
                import random
                random.setstate(checkpoint_raw["rng_state_python"])
                fabric.print(f"Restored Python RNG state")
            state["rng_state_torch"] = checkpoint_raw.get("rng_state_torch")
            state["rng_state_cuda"] = checkpoint_raw.get("rng_state_cuda")
            state["rng_state_numpy"] = checkpoint_raw.get("rng_state_numpy")
            state["rng_state_python"] = checkpoint_raw.get("rng_state_python")
            if state["microbatch_step"] == pre_load_step and state["microbatch_step"] == 0:
                fabric.print(f"WARNING: microbatch_step is still 0 after loading - checkpoint may not have loaded correctly!")
            
    if resume_ckpt is None and model_checkpoint is not None:
        fabric.print("-------------------- Pretrained Checkpoint Load triggered ------------------------------")
        fabric.print(f"Loaded full checkpoint (including optim and data state) from {model_checkpoint}")
        train_dataloader = state["train_dataloader"]
        packing_collator = state.get("packing_collator")
        
        lr_scheduler = state.pop("lr_scheduler", None)
        checkpoint_raw = torch.load(model_checkpoint, map_location="cpu", weights_only=False)
        
        _original_torch_load = torch.load
        torch.load = lambda *args, **kwargs: _original_torch_load(*args, weights_only=False, **{k: v for k, v in kwargs.items() if k != 'weights_only'})
        try:
            fabric.load(model_checkpoint, state, strict=False)
        finally:
            torch.load = _original_torch_load
        if lr_scheduler is not None:
            state["lr_scheduler"] = lr_scheduler
            fabric.print(f"LR scheduler NOT restored from checkpoint (using current config's max_steps)")
        
        if "train_dataloader" in checkpoint_raw:
            fabric.print("Restoring dataloader state from checkpoint...")
            dl_state = checkpoint_raw["train_dataloader"]
            if isinstance(dl_state, dict):
                dl_state_modified = dl_state.copy()
                dl_state_modified["_num_yielded"] = 0
                dl_state_modified["_sampler_iter_yielded"] = 0
                dl_state_modified["_iterator_finished"] = False
                train_dataloader.load_state_dict(dl_state_modified)
                fabric.print(f"  Dataloader state restored (with _num_yielded=0)")
            else:
                train_dataloader.load_state_dict(dl_state)
                fabric.print(f"  Restored full dataloader state")
        state["train_dataloader"] = train_dataloader
        
        if packing_collator is not None and "packing_collator" in checkpoint_raw:
            fabric.print("Restoring packing collator state from checkpoint...")
            pc_state = checkpoint_raw["packing_collator"]
            if isinstance(pc_state, dict):
                packing_collator.load_state_dict(pc_state)
            elif hasattr(pc_state, 'state_dict'):
                packing_collator.load_state_dict(pc_state.state_dict())
            fabric.print(f"  Restored packing collator buffer size: {len(packing_collator.doc_buffer)}")
        if packing_collator is not None:
            state["packing_collator"] = packing_collator
    if resume_ckpt or model_checkpoint:
        fabric.print(f"Loaded state is from step {state['microbatch_step']}")
        fabric.print(f"Loaded optimizer_step: {state.get('optimizer_step', 'NOT FOUND')}")
        
        opt = state["optimizer"]
        opt_state = opt.state if hasattr(opt, "state") else {}
        params_with_state = sum(1 for s in opt_state.values() if s)
        fabric.print(f"Optimizer has state for {params_with_state} parameters")
        
        total_m_norm = 0.0
        total_v_norm = 0.0
        adamw_steps = []
        for s in opt_state.values():
            if s:
                if 'm' in s:
                    total_m_norm += s['m'].float().norm().item()
                if 'v' in s:
                    total_v_norm += s['v'].float().norm().item()
                if 'step' in s:
                    adamw_steps.append(s['step'])
        
        fabric.print(f"  Total momentum (m) norm: {total_m_norm:.4f}")
        fabric.print(f"  Total second moment (v) norm: {total_v_norm:.4f}")
        if adamw_steps:
            fabric.print(f"  AdamW step counts: min={min(adamw_steps)}, max={max(adamw_steps)}")
        
        if total_m_norm < 1e-6 and total_v_norm < 1e-6 and state['microbatch_step'] > 100:
            fabric.print("  WARNING: Optimizer state appears empty despite training having progressed!")
            fabric.print("  This will cause training to restart from scratch optimizer-wise!")
        
        fabric.print(f"{time.ctime()[:-5]} : Time to load ckpt state: {time.time() - t0:.02f} seconds.")
        fabric.print("-------------------- Checkpoint loaded    ------------------------------")
    else:
        fabric.print("-------------------- No Checkpoint loaded ------------------------------")
    return resume_ckpt


def maybe_save_checkpoint(fabric, state, cfg, is_accumulating=False, force_save=False):
    t0 = time.time()
    prefix = fabric.get_prefix_for_checkpoint()
    fully_qualified_checkpoint_path = f"{cfg.out_dir}/{prefix}/step-{state['microbatch_step']:08d}-{cfg.run_name}"

    save_at_interval = not is_accumulating and state["microbatch_step"] % cfg.save_step_interval == 0
    if cfg.save_n_min_before_job_done is not None and (state["microbatch_step"] % 32) == 0:
        time_spent = time.time() - global_start_time
        remaining_time = cfg.global_total_time - time_spent
        remaining_time = remaining_time / 60.0
        remaining_time = fabric.all_reduce(remaining_time, reduce_op="mean")  # slowdown?
        save_before_timeout = remaining_time <= cfg.save_n_min_before_job_done
        if save_before_timeout:
            fabric.print(f"{time.ctime()[:-5]}: Saving at {remaining_time:.02f} minutes left")
            cfg.save_n_min_before_job_done = None  # reset
    else:
        save_before_timeout = False

    save_at_first_step = cfg.save_first_step and (state["microbatch_step"] == 0)
    effective_max = cfg.exit_at_step if cfg.exit_at_step is not None else cfg.max_steps
    save_at_last_step = cfg.save_last_step and (state["optimizer_step"] >= (effective_max - 1))

    if save_at_interval or save_at_last_step or save_at_first_step or save_before_timeout or force_save:
        fabric.print(f"--------------------- {time.ctime()[:-5]} Model Save triggered --------")
        fabric.print(f"Saving to {str(fully_qualified_checkpoint_path)!r}")

        if fabric.global_rank == 0 and cfg.logger_name == "wandb" and hasattr(fabric.logger, "experiment"):
            state["wandb_run_id"] = fabric.logger.experiment.id

        import numpy as np
        import random
        state["rng_state_torch"] = torch.get_rng_state()
        state["rng_state_cuda"] = torch.cuda.get_rng_state(device=fabric.device)
        state["rng_state_numpy"] = np.random.get_state()
        state["rng_state_python"] = random.getstate()

        if fabric.global_rank == 0:
            opt = state["optimizer"]
            opt_state = opt.state if hasattr(opt, "state") else {}
            params_with_state = sum(1 for s in opt_state.values() if s)
            total_m_norm = sum(s.get('m', torch.zeros(1)).float().norm().item() for s in opt_state.values() if s and 'm' in s)
            total_v_norm = sum(s.get('v', torch.zeros(1)).float().norm().item() for s in opt_state.values() if s and 'v' in s)
            fabric.print(f"  Optimizer state: {params_with_state} params, m_norm={total_m_norm:.2f}, v_norm={total_v_norm:.2f}")

        dl = state["train_dataloader"]
        ds = dl.dataset if hasattr(dl, 'dataset') else None
        if ds is not None and hasattr(ds, '_state'):
            fabric.print(f"  Dataset position: file_idx={ds._state.get('file_idx')}, "
                         f"row_group_idx={ds._state.get('row_group_idx')}, "
                         f"buffer_len={len(ds._state.get('buffer', []))}")

        fabric.save(fully_qualified_checkpoint_path, state)
        fabric.print(f"------------------- {time.ctime()[:-5]} Checkpoint saved ({time.time() - t0:.02f} seconds)")


def form_save_state(state):
    save_state_dict = {}
    model_state, optim_state = state_dict_helpers.get_state_dict(state["model"], state["optimizer"])
    save_state_dict["model"] = model_state
    save_state_dict["optimizer"] = optim_state
    save_state_dict["train_dataloader"] = state["train_dataloader"].state_dict()
    if state.get("packing_collator") is not None:
        save_state_dict["packing_collator"] = state["packing_collator"].state_dict()

    for key, value in state.items():
        if key not in ["optimizer", "model", "packing_collator"] and "dataloader" not in key:
            save_state_dict[key] = value
    return save_state_dict


def load_save_state(state, resume_ckpt):
    checkpoint_state = torch.load(resume_ckpt, map_location=torch.device("cpu"))
    state_dict_helpers.set_state_dict(
        state["model"],
        state["optimizer"],
        model_state_dict=checkpoint_state["model"],
        optim_state_dict=checkpoint_state["optimizer"],
        options=None,
    )
    state["train_dataloader"].load_state_dict(checkpoint_state["train_dataloader"])
    if "packing_collator" in checkpoint_state and state.get("packing_collator") is not None:
        state["packing_collator"].load_state_dict(checkpoint_state["packing_collator"])
    for key, value in checkpoint_state.items():
        if key not in ["optimizer", "model", "packing_collator"] and "dataloader" not in key:
            state[key] = value


def warmup_or_early_fail_allreduce(fabric):
    if torch.distributed.is_initialized():
        fabric.print("Staging allreduce warmup")
        device = fabric.device
        # Creating random data for warmup
        flat_params = torch.randn(128 * 1024 * 1024 // 4, device=device)
        num_stages = 8
        chunk_size = flat_params.numel() // num_stages

        for i in range(num_stages):
            end = min((i + 1) * chunk_size, flat_params.numel())
            chunk = flat_params[:end]
            torch.distributed.all_reduce(chunk)
            torch.cuda.current_stream().synchronize()  # Force completion
            fabric.print(f"Warmup stage {i} [{chunk.numel() // (1024 * 1024 // 4)} MB] really completed")

        torch.distributed.barrier()
        fabric.print(f"{time.ctime()[:-5]}: All warmup stages passed")


def _get_time_from_slurm() -> int:
    try:
        global_total_str_parse = os.popen("squeue -h -j $SLURM_JOBID -o %L").read()  # this is slow
        global_total_str_parse = global_total_str_parse.strip("\n")
        global_total_str_parse = [int(i) for i in re.split(":|-", global_total_str_parse)]
        if len(global_total_str_parse) == 4:
            global_total_time = (
                24 * 3600 * global_total_str_parse[0]
                + 3600 * global_total_str_parse[1]
                + 60 * global_total_str_parse[2]
                + global_total_str_parse[3]
            )
        elif len(global_total_str_parse) == 3:
            global_total_time = (
                3600 * global_total_str_parse[0] + 60 * global_total_str_parse[1] + global_total_str_parse[2]
            )
        elif len(global_total_str_parse) == 2:
            global_total_time = 60 * global_total_str_parse[0] + global_total_str_parse[1]
    except Exception as e:
        print(e)
        global_total_time = 9999999999999999
    return global_total_time


import sys
import datetime


def main():
    cfg: CLISettings = CLI(CLISettings)  # type: ignore

    if int(os.getenv("SLURM_PROCID", "0")) == 0:
        print("--------------------------------------------------------------------")
        print(f"------------------ Launching run {cfg.run_name}------------------")
        print("--------------------------------------------------------------------")
        print("--------------------------------------------------------------------")
        print(f"Platform: {sys.platform}, Python: {sys.version.split(' (')[0]}, PyTorch: {torch.__version__}")
        print(f"CPU threads: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.")
        driver = f"HIP/ROCM {torch.version.hip}" if torch.version.hip else f"CUDA: {torch.version.cuda}"
        print(f"GPU : {torch.cuda.get_device_name()}. {driver}.")

    set_torch_flags(cfg)
    
    wandb_run_id = None
    if cfg.resume and cfg.resume_checkpoint_path:
        try:
            ckpt = torch.load(cfg.resume_checkpoint_path, map_location="cpu")
            wandb_run_id = ckpt.get("wandb_run_id") if isinstance(ckpt, dict) else None
        except Exception:
            pass
    
    fabric = setup_fabric(cfg, wandb_run_id=wandb_run_id)
    fabric.print(f"{time.ctime()[:-5]}: Calling startup()...")
    state = startup(fabric, cfg)
    fabric.print(f"{time.ctime()[:-5]}: startup() complete, calling train()...")

    signal.signal(signal.SIGUSR1, lambda s, f: (fabric.print("SIGUSR1: saving checkpoint..."), state.__setitem__("should_exit_training", True), maybe_save_checkpoint(fabric, state, cfg, force_save=True)))

    train_time = time.time()
    try:
        train(fabric, state, cfg)
    except (KeyboardInterrupt, Exception) as e:
        fabric.print(f"Training interrupted: {e}. Saving emergency checkpoint...")
        maybe_save_checkpoint(fabric, state, cfg, is_accumulating=False, force_save=True)
        raise

    fabric.print("--------------------------------------------------------------------")
    fabric.print(f"Training time: {str(datetime.timedelta(seconds=time.time() - train_time))} ")
    fabric.log_to_summary(
        {"train_time": time.time() - global_start_time, "total_time": time.time() - global_start_time}
    )
    if fabric.device.type == "cuda":
        max_alloc = f"{torch.cuda.max_memory_allocated(fabric.device) / float(1024**3):,.3f} GB"
        max_reserved = f"{torch.cuda.max_memory_reserved(fabric.device) / float(1024**3):,.3f} GB"
        fabric.print(f"Max. Mem allocated: {max_alloc}. Max. Mem reserved: {max_reserved}.")
    fabric.print("--------------------------------------------------------------------")
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    if int(os.getenv("SLURM_PROCID", "0")) == 0:
        print(f"Run {cfg.run_name} finished without error.")
        print(f"---------Total time: {str(datetime.timedelta(seconds=time.time() - global_start_time))} ---------")
        print("-----------------Shutdown complete.--------------------------")


def guarded_main():
    try:
        main()
    except BaseException:  # gate around hell to guarantee NCCL deconstruction
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        if int(os.getenv("SLURM_PROCID", "0")) == 0:
            print("Run finished with errors.")
            print(f"---------Total time: {str(datetime.timedelta(seconds=time.time() - global_start_time))} ---------")
            print("-----------------Shutdown complete.--------------------------")

            raise


if __name__ == "__main__":
    guarded_main()
