"""Shared-KV recurrent fine-tuning trainer."""
from __future__ import annotations
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from jsonargparse import CLI
from recpre.schedulers import get_lr_scheduler

from recft.compile import (
    compile_transformer_blocks,
    configure_compile,
    unwrap_transformer_blocks,
)
from recft.data import PrefetchIterator, parquet_batches
from recft.eval import paired_kv_eval
from recft.lora import merged_state_dict, unwrap_lora
from recft.model import RecFTModule, configure_trainable_parameters
from recft.settings import FinetuneSettings
from recpre.monitor import (
    compute_grad_norm_buckets,
    disable_monitoring_and_retrieve_metrics,
    enable_monitoring_on_step,
)

TOKENIZER_ID = os.environ.get("PARCAE_TOKENIZER", "SandyResearch/parcae-tokenizer")


def save_torch(payload, path: str, attempts: int = 6) -> None:
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    for attempt in range(attempts):
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, temporary)
            temporary.replace(target)
            return
        except (OSError, RuntimeError):
            temporary.unlink(missing_ok=True)
            if attempt + 1 == attempts:
                raise
            time.sleep(5)


def run(cfg: FinetuneSettings):
    from parcae_lm.tokenizer import Tokenizer
    from receval.models.parcae import ModelingParcae
    import torch.distributed as dist

    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    ddp = world > 1
    if ddp:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    is_main = (rank == 0)
    process_seed = cfg.seed + rank
    random.seed(process_seed)
    torch.manual_seed(process_seed)
    if is_main:
        print(f"[seed] training={cfg.seed}", flush=True)

    micro = cfg.micro_batch_size
    denom = world * micro
    if cfg.world_batch_size % denom != 0:
        raise SystemExit(f"world_batch_size {cfg.world_batch_size} is not divisible by "
                         f"world({world}) * micro_batch_size({micro}) = {denom}")
    accum = cfg.world_batch_size // denom
    eff = world * micro * accum
    T = cfg.num_recurrence
    if is_main:
        print(f"[dist] world={world} micro={micro} accum={accum} -> EFFECTIVE BATCH {eff} seq "
              f"({eff*cfg.seq_len:,} tok/step; {cfg.steps*eff*cfg.seq_len:,} tok total)", flush=True)

    dev = torch.device("cuda", local_rank) if ddp else torch.device("cuda")
    if cfg.precision not in {"bf16", "fp32"}:
        raise SystemExit(f"precision must be bf16 or fp32, got {cfg.precision}")
    model = ModelingParcae.from_pretrained(
        cfg.model,
        device=str(dev),
        dtype=torch.float32,
    ).eval()
    tok = Tokenizer(TOKENIZER_ID)
    trainable_names = configure_trainable_parameters(model, cfg.trainable, cfg.lora)
    if is_main:
        trainable_count = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        print(
            f"[params] mode={cfg.trainable} tensors={len(trainable_names)} "
            f"parameters={trainable_count:,}",
            flush=True,
        )

    oc = cfg.optim_config
    if cfg.optimizer.lower() == "muonadamw":
        from recpre.optim import get_muon_param_groups_from_config
        from recpre.optim.muon_adamw import MuonAdamW
        mlr = oc.muon_lr if oc.muon_lr is not None else oc.lr * (8e-3 / 6e-3)
        ocd = dict(lr=oc.lr, weight_decay=oc.weight_decay, betas=tuple(oc.betas), eps=oc.eps,
                   muon_lr=mlr, muon_momentum=0.95, muon_wd=oc.muon_wd, muon_ns_steps=5)
        groups = get_muon_param_groups_from_config(
            (
                (name, parameter)
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            ),
            ocd,
            True,
            verbose=False,
        )
        opt = MuonAdamW(groups)
        if is_main:
            na = sum(p.numel() for g in groups if g.get("kind") == "adamw" for p in g["params"])
            nm = sum(p.numel() for g in groups if g.get("kind") == "muon" for p in g["params"])
            print(f"[optim] MuonAdamW: adamw_lr={oc.lr:g} muon_lr={mlr:g} betas={tuple(oc.betas)} "
                  f"eps={oc.eps:g} adamw_wd={oc.weight_decay:g} muon_wd={oc.muon_wd:g} | Muon {nm/1e6:.1f}M "
                  f"({100*nm/(na+nm):.1f}%), AdamW {na/1e6:.1f}M", flush=True)
    else:
        opt = torch.optim.AdamW(
            (
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            lr=oc.lr,
            weight_decay=oc.weight_decay,
            betas=tuple(oc.betas),
        )
        if is_main:
            print(f"[optim] AdamW lr={oc.lr:g} betas={tuple(oc.betas)} wd={oc.weight_decay:g}", flush=True)
    base_lrs = [g["lr"] for g in opt.param_groups]

    params = [p for p in model.parameters() if p.requires_grad]
    os.makedirs(cfg.out_dir, exist_ok=True)
    out_json = os.path.join(cfg.out_dir, "run.json")
    log = {
        "config": asdict(cfg),
        "provenance": {"tokenizer": TOKENIZER_ID},
        "history": [],
        "dist": {
            "world": world,
            "micro": micro,
            "accum": accum,
            "eff_batch_seq": eff,
        },
    }
    ck = torch.load(cfg.resume, map_location="cpu", weights_only=False) if cfg.resume else None
    # Fresh runs get their own wandb run; a resumed run continues the one stored in its ckpt.pt.
    resumed_wandb_id = ck.get("wandb_id") if ck is not None else None
    wandb_run = None
    if is_main and cfg.wandb_project:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "wandb_project is set but wandb is not installed"
            ) from exc
        wandb_run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity or None,
            group=cfg.wandb_group or None,
            name=cfg.run_name,
            id=resumed_wandb_id,
            resume="must" if resumed_wandb_id else None,
            tags=cfg.wandb_tags or None,
            config=log["config"],
        )

    start_step = 0
    if ck is not None:
        model.load_state_dict({k: v.to(dev, torch.float32) for k, v in ck["model"].items()}, strict=True)
        cw, ca = ck.get("world"), ck.get("accum")
        if cw is not None and (cw != world or ca != accum):
            raise SystemExit(f"[resume] topology mismatch: ckpt was world={cw} accum={ca}, this run is "
                             f"world={world} accum={accum}. They must match.")
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"]
        # Poisson depths are seeded from model.step (one increment per micro-batch forward).
        model.step = ck.get("model_step", start_step * accum)
        if ck.get("rng") is not None:
            rng = ck["rng"][rank]
            torch.set_rng_state(rng["cpu"])
            torch.cuda.set_rng_state(rng["cuda"], dev)
            random.setstate(rng["python"])
        base_lrs = ck.get("base_lrs", base_lrs)
        if "history" in ck:
            log["history"] = ck["history"]
        if is_main:
            print(f"[resume] {cfg.resume}: continuing from step {start_step} "
                  f"(optimizer state restored)", flush=True)

    schedule_steps = cfg.schedule_steps or cfg.steps
    warmup_steps = (
        round(schedule_steps * cfg.lr_schedule.warmup_frac)
        if cfg.lr_schedule.schedule == "wsd"
        else 0
    )
    cooldown_steps = round(schedule_steps * cfg.lr_schedule.decay_frac)
    if not 0 <= cfg.lr_schedule.min_lr_ratio <= 1:
        raise ValueError("min_lr_ratio must be in [0, 1]")
    if warmup_steps + cooldown_steps > schedule_steps:
        raise ValueError("warmup and decay phases exceed total steps")
    lr_scheduler = get_lr_scheduler(
        "trapezoid",
        base_lr=1.0,
        min_lr=cfg.lr_schedule.min_lr_ratio,
        warmup_steps=warmup_steps,
        cooldown_steps=cooldown_steps,
        max_steps=schedule_steps,
    )

    if cfg.backprop.mode == "last4first":
        grad_loops = {0} | set(range(max(0, T - 4), T))
    else:
        grad_loops = None
    if is_main:
        if cfg.recurrence_sampling == "poisson":
            print(
                f"[bptt] mode=pretrain-iterate "
                f"sampling=poisson-truncated-full "
                f"iteration={getattr(model, 'recurrent_iteration_method', '?')} "
                f"mean_backprop_depth={getattr(model.config, 'mean_backprop_depth', '?')}",
                flush=True,
            )
        else:
            print(
                f"[bptt] mode={cfg.backprop.mode} grad_loops="
                f"{sorted(grad_loops) if grad_loops else 'ALL'} "
                f"recurrence_sampling={cfg.recurrence_sampling}",
                flush=True,
            )

    if (
        cfg.recurrence_sampling != "fixed"
        and cfg.compile_model
        and cfg.compile_backend == "full"
    ):
        raise ValueError(
            "stochastic recurrence depth is incompatible with full-model "
            "torch.compile(dynamic=False)"
        )

    if cfg.compile_model:
        configure_compile(
            local_rank,
            optimize_ddp=False,
            compiled_autograd=False,
        )
        if cfg.compile_backend == "selective":
            compile_transformer_blocks(model, cfg.compile_mode)
            if is_main:
                print(
                    f"[compile] transformer blocks mode={cfg.compile_mode}",
                    flush=True,
                )

    objective = RecFTModule(
        model,
        num_steps=T,
        aux=cfg.aux,
        gamma=cfg.gamma,
        grad_loops=grad_loops,
        precision=cfg.precision,
        swap_detach_hidden=cfg.swap_detach_hidden,
        recurrence_sampling=cfg.recurrence_sampling,
        lockstep_n=cfg.lockstep_n,
    )
    if cfg.compile_model and cfg.compile_backend == "full":
        objective = torch.compile(
            objective,
            mode=cfg.compile_mode,
            fullgraph=False,
            dynamic=False,
        )
        if is_main:
            print(f"[compile] RecFTModule mode={cfg.compile_mode}", flush=True)
    train_module = objective
    ddp_module = None
    if ddp:
        from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import (
            bf16_compress_hook,
        )
        from torch.nn.parallel import DistributedDataParallel

        ddp_module = DistributedDataParallel(
            objective,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            bucket_cap_mb=200,
            gradient_as_bucket_view=True,
            find_unused_parameters=True,
        )
        if cfg.bf16_grad_comm:
            ddp_module.register_comm_hook(None, bf16_compress_hook)
        train_module = ddp_module

    if not cfg.train_dir:
        raise SystemExit("train_dir is required (a parquet dir held out from pretraining)")
    source = parquet_batches(
        cfg.train_dir,
        tok,
        cfg.seq_len,
        micro,
        (cfg.steps - start_step) * accum,
        rank=rank,
        world_size=world,
        skip_batches=start_step * accum,
        seed=cfg.seed,
    )
    it = PrefetchIterator(source, depth=2)

    step = start_step
    t0 = time.time()
    save_at = set(int(s) for s in cfg.save_at)
    grad_norm_ema = 0.0
    consecutive_abort_norms = 0

    def save_resume_ckpt(at_step: int) -> None:
        """Write ckpt.pt: weights, optimizer, step, the depth-sampler step and every rank's RNG state."""
        rng = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(dev), "python": random.getstate()}
        rngs = [None] * world if is_main else None
        if ddp:
            dist.gather_object(rng, rngs, dst=0)
        else:
            rngs = [rng]
        if not is_main:
            return
        ck_path = os.path.join(cfg.out_dir, "ckpt.pt")
        save_torch(
            {
                "model": {k.replace("._orig_mod.", "."): v.float().cpu() for k, v in model.state_dict().items()},
                "opt": opt.state_dict(),
                "step": at_step,
                "model_step": int(getattr(model, "step", 0)),
                "rng": rngs,
                "history": log["history"],
                "world": world,
                "accum": accum,
                "base_lrs": base_lrs,
                "wandb_id": wandb_run.id if wandb_run is not None else None,
            },
            ck_path,
        )
        print(f"wrote {ck_path} (resumable @ step {at_step})", flush=True)

    model.train()
    while step < cfg.steps:
        spike_metrics: dict[str, float] = {}
        telemetry_metrics: dict[str, float] = {}
        telemetry_step = (
            cfg.model_telemetry
            and cfg.telemetry_every > 0
            and (step + 1) % cfg.telemetry_every == 0
        )
        active_gamma = (
            cfg.gamma_after
            if cfg.gamma_end_step is not None
            and step >= cfg.gamma_end_step
            else cfg.gamma
        )
        objective.gamma = active_gamma
        if cfg.compile_model:
            torch.compiler.cudagraph_mark_step_begin()
        opt.zero_grad(set_to_none=True)
        ce_v = aux_v = depth_v = 0.0
        batches = []
        for _ in range(accum):
            try:
                batches.append(next(it))
            except StopIteration:
                break
        if not batches:
            if is_main:
                print("[data] exhausted; stopping", flush=True)
            break

        got = len(batches)
        for micro_index, cpu_ids in enumerate(batches):
            ids = cpu_ids.to(dev, non_blocking=True)
            should_sync = micro_index == got - 1
            if telemetry_step and should_sync:
                model.apply(partial(enable_monitoring_on_step, extreme=False))
            sync_context = (
                ddp_module.no_sync()
                if ddp and not should_sync
                else nullcontext()
            )
            with sync_context:
                total, ce, aux = train_module(ids)
                (total / got).backward()
            ce_v += float(ce.detach()) / got
            aux_v += float(aux.detach()) / got
            depth_v += float(objective.last_num_steps) / got
        if telemetry_step:
            raw_metrics: dict[str, object] = {}
            model.apply(
                partial(
                    disable_monitoring_and_retrieve_metrics,
                    metrics=raw_metrics,
                )
            )
            for name, value in raw_metrics.items():
                if torch.is_tensor(value):
                    value = value.detach().float().mean().item()
                telemetry_metrics[f"recurrent/{name}"] = float(value)

        sc = (
            1.0
            if cfg.lr_schedule.schedule == "none"
            else lr_scheduler.get_lr(step)
        )
        if cfg.lr_schedule.schedule != "none":
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * sc
        grad_norm = torch.nn.utils.clip_grad_norm_(params, oc.grad_clip)
        grad_norm_value = float(grad_norm)
        spike = (
            cfg.spike_detection_enabled
            and grad_norm_ema > 0
            and step > cfg.spike_warmup_steps
            and grad_norm_value > cfg.spike_factor * grad_norm_ema
        )
        bucket_norms = None
        if spike or telemetry_step:
            bucket_norms = compute_grad_norm_buckets(model)
            bucket_norms.sort(key=lambda item: item[1].item(), reverse=True)
            undo_clip = max(1.0, grad_norm_value / oc.grad_clip)
        if telemetry_step and bucket_norms is not None:
            for name, norm_value, fraction in bucket_norms:
                telemetry_metrics[f"grad/{name}_norm"] = (
                    float(norm_value) * undo_clip
                )
                telemetry_metrics[f"grad/{name}_fraction"] = float(fraction)
        if spike and bucket_norms is not None:
            for name, norm_value, fraction in bucket_norms[: cfg.spike_log_top_k]:
                spike_metrics[f"spike/{name}_grad_norm"] = (
                    float(norm_value) * undo_clip
                )
                spike_metrics[f"spike/{name}_frac"] = float(fraction)
            if is_main:
                entry = {
                    "step": step,
                    "grad_norm": grad_norm_value,
                    "ema": grad_norm_ema,
                    "num_recurrence": depth_v,
                    "buckets": [
                        (name, float(norm_value) * undo_clip, float(fraction))
                        for name, norm_value, fraction in bucket_norms
                    ],
                }
                with open(
                    os.path.join(cfg.out_dir, "spike_log.jsonl"), "a"
                ) as spike_file:
                    spike_file.write(json.dumps(entry) + "\n")
                leaders = ", ".join(
                    f"{name}={float(norm_value) * undo_clip:.2f}"
                    for name, norm_value, _ in bucket_norms[:3]
                )
                print(
                    f"[grad-spike] step={step} norm={grad_norm_value:.2f} "
                    f"ema={grad_norm_ema:.2f} top={leaders}",
                    flush=True,
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "spike/event": 1,
                            "spike/grad_norm": grad_norm_value,
                            "spike/grad_norm_ema": grad_norm_ema,
                            "spike/num_recurrence": depth_v,
                            **spike_metrics,
                        },
                        step=step,
                    )
        grad_norm_ema = (
            cfg.spike_ema_decay * grad_norm_ema
            + (1 - cfg.spike_ema_decay) * grad_norm_value
            if grad_norm_ema > 0
            else grad_norm_value
        )
        if (
            cfg.grad_norm_abort_threshold is not None
            and grad_norm_value > cfg.grad_norm_abort_threshold
        ):
            consecutive_abort_norms += 1
        else:
            consecutive_abort_norms = 0
        if consecutive_abort_norms >= cfg.grad_norm_abort_patience:
            if is_main:
                print(
                    f"[grad-guard] stopping before optimizer step {step}: "
                    f"norm exceeded {cfg.grad_norm_abort_threshold:g} for "
                    f"{consecutive_abort_norms} consecutive steps",
                    flush=True,
                )
            break
        opt.step()
        step += 1

        if is_main and cfg.log_every and step % cfg.log_every == 0:
            el = time.time() - t0
            peak_allocated_gib = torch.cuda.max_memory_allocated(dev) / 2**30
            peak_reserved_gib = torch.cuda.max_memory_reserved(dev) / 2**30
            record = {
                "step": step,
                "tokens": step * eff * cfg.seq_len,
                "train/ce": ce_v,
                "train/aux": aux_v,
                "train/loss": ce_v + active_gamma * aux_v,
                "train/gamma": active_gamma,
                "train/num_recurrence": depth_v,
                "train/grad_norm": float(grad_norm),
                "optim/lr": opt.param_groups[0]["lr"],
                "optim/lr_scale": sc,
                "optim/adamw_weight_decay": next(
                    (
                        float(group["weight_decay"])
                        for group in opt.param_groups
                        if group.get("kind") == "adamw"
                    ),
                    0.0,
                ),
                "optim/muon_weight_decay": next(
                    (
                        float(group["weight_decay"])
                        for group in opt.param_groups
                        if group.get("kind") == "muon"
                    ),
                    0.0,
                ),
                "time/seconds_per_step": el / max(step - start_step, 1),
                "time/elapsed_minutes": el / 60,
                "memory/peak_allocated_gib": peak_allocated_gib,
                "memory/peak_reserved_gib": peak_reserved_gib,
            }
            record.update(spike_metrics)
            record.update(telemetry_metrics)
            log["history"].append(record)
            if wandb_run is not None:
                wandb_run.log(record, step=step)
            print(f"[{step:5d}/{cfg.steps}] ce={ce_v:.4f} aux={aux_v:.4f} "
                  f"T={depth_v:.2f} "
                  f"lr={opt.param_groups[0]['lr']:.3e} "
                  f"mem={peak_allocated_gib:.1f}/{peak_reserved_gib:.1f}GiB "
                  f"({el/max(step-start_step,1):.2f}s/it, {el/60:.1f}min elapsed)", flush=True)
            torch.cuda.reset_peak_memory_stats(dev)

        if cfg.max_seconds and (time.time() - t0) > cfg.max_seconds:
            if is_main:
                print(f"[wall-guard] {time.time()-t0:.0f}s > max_seconds {cfg.max_seconds}; "
                      f"stopping at step {step} and saving", flush=True)
            break

        if cfg.ckpt_every and step % cfg.ckpt_every == 0 and step < cfg.steps:
            save_resume_ckpt(step)

        if is_main and step in save_at:
            mp = os.path.join(cfg.out_dir, f"step{step}_model.pt")
            save_torch(
                {k: v.half().cpu() for k, v in merged_state_dict(model).items()},
                mp,
            )
            print(f"[snapshot] step {step} -> {mp}", flush=True)
        if ddp and step in save_at:
            dist.barrier()

    opt.zero_grad(set_to_none=True)
    unwrap_transformer_blocks(model)
    model.eval()
    torch.cuda.empty_cache()
    final_eval = paired_kv_eval(model, tok, cfg, dev, is_main)
    if is_main and final_eval is not None:
        log["final_eval"] = final_eval
        d = final_eval["delta"]
        if wandb_run is not None:
            wandb_run.log(
                {
                    "eval/full_loss": final_eval["full"]["loss"],
                    "eval/full_ppl": final_eval["full"]["perplexity"],
                    "eval/full_bpb": final_eval["full"]["bits_per_byte"],
                    "eval/compress_s1_loss": final_eval["compress-s1"]["loss"],
                    "eval/compress_s1_ppl": final_eval["compress-s1"]["perplexity"],
                    "eval/compress_s1_bpb": final_eval["compress-s1"]["bits_per_byte"],
                    "eval/delta_loss": d["loss"],
                    "eval/delta_ppl": d["perplexity"],
                    "eval/delta_bpb": d["bits_per_byte"],
                },
                step=step,
            )
        print(
            "[final-eval] "
            f"full_ppl={final_eval['full']['perplexity']:.4f} "
            f"compress_s1_ppl={final_eval['compress-s1']['perplexity']:.4f} "
            f"delta_ppl={d['perplexity']:+.4f} "
            f"delta_bpb={d['bits_per_byte']:+.4f}",
            flush=True,
        )

    if cfg.save_ckpt:
        save_resume_ckpt(step)
    if is_main:
        json.dump(log, open(out_json, "w"), indent=2)
        sd_path = os.path.join(cfg.out_dir, "model.pt")
        save_torch(
            {k: v.half().cpu() for k, v in merged_state_dict(model).items()},
            sd_path,
        )
        print(f"wrote {out_json} and {sd_path}", flush=True)
        if cfg.export_hf:
            hf_dir = os.path.join(cfg.out_dir, "hf")
            unwrap_lora(model, merge=True)
            model.save_pretrained(hf_dir)
            tok.processor.save_pretrained(hf_dir)
            print(f"wrote {hf_dir} (HuggingFace-compatible export)", flush=True)
        if wandb_run is not None:
            wandb_run.finish()
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


def main():
    cfg: FinetuneSettings = CLI(FinetuneSettings)
    run(cfg)


if __name__ == "__main__":
    main()
