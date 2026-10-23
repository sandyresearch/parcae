from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict

import torch

from receval.tasks.kv_cache_eval import KVCacheModelWrapper
from receval.tasks.val_loss import load_val_texts_from_parquet, run_val_loss


class EvalSettings:
    def __init__(self, cfg, device, is_main: bool) -> None:
        self.sequence_length = cfg.seq_len
        self.precision = cfg.precision
        self.device = device
        self.device_type = device.type
        self.ddp_rank = 0 if is_main else 1
        self.ddp_world_size = 1

    @property
    def _is_main_process(self) -> bool:
        return self.ddp_rank == 0

    def _get_autocast_context(self):
        if self.precision == "bf16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()


@torch.inference_mode()
def paired_kv_eval(model, tokenizer, cfg, device, is_main: bool):
    if not cfg.val_dir or cfg.final_eval_samples <= 0:
        return None

    texts = load_val_texts_from_parquet(cfg.val_dir, max_files=1)
    if not texts:
        raise RuntimeError(f"no validation texts found under {cfg.val_dir}")

    settings = EvalSettings(cfg, device, is_main)
    sample_count = min(cfg.final_eval_samples, len(texts))
    full = run_val_loss(
        model,
        tokenizer,
        settings,
        texts,
        max_samples=sample_count,
    )
    compressed = run_val_loss(
        KVCacheModelWrapper(
            model,
            lookup_strategy="compress-s1",
            num_steps=cfg.num_recurrence,
            mode="token_by_token",
        ),
        tokenizer,
        settings,
        texts,
        max_samples=sample_count,
    )
    full_metrics = asdict(full)
    compressed_metrics = asdict(compressed)
    return {
        "num_samples": sample_count,
        "full": full_metrics,
        "compress-s1": compressed_metrics,
        "delta": {
            key: compressed_metrics[key] - full_metrics[key]
            for key in ("loss", "perplexity", "bits_per_byte")
        },
    }
