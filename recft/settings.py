from dataclasses import dataclass, field
from typing import Literal


@dataclass
class OptimConfig:
    lr: float = 5e-5
    muon_lr: float | None = None
    betas: tuple[float, float] = (0.8, 0.95)
    eps: float = 1e-10
    weight_decay: float = 0.0
    muon_wd: float = 0.2
    grad_clip: float = 1.0


@dataclass
class LRSchedule:
    schedule: Literal["none", "linear", "wsd"] = "none"
    warmup_frac: float = 0.0
    decay_frac: float = 1.0
    min_lr_ratio: float = 0.0


@dataclass
class BackpropConfig:
    mode: Literal["full", "last4first"] = "full"


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0  # update scale is alpha / rank
    targets: list[str] | None = None  # None: core layer 0 K/V projections for the model's attention type


@dataclass
class FinetuneSettings:
    run_name: str = "parcae-ft"
    model: str = "SandyResearch/parcae-770m"
    out_dir: str = "outputs/parcae-ft"
    resume: str | None = None
    seed: int = 42

    aux: Literal["swap_ce", "none"] = "swap_ce"
    gamma: float = 0.01
    gamma_end_step: int | None = None
    gamma_after: float = 0.0
    swap_detach_hidden: bool = True
    trainable: Literal["all", "freeze_prelude_input", "lora"] = "all"
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    optimizer: str = "MuonAdamW"
    optim_config: OptimConfig = field(default_factory=OptimConfig)
    lr_schedule: LRSchedule = field(default_factory=LRSchedule)
    backprop: BackpropConfig = field(default_factory=BackpropConfig)
    precision: Literal["bf16", "fp32"] = "bf16"
    compile_model: bool = False
    compile_backend: Literal["selective", "full"] = "selective"
    compile_mode: str = "max-autotune-no-cudagraphs"
    bf16_grad_comm: bool = True

    world_batch_size: int = 256
    micro_batch_size: int = 4
    seq_len: int = 2048
    num_recurrence: int = 8
    recurrence_sampling: Literal["fixed", "poisson"] = "fixed"
    lockstep_n: bool = False
    steps: int = 313
    schedule_steps: int | None = None

    train_dir: str = ""
    val_dir: str = ""

    save_at: list[int] = field(default_factory=list)
    save_ckpt: bool = True
    ckpt_every: int = 0  # also write the resumable ckpt.pt every N steps (0 = only at the end)
    export_hf: bool = True
    final_eval_samples: int = 1
    log_every: int = 25
    model_telemetry: bool = False
    telemetry_every: int = 25
    max_seconds: float = 0.0
    spike_detection_enabled: bool = True
    spike_factor: float = 5.0
    spike_warmup_steps: int = 50
    spike_ema_decay: float = 0.98
    spike_log_top_k: int = 20
    grad_norm_abort_threshold: float | None = None
    grad_norm_abort_patience: int = 2

    wandb_project: str = "parcae-kv-finetune"
    wandb_entity: str = ""
    wandb_group: str = "shared-kv"
    wandb_tags: list[str] = field(default_factory=list)
