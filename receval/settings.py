import os
import glob
import torch
import yaml
import json
from dataclasses import dataclass, field
from typing import Optional, Literal, Any
from pathlib import Path


@dataclass
class LMEvalTaskSettings:
    tasks: list[str] = field(default_factory=lambda: ["hellaswag"])
    num_fewshot: Optional[int] = None
    batch_size: int = 32
    limit: Optional[int] = None


@dataclass
class SampleTaskSettings:
    prompts: list[str] = field(default_factory=lambda: [
        "The capital of France is",
        "The chemical symbol of gold is",
    ])
    num_unconditioned: int = 4
    max_tokens: int = 128
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None


@dataclass
class BPBTaskSettings:
    val_data_dir: Optional[str] = None
    max_samples: int = 1000
    max_files: int = 5


@dataclass
class CoreTaskSettings:
    max_per_task: int = -1
    seeds: list[int] = field(default_factory=lambda: [1234])


@dataclass
class CoreExtendedTaskSettings:
    max_per_task: int = -1
    seeds: list[int] = field(default_factory=lambda: [1234])
    strategies: list[str] = field(default_factory=lambda: ["full"])


@dataclass
class KVCacheTaskSettings:
    max_per_task: int = -1
    seeds: list[int] = field(default_factory=lambda: [1234])
    strategies: list[str] = field(default_factory=lambda: ["full", "compress-s1"])


@dataclass
class KVCacheValTaskSettings:
    val_data_dir: Optional[str] = None
    max_samples: int = 1000
    max_files: int = 5
    strategies: list[str] = field(default_factory=lambda: ["full", "compress-s1"])


@dataclass
class TaskSettings:
    lm_eval: LMEvalTaskSettings = field(default_factory=LMEvalTaskSettings)
    sample: SampleTaskSettings = field(default_factory=SampleTaskSettings)
    bpb: BPBTaskSettings = field(default_factory=BPBTaskSettings)
    core: CoreTaskSettings = field(default_factory=CoreTaskSettings)
    core_extended: CoreExtendedTaskSettings = field(default_factory=CoreExtendedTaskSettings)
    kv_cache: KVCacheTaskSettings = field(default_factory=KVCacheTaskSettings)
    kv_cache_val: KVCacheValTaskSettings = field(default_factory=KVCacheValTaskSettings)


@dataclass
class SGLangSettings:
    dp_size: int = 8
    tp_size: int = 1
    host: str = "127.0.0.1"
    port: int = 30000
    mode: Literal["exact", "turbo"] = "exact"
    shim_root: Optional[str] = None
    tokenizer_path: Optional[str] = None
    state_init: str = "zero"
    t_max: Optional[int] = None
    context_length: int = 2055
    server_timeout_s: float = 900
    request_timeout_s: float = 600


@dataclass
class CLISettings:
    out_dir: str = ""
    eval_name: str = "eval"
    eval_tasks: str = "lm_eval"
    step: Optional[int] = None
    checkpoint_path: Optional[str] = None
    hf_path: Optional[str] = None
    hf_repo: Optional[str] = None
    device_type: str = ""
    precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    device_batch_size: int = 32
    eval_backend: Literal["local", "sglang"] = "local"
    tasks: TaskSettings = field(default_factory=TaskSettings)
    sglang: SGLangSettings = field(default_factory=SGLangSettings)
    override_mean_recurrence: Optional[int] = None  # Override recurrence depth for parcae models
    run_name: Optional[str] = field(init=False, default=None)
    model_name: Optional[str] = field(init=False, default=None)
    model_impl: str = field(init=False, default="gpt")
    model_config: Optional[Any] = field(init=False, default=None)
    tokenizer_path: Optional[str] = field(init=False, default=None)
    sequence_length: Optional[int] = field(init=False, default=None)
    device: Optional[torch.device] = field(init=False, default=None)
    ddp: bool = field(init=False, default=False)
    ddp_rank: int = field(init=False, default=0)
    ddp_local_rank: int = field(init=False, default=0)
    ddp_world_size: int = field(init=False, default=1)

    def __post_init__(self):
        if self.hf_path is None and self.out_dir:
            self._load_from_run_dir()
        self.run_name = self.run_name or f"eval-{Path(self.out_dir).name}"
        self.eval_task_list = [t.strip() for t in self.eval_tasks.split(",")]
        valid_tasks = {"lm_eval", "bpb", "sample", "core", "core_extended", "kv_cache", "kv_cache_val"}
        invalid = set(self.eval_task_list) - valid_tasks
        if invalid:
            raise ValueError(f"Invalid eval tasks: {invalid}. Valid: {valid_tasks}")
        self._setup_device()

    def _load_from_run_dir(self):
        run_dir = Path(self.out_dir)
        if not run_dir.exists():
            return
        config_paths = [
            run_dir / "run_config.json",
            run_dir / "model_config.json",
            run_dir / "config.yaml",
            run_dir / "config.json",
        ]
        for cfg_path in config_paths:
            if cfg_path.exists():
                self._load_config(cfg_path)
                break
        if self.checkpoint_path is None and self.hf_repo is None:
            self.checkpoint_path = self._find_checkpoint(run_dir, self.step)

    def _load_config(self, cfg_path: Path):
        if cfg_path.suffix == ".json":
            with open(cfg_path) as f:
                cfg = json.load(f)
        else:
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
        self.model_name = cfg.get("model_name")
        self.tokenizer_path = cfg.get("tokenizer_path")
        model_overwrite = cfg.get("model_overwrite", {})
        if self.model_name:
            from parcae_lm.models.config import Config as DynamicConfig
            self.model_config = DynamicConfig.from_name(self.model_name, **model_overwrite)
            self.sequence_length = self.model_config.block_size
            arch = getattr(self.model_config, 'model_class_name', 'GPT')
            self.model_impl = "parcae" if arch == "Parcae" else "gpt"
        else:
            raw_impl = cfg.get("model_impl", "gpt")
            self.model_impl = "gpt" if raw_impl == "dynamic" else raw_impl

    def _find_checkpoint(self, run_dir: Path, step: Optional[int] = None) -> Optional[str]:
        patterns = [
            f"{run_dir}/checkpoints-*/step-*",
            f"{run_dir}/checkpoints-*/step-*.pt",
            f"{run_dir}/checkpoint*.pt",
            f"{run_dir}/*.pt",
        ]
        all_ckpts = []
        for pattern in patterns:
            for path in glob.glob(pattern):
                if os.path.isfile(path):
                    all_ckpts.append(path)
        if not all_ckpts:
            return None
        def get_step_num(path):
            name = Path(path).name
            if "step-" in name:
                try:
                    return int(name.split("step-")[1].split("-")[0])
                except (ValueError, IndexError):
                    pass
            return 0
        if step is not None:
            for ckpt in all_ckpts:
                if get_step_num(ckpt) == step:
                    return ckpt
        all_ckpts.sort(key=get_step_num, reverse=True)
        return all_ckpts[0] if all_ckpts else None

    def _setup_device(self):
        if self.device_type == "":
            if torch.cuda.is_available():
                self.device_type = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device_type = "mps"
            else:
                self.device_type = "cpu"
        self.ddp = int(os.environ.get("RANK", -1)) != -1
        if self.ddp:
            self.ddp_rank = int(os.environ["RANK"])
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            self.device = torch.device(f"{self.device_type}:{self.ddp_local_rank}")
        else:
            self.ddp_rank = 0
            self.ddp_local_rank = 0
            self.ddp_world_size = 1
            self.device = torch.device(self.device_type)

    @property
    def _is_main_process(self) -> bool:
        return self.ddp_rank == 0

    def _get_autocast_context(self):
        from contextlib import nullcontext
        if self.device_type == "cuda":
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[self.precision]
            return torch.amp.autocast(device_type=self.device_type, dtype=dtype)
        return nullcontext()
