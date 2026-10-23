from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from sglang_parcae.configuration_parcae import ParcaeConfig
from sglang_parcae.planes import plane_count, validate_planes


def resolve_path(model: str) -> Path:
    path = Path(model).expanduser()
    if path.exists():
        return path.resolve()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit(
            f"{model!r} is not an existing path and huggingface_hub is not installed. "
            f"Pass a local checkpoint directory."
        )
    return Path(snapshot_download(repo_id=model,
                                  allow_patterns=["*.json", "*.bin", "*.safetensors"]))


def resolve_run_dir(run_dir: Path, step: int | None = None) -> tuple[dict, Path]:
    config_path = run_dir / "model_config.json"
    if not config_path.exists():
        raise SystemExit(
            f"{run_dir} has no model_config.json -- expected a Parcae training run directory"
        )
    raw = json.loads(config_path.read_text())

    checkpoints = [
        path
        for pattern in ("checkpoints-*/step-*", "checkpoints-*/step-*.pt",
                        "checkpoint*.pt", "*.pt")
        for path in run_dir.glob(pattern)
        if path.is_file()
    ]

    def step_number(path: Path) -> int:
        match = re.search(r"step-(\d+)", path.name)
        return int(match.group(1)) if match else 0

    if step is not None:
        checkpoints = [path for path in checkpoints if step_number(path) == step]
    if not checkpoints:
        suffix = f" at step {step}" if step is not None else ""
        raise SystemExit(f"no checkpoint found under {run_dir}{suffix}")
    checkpoint = max(checkpoints, key=lambda path: (step_number(path), path.stat().st_mtime_ns))
    return raw, checkpoint


def _load_state_dict(ckpt_path: Path) -> dict:
    """torch.load the checkpoint and return the model state dict with prefixes stripped."""
    import torch

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    if not isinstance(sd, dict):
        raise TypeError(f"Unexpected checkpoint format: {type(sd)}")
    cleaned = {}
    for k, v in sd.items():
        for prefix in ("module.", "_orig_mod.", "model."):
            while k.startswith(prefix):
                k = k[len(prefix):]
        cleaned[k] = v
    return cleaned


def export_training_weights(checkpoint: Path, destination: Path) -> Path:
    import torch

    print(f"  extracting model weights from {checkpoint}")
    state = _load_state_dict(checkpoint)
    if not isinstance(state, dict) or not state:
        raise SystemExit(f"{checkpoint} does not contain a model state dict")
    if not all(isinstance(name, str) for name in state):
        raise SystemExit(f"{checkpoint} has invalid model state-dict keys")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(destination)
    return destination


def resolve_tokenizer(model_src: Path, explicit: str | None) -> Path:
    if explicit is not None:
        return resolve_path(explicit)
    if (model_src / "tokenizer.json").exists():
        return model_src
    raise SystemExit(
        f"no tokenizer.json in {model_src}; pass --tokenizer with the tokenizer the checkpoint "
        f"was trained on."
    )


def parcae_scales(raw: dict) -> tuple[float, float, float]:
    parsed = reference_config(raw)
    return (
        float(parsed.init.logit_scale),
        float(parsed.init.embedding_scale),
        float(parsed.init.get_std("embedding")),
    )


def reference_config(raw: dict):
    from parcae_lm.models.config import RoPESettings
    from parcae_lm.models.parcae.config import ParcaeConfig

    cfg = {k: v for k, v in raw.items() if k not in ("_class_name", "init")}
    # Released configs also carry rve_*, gdn_* and recurrence_depth_variance; they are dropped when inert.
    if cfg.get("rve_rank", 1) > 1:
        raise ValueError("rve checkpoints are not supported")
    if "GDN" in str(cfg.get("attention_class_name", "")):
        raise ValueError("GDN checkpoints are not supported")
    if "fixed-var" in str(cfg.get("sampling_scheme", "")):
        raise ValueError("fixed-var sampling schemes are not supported")
    known = {f.name for f in dataclasses.fields(ParcaeConfig)}
    inert = {"rve_rank", "rve_bias", "rve_scale_by_sqrt_rank", "recurrence_depth_variance"}
    for key in list(cfg):
        if key in known:
            continue
        if key in inert or key.startswith("gdn_"):
            cfg.pop(key)
        else:
            raise ValueError(f"unknown config key {key!r}")
    if isinstance(cfg.get("rope_settings"), dict):
        cfg["rope_settings"] = RoPESettings(**cfg["rope_settings"])
    return ParcaeConfig(**cfg)


def ve_gate_layers(raw: dict) -> list[str]:
    from parcae_lm.modules.mixer import has_ve

    cfg = reference_config(raw)
    rec = cfg.recurrent_block_config
    n_p, n_c, n_d = (cfg.n_layers_in_prelude, cfg.n_layers_in_recurrent_block,
                     cfg.n_layers_in_coda)
    coda_base = n_p + n_c * cfg.mean_recurrence
    return (
        [f"prelude.{i}" for i in range(n_p) if has_ve(i, cfg.n_layer)]
        + [f"core.{i}" for i in range(n_c) if has_ve(i + n_p, rec.n_layer)]
        + [f"coda.{i}" for i in range(n_d) if has_ve(i + coda_base, cfg.n_layer)]
    )


def build_config(raw_cfg: dict, strategy: str, t_max: int, logit_scale: float,
                 embedding_scale: float, state_init_std: float,
                 dtype: str = "bfloat16", t_default: int | None = None) -> ParcaeConfig:
    if raw_cfg.get("attention_class_name") == "KDACausalSelfAttention":
        raise NotImplementedError("KDA is not supported by the SGLang backend yet.")

    n_prelude = raw_cfg["n_layers_in_prelude"]
    n_core = raw_cfg["n_layers_in_recurrent_block"]
    n_coda = raw_cfg["n_layers_in_coda"]
    n_head = raw_cfg["num_attention_heads"]
    n_embd = raw_cfg["n_embd"]

    validate_planes(strategy, t_max, n_prelude, n_core, n_coda)
    planes = plane_count(strategy, t_max, n_prelude, n_core, n_coda)

    is_mla = raw_cfg.get("kv_lora_rank") is not None
    architecture = "ParcaeMLAForCausalLM" if is_mla else "ParcaeForCausalLM"

    if is_mla:
        # MLA K/V are cached expanded per head; latent caching is not implemented yet.
        num_key_value_heads = n_head
        head_dim = raw_cfg["qk_nope_head_dim"] + raw_cfg["qk_rope_head_dim"]
        v_head_dim = raw_cfg["v_head_dim"]
    else:
        num_key_value_heads = raw_cfg["num_key_value_heads"]
        head_dim = n_embd // n_head
        v_head_dim = head_dim

    return ParcaeConfig(**{
        # Read by SGLang's ModelConfig.
        "architectures": [architecture],
        "model_type": "parcae",
        "hidden_size": n_embd,
        "intermediate_size": raw_cfg["intermediate_size"],
        "num_attention_heads": n_head,
        "num_key_value_heads": num_key_value_heads,
        "num_hidden_layers": planes,
        "head_dim": head_dim,
        "v_head_dim": v_head_dim,
        "vocab_size": raw_cfg["padded_vocab_size"],
        "max_position_embeddings": raw_cfg["block_size"],
        "tie_word_embeddings": raw_cfg["tie_embeddings"],

        "torch_dtype": dtype,
        "rope_theta": raw_cfg["rope_settings"]["rope_base"],
        "rms_norm_eps": raw_cfg["norm_eps"],

        # Parcae structure read by the model class.
        "n_layers_in_prelude": n_prelude,
        "n_layers_in_recurrent_block": n_core,
        "n_layers_in_coda": n_coda,
        "n_effective_layers": n_prelude + n_core + n_coda,

        "ve_gate_layers": ve_gate_layers(raw_cfg),
        "lookup_strategy": strategy,
        "recurrence_depth_max": t_max,
        "recurrence_depth_default": t_default or raw_cfg["mean_recurrence"],
        "recurrent_embedding_dimension": raw_cfg["recurrent_embedding_dimension"],
        "recurrent_intermediation_embedding_dimension": raw_cfg["recurrent_intermediation_embedding_dimension"],
        "prelude_norm": raw_cfg["prelude_norm"],
        "injection_type": raw_cfg["injection_type"],
        "state_init": raw_cfg["state_init"],
        "qk_norm": raw_cfg["qk_norm"],

        "attention_class_name": raw_cfg.get("attention_class_name", "CausalSelfAttention"),
        "is_mla": is_mla,
        # MLA
        "kv_lora_rank": raw_cfg.get("kv_lora_rank"),
        "q_lora_rank": raw_cfg.get("q_lora_rank"),
        "qk_nope_head_dim": raw_cfg.get("qk_nope_head_dim"),
        "qk_rope_head_dim": raw_cfg.get("qk_rope_head_dim"),
        "nonlin_name": raw_cfg["nonlin_name"],
        "attention_nonlin_name": raw_cfg.get("attention_nonlin_name"),
        "logit_softcap": raw_cfg.get("logit_softcap"),
        # SGLang's LogitsProcessor reads this
        "final_logit_softcapping": raw_cfg.get("logit_softcap"),
        "logit_scale": logit_scale,
        "embedding_scale": embedding_scale,
        "state_init_std": state_init_std,
        "parcae": raw_cfg,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model",
                        help="Parcae checkpoint directory (config.json + weights) or a Hugging Face "
                             "Hub repo id.")
    source.add_argument("--run-dir", type=Path,
                        help="Training run directory containing model_config.json and "
                             "checkpoints-*/step-*; the latest checkpoint is exported automatically.")
    parser.add_argument("--step", type=int,
                        help="With --run-dir, use this checkpoint step instead of the latest.")
    parser.add_argument("--checkpoint-path", type=Path,
                        help="With --run-dir, use this exact training checkpoint.")
    parser.add_argument("--tokenizer", default=None,
                        help="Path or Hub id of the tokenizer the checkpoint was trained on. Defaults "
                             "to the tokenizer in the checkpoint directory; required if there is none.")
    parser.add_argument("--strategy", choices=["full", "compress-s1"], required=True)
    parser.add_argument("--t-max", type=int, default=8,
                        help="Maximum recurrence depth.")
    parser.add_argument("--t-default", type=int, default=None,
                        help="Recurrence depth used when a request does not specify one (default: "
                             "the checkpoint's mean_recurrence). Must be <= --t-max.")
    parser.add_argument("--state-init", default=None,
                        help="Override state_init (e.g. 'zero').")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="torch_dtype written into the shim.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.step is not None and args.checkpoint_path is not None:
        parser.error("use either --step or --checkpoint-path, not both")

    checkpoint = None
    if args.run_dir is not None:
        src = args.run_dir.expanduser().resolve()
        raw, checkpoint = resolve_run_dir(src, args.step)
        if args.checkpoint_path is not None:
            checkpoint = args.checkpoint_path.expanduser().resolve()
            if not checkpoint.is_file():
                parser.error(f"--checkpoint-path does not exist: {checkpoint}")
    else:
        if args.checkpoint_path is not None:
            parser.error("--checkpoint-path requires --run-dir")
        src = resolve_path(args.model)
        if not (src / "config.json").exists():
            raise SystemExit(f"{src} has no config.json -- expected a Parcae checkpoint directory")
        raw = json.loads((src / "config.json").read_text())
    if args.state_init is not None:
        raw = {**raw, "state_init": args.state_init}
    logit_scale, embedding_scale, state_init_std = parcae_scales(raw)
    if args.t_default is not None and not 1 <= args.t_default <= args.t_max:
        parser.error(f"--t-default {args.t_default} outside [1, --t-max {args.t_max}]")
    config = build_config(raw, args.strategy, args.t_max, logit_scale, embedding_scale,
                          state_init_std, args.dtype, args.t_default)

    args.out = args.out.expanduser().resolve()
    if args.out == src.resolve():
        parser.error("--out must differ from the source model or run directory")
    args.out.mkdir(parents=True, exist_ok=True)

    config.save_pretrained(args.out)

    if checkpoint is not None:
        checkpoint = checkpoint.resolve()
        marker_path = args.out / ".parcae_shim_source.json"
        marker = {
            "checkpoint": str(checkpoint),
            "size": checkpoint.stat().st_size,
            "mtime_ns": checkpoint.stat().st_mtime_ns,
        }
        weights = args.out / "pytorch_model.bin"
        cached = False
        if weights.is_file() and marker_path.is_file():
            try:
                cached = json.loads(marker_path.read_text()) == marker
            except (OSError, json.JSONDecodeError):
                pass
        stale = args.out / "model.safetensors"
        if stale.exists() or stale.is_symlink():
            stale.unlink()
        if cached:
            print(f"  reusing extracted weights from {checkpoint}")
        else:
            if weights.exists() or weights.is_symlink():
                weights.unlink()
            weights = export_training_weights(checkpoint, weights)
            marker_path.write_text(json.dumps(marker, indent=2))
    else:
        weights = next(
            (src / n for n in ("model.safetensors", "pytorch_model.bin") if (src / n).exists()),
            None,
        )
        if weights is None:
            raise SystemExit(f"no weights found in {src}")
        for stale_name in ("model.safetensors", "pytorch_model.bin"):
            stale = args.out / stale_name
            if stale.name != weights.name and (stale.exists() or stale.is_symlink()):
                stale.unlink()
        link = args.out / weights.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(weights.resolve())  # Symlink: the weights stay in the --model directory.

    # Symlink the tokenizer files.
    tok_dir = resolve_tokenizer(src, args.tokenizer)
    tok_files = [f for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                             "tokenizer.model", "vocab.json", "merges.txt")
                 if (tok_dir / f).exists()]
    if "tokenizer.json" not in tok_files:
        raise SystemExit(f"{tok_dir} has no tokenizer.json (found {tok_files}) -- cannot build a "
                         f"self-contained model dir")
    for f in tok_files:
        link = args.out / f
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(tok_dir / f)

    if not (args.out / "tokenizer_config.json").exists():
        (args.out / "tokenizer_config.json").write_text(json.dumps({
            "tokenizer_class": "PreTrainedTokenizerFast",
            "model_max_length": int(config.max_position_embeddings),
            "clean_up_tokenization_spaces": False,
        }, indent=2))

    print(f"wrote {args.out}")
    print(f"  architecture   {config.architectures[0]}")
    print(f"  strategy       {args.strategy}  (T_max={args.t_max})")
    print(f"  planes         {config.num_hidden_layers}  "
          f"= {config.n_layers_in_prelude} + "
          f"{'%d*' % args.t_max if args.strategy == 'full' else ''}"
          f"{config.n_layers_in_recurrent_block} + {config.n_layers_in_coda}")
    print(f"  ve tables over {config.n_effective_layers} effective layers")
    print(f"  logit_scale    {logit_scale:.8f}   embedding_scale {embedding_scale:.8f}")
    print(f"  state_init_std {state_init_std:.8f}  (h_0 ~ truncnorm)")
    print(f"  dtype          {args.dtype}")
    print(f"  weights        {weights.name} -> {weights}")
    print(f"  tokenizer      {tok_dir}  ({', '.join(tok_files)})")


if __name__ == "__main__":
    main()
