import os
import sys
import time
import json
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from jsonargparse import CLI
from receval.settings import CLISettings
from receval.tasks import run_lm_eval
from receval.tasks.val_loss import run_val_loss, load_val_texts_from_parquet
from receval.tasks.core_eval import run_core_eval
from receval.tasks.core_extended_eval import run_core_extended_eval
from receval.tasks.kv_cache_eval import run_kv_cache_core_eval, KVCacheModelWrapper



def print0(*args, **kwargs):
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs, flush=True)


def setup_distributed(settings: CLISettings):
    if settings.ddp:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(settings.ddp_local_rank)
    if settings.device_type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        try:
            torch.backends.cuda.enable_cudnn_sdp(False)
        except AttributeError:
            pass  # Not all PyTorch versions have this


def cleanup_distributed(settings: CLISettings):
    if settings.ddp:
        torch.distributed.destroy_process_group()


def load_model(settings: CLISettings):
    from parcae_lm.tokenizer import Tokenizer

    # HuggingFace transformers model (e.g. GPT-2)
    if settings.hf_path is not None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print0(f"Loading HuggingFace model: {settings.hf_path}")
        model = AutoModelForCausalLM.from_pretrained(settings.hf_path)
        if settings.device is not None:
            model.to(settings.device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(settings.hf_path)
        return model, tokenizer

    # Parcae/GPT model from HuggingFace repo (e.g. SandyResearch/parcae-140m)
    if settings.hf_repo is not None:
        from receval.models.parcae import ModelingParcae
        print0(f"Loading from HuggingFace: {settings.hf_repo}")
        model = ModelingParcae.from_pretrained(settings.hf_repo, device=settings.device)
        if settings.checkpoint_path is not None:
            state = torch.load(
                settings.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            state_dict = state.get("model", state) if isinstance(state, dict) else state
            state_dict = {
                key.replace("._orig_mod.", "."): value
                for key, value in state_dict.items()
            }
            incompatible = model.load_state_dict(state_dict, strict=False)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(
                    f"checkpoint mismatch: missing={len(incompatible.missing_keys)} "
                    f"unexpected={len(incompatible.unexpected_keys)}"
                )
        model.eval()
        if settings.override_mean_recurrence is not None:
            print0(f"Overriding mean_recurrence: {model.config.mean_recurrence} -> {settings.override_mean_recurrence}")
            model.config.mean_recurrence = settings.override_mean_recurrence
        settings.model_impl = "parcae"
        settings.sequence_length = model.config.block_size
        if settings.tokenizer_path:
            tokenizer = Tokenizer(settings.tokenizer_path)
        else:
            tokenizer = Tokenizer(
                os.environ.get(
                    "PARCAE_TOKENIZER",
                    "SandyResearch/parcae-tokenizer",
                )
            )
        return model, tokenizer

    # Local checkpoint
    if settings.model_impl == "gpt":
        from receval.models.gpt import ModelingGPT
        ModelClass = ModelingGPT
    elif settings.model_impl == "parcae":
        from receval.models.parcae import ModelingParcae
        ModelClass = ModelingParcae
    else:
        raise ValueError(f"Unknown model_impl: {settings.model_impl}")

    print0(f"Loading {settings.model_impl} model: {settings.model_name}")
    assert settings.model_config is not None, "model_config required"

    if settings.override_mean_recurrence is not None and hasattr(settings.model_config, 'mean_recurrence'):
        original_recurrence = settings.model_config.mean_recurrence
        settings.model_config.mean_recurrence = settings.override_mean_recurrence
        print0(f"Overriding mean_recurrence: {original_recurrence} -> {settings.override_mean_recurrence}")

    model = ModelClass(settings.model_config)

    if settings.checkpoint_path:
        print0(f"Loading checkpoint: {settings.checkpoint_path}")
        state = torch.load(settings.checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = state["model"] if "model" in state else state
        state_dict = {
            key.replace("._orig_mod.", "."): value
            for key, value in state_dict.items()
        }
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"checkpoint mismatch: missing={len(incompatible.missing_keys)} "
                f"unexpected={len(incompatible.unexpected_keys)}"
            )

    model.to(settings.device)
    model.eval()

    if settings.tokenizer_path:
        tokenizer = Tokenizer(settings.tokenizer_path)
    else:
        raise ValueError("tokenizer_path not found in run config")

    return model, tokenizer


def run_sample_task(model, tokenizer, settings: CLISettings):
    if not settings._is_main_process:
        return {}

    task_settings = settings.tasks.sample
    results = {"conditioned": [], "unconditioned": []}

    print0("\n" + "=" * 60)
    print0("Sample Generation")
    print0("=" * 60)

    for prompt in task_settings.prompts:
        tokens = tokenizer.encode(prompt, return_tensors=False)
        input_ids = torch.tensor([tokens], device=settings.device)
        with settings._get_autocast_context():
            output = model.generate(
                input_ids,
                max_new_tokens=task_settings.max_tokens,
                temperature=task_settings.temperature,
                top_k=task_settings.top_k,
                top_p=task_settings.top_p,
                do_sample=task_settings.temperature > 0,
            )
        text = tokenizer.decode(output[0].tolist())
        print0(f"\n{text}")
        results["conditioned"].append(text)

    if task_settings.num_unconditioned > 0:
        print0("\nUnconditioned samples:")
        bos = getattr(tokenizer, "bos_id", 1)
        input_ids = torch.tensor([[bos]], device=settings.device)
        for _ in range(task_settings.num_unconditioned):
            with settings._get_autocast_context():
                output = model.generate(
                    input_ids,
                    max_new_tokens=task_settings.max_tokens,
                    temperature=task_settings.temperature,
                    do_sample=True,
                )
            text = tokenizer.decode(output[0].tolist())
            print0(f"\n{text}")
            results["unconditioned"].append(text)

    return results


def main():
    settings: CLISettings = CLI(CLISettings)
    if settings.eval_backend == "sglang":
        if settings.ddp_rank != 0:
            return
        if settings.hf_path or settings.hf_repo:
            raise ValueError("SGLang evaluation requires a local training run")
        if settings.precision != "bf16":
            raise ValueError("SGLang evaluation currently requires precision=bf16")
        unsupported = set(settings.eval_task_list) - {
            "core", "core_extended", "kv_cache"
        }
        if unsupported:
            raise ValueError(
                f"SGLang does not support these tasks yet: {sorted(unsupported)}"
            )
    else:
        setup_distributed(settings)

    print0("=" * 60)
    print0("Evaluation")
    print0("=" * 60)
    print0(f"Run dir: {settings.out_dir}")
    print0(f"Model: {settings.model_name}")
    print0(f"Checkpoint: {settings.checkpoint_path}")
    print0(f"Tasks: {settings.eval_tasks}")
    print0(f"Device: {settings.device}")
    if settings.override_mean_recurrence is not None:
        print0(f"Override recurrence: {settings.override_mean_recurrence}")
    print0("=" * 60)

    if settings.checkpoint_path is None and settings.hf_path is None and settings.hf_repo is None:
        print0("\nERROR: No checkpoint found!")
        print0("Check that out_dir contains checkpoints or pass --checkpoint_path")
        cleanup_distributed(settings)
        return

    if settings.eval_backend == "sglang":
        from parcae_lm.tokenizer import Tokenizer
        from receval.models.sglang import managed_sglang_model, tokenizer_path

        tokenizer = Tokenizer(tokenizer_path(settings))
        settings.device = torch.device("cpu")
        model = None
    else:
        model, tokenizer = load_model(settings)
    all_results = {}

    for task in settings.eval_task_list:
        print0(f"\nRunning task: {task}")
        t0 = time.time()

        if task == "lm_eval":
            results = run_lm_eval(model, tokenizer, settings)
            all_results["lm_eval"] = results.results
            if settings._is_main_process:
                print0(f"\nLM-Eval Results:")
                for task_name, task_result in results.task_scores.items():
                    print0(f"  {task_name}:")
                    if isinstance(task_result, dict):
                        for metric, value in task_result.items():
                            if isinstance(value, (int, float)):
                                print0(f"    {metric}: {value:.4f}")
                    else:
                        print0(f"    {task_result}")

        elif task == "sample":
            results = run_sample_task(model, tokenizer, settings)
            all_results["sample"] = results

        elif task == "bpb":
            val_data_dir = settings.tasks.bpb.val_data_dir
            if not val_data_dir:
                print0("WARNING: No val_data_dir specified for BPB task, skipping")
                all_results["bpb"] = {}
            else:
                print0(f"Loading validation data from: {val_data_dir}")
                val_texts = load_val_texts_from_parquet(
                    val_data_dir,
                    max_files=settings.tasks.bpb.max_files
                )
                print0(f"Loaded {len(val_texts)} validation texts")
                results = run_val_loss(
                    model, tokenizer, settings, val_texts,
                    max_samples=settings.tasks.bpb.max_samples
                )
                all_results["bpb"] = {
                    "loss": results.loss,
                    "perplexity": results.perplexity,
                    "bits_per_byte": results.bits_per_byte,
                    "num_tokens": results.num_tokens,
                    "num_bytes": results.num_bytes,
                }
                if settings._is_main_process:
                    print0(f"\nValidation Results:")
                    print0(f"  Loss: {results.loss:.4f}")
                    print0(f"  Perplexity: {results.perplexity:.2f}")
                    print0(f"  Bits per byte: {results.bits_per_byte:.4f}")
                    print0(f"  Tokens evaluated: {results.num_tokens:,}")
                    print0(f"  Bytes evaluated: {results.num_bytes:,}")

        elif task == "core":
            max_seq_len = settings.sequence_length or 2048
            seeds = settings.tasks.core.seeds
            if settings.eval_backend == "sglang":
                with managed_sglang_model(settings, "full") as server_model:
                    results = run_core_eval(
                        server_model, tokenizer, torch.device("cpu"),
                        max_seq_len=max_seq_len,
                        max_per_task=settings.tasks.core.max_per_task,
                        seeds=seeds,
                    )
            else:
                with settings._get_autocast_context():
                    results = run_core_eval(
                        model, tokenizer, settings.device,
                        max_seq_len=max_seq_len,
                        max_per_task=settings.tasks.core.max_per_task,
                        seeds=seeds
                    )
            all_results["core"] = results
            if settings._is_main_process:
                print0(f"\nCORE Results:")
                # Check if we have aggregated results (multiple seeds)
                if 'aggregated' in results:
                    for task_name in results["results"].keys():
                        acc = results["aggregated"]["results"][task_name]
                        acc_std = results["aggregated"]["results_std"][task_name]
                        centered = results["aggregated"]["centered_results"][task_name]
                        centered_std = results["aggregated"]["centered_results_std"][task_name]
                        print0(f"  {task_name}: acc={acc:.4f}±{acc_std:.4f} centered={centered:.4f}±{centered_std:.4f}")
                    print0(f"\n  CORE metric: {results['core_metric']:.4f} ± {results['aggregated']['core_metric_std']:.4f}")
                else:
                    for task_name, acc in results["results"].items():
                        centered = results["centered_results"][task_name]
                        print0(f"  {task_name}: acc={acc:.4f} centered={centered:.4f}")
                    print0(f"\n  CORE metric: {results['core_metric']:.4f}")

        elif task == "core_extended":
            max_seq_len = settings.sequence_length or 2048
            seeds = settings.tasks.core_extended.seeds
            strategies = settings.tasks.core_extended.strategies
            if settings.eval_backend == "sglang":
                results = {}
                for strategy in strategies:
                    with managed_sglang_model(settings, strategy) as server_model:
                        results[strategy] = run_core_extended_eval(
                            server_model, tokenizer, settings.device,
                            max_seq_len=max_seq_len,
                            max_per_task=settings.tasks.core_extended.max_per_task,
                            seeds=seeds,
                        )
            elif strategies != ["full"]:
                results = {}
                for strategy in strategies:
                    eval_model = (
                        model
                        if strategy == "full"
                        else KVCacheModelWrapper(
                            model,
                            lookup_strategy=strategy,
                            num_steps=settings.override_mean_recurrence,
                            mode="prefix_decode",
                        )
                    )
                    with settings._get_autocast_context():
                        results[strategy] = run_core_extended_eval(
                            eval_model, tokenizer, settings.device,
                            max_seq_len=max_seq_len,
                            max_per_task=settings.tasks.core_extended.max_per_task,
                            seeds=seeds,
                        )
            else:
                with settings._get_autocast_context():
                    results = run_core_extended_eval(
                        model, tokenizer, settings.device,
                        max_seq_len=max_seq_len,
                        max_per_task=settings.tasks.core_extended.max_per_task,
                        seeds=seeds
                    )
            all_results["core_extended"] = results
            if settings._is_main_process:
                print0(f"\nCORE Extended Results:")
                if settings.eval_backend == "sglang" or strategies != ["full"]:
                    for strategy, strategy_results in results.items():
                        print0(
                            f"  {strategy}: CORE={strategy_results['core_metric']:.4f} "
                            f"CORE-Extended={strategy_results['core_extended_metric']:.4f}"
                        )
                elif 'aggregated' in results:
                    for task_name in results["results"].keys():
                        acc = results["aggregated"]["results"][task_name]
                        acc_std = results["aggregated"]["results_std"][task_name]
                        centered = results["aggregated"]["centered_results"][task_name]
                        centered_std = results["aggregated"]["centered_results_std"][task_name]
                        print0(f"  {task_name}: acc={acc:.4f}±{acc_std:.4f} centered={centered:.4f}±{centered_std:.4f}")
                    print0(f"\n  CORE metric: {results['core_metric']:.4f} ± {results['aggregated']['core_metric_std']:.4f}")
                    print0(f"  CORE Extended metric: {results['core_extended_metric']:.4f} ± {results['aggregated']['core_extended_metric_std']:.4f}")
                else:
                    for task_name, acc in results["results"].items():
                        centered = results["centered_results"][task_name]
                        print0(f"  {task_name}: acc={acc:.4f} centered={centered:.4f}")
                    print0(f"\n  CORE metric: {results['core_metric']:.4f}")
                    print0(f"  CORE Extended metric: {results['core_extended_metric']:.4f}")

        elif task == "kv_cache":
            if settings.model_impl != "parcae":
                print0("WARNING: kv_cache task only supports Parcae models, skipping")
                all_results["kv_cache"] = {}
            elif settings.eval_backend == "sglang":
                results = {}
                for strategy in settings.tasks.kv_cache.strategies:
                    with managed_sglang_model(settings, strategy) as server_model:
                        results[strategy] = run_core_eval(
                            server_model,
                            tokenizer,
                            torch.device("cpu"),
                            max_seq_len=settings.sequence_length or 2048,
                            max_per_task=settings.tasks.kv_cache.max_per_task,
                            seeds=settings.tasks.kv_cache.seeds,
                        )
                all_results["kv_cache"] = results
            else:
                max_seq_len = settings.sequence_length or 2048
                with settings._get_autocast_context():
                    results = run_kv_cache_core_eval(
                        model, tokenizer, settings.device,
                        max_seq_len=max_seq_len,
                        max_per_task=settings.tasks.kv_cache.max_per_task,
                        seeds=settings.tasks.kv_cache.seeds,
                        strategies=settings.tasks.kv_cache.strategies,
                        num_steps=settings.override_mean_recurrence,
                    )
                all_results["kv_cache"] = results
                if settings._is_main_process:
                    print0(f"\nKV-Cache CORE Results:")
                    for strategy, r in results.items():
                        print0(f"  {strategy}: CORE={r['core_metric']:.4f}")

        elif task == "kv_cache_val":
            val_data_dir = settings.tasks.kv_cache_val.val_data_dir
            if not val_data_dir:
                print0("WARNING: No val_data_dir specified for kv_cache_val task, skipping")
                all_results["kv_cache_val"] = {}
            elif settings.model_impl != "parcae":
                print0("WARNING: kv_cache_val task only supports Parcae models, skipping")
                all_results["kv_cache_val"] = {}
            else:
                print0(f"Loading validation data from: {val_data_dir}")
                val_texts = load_val_texts_from_parquet(
                    val_data_dir, max_files=settings.tasks.kv_cache_val.max_files,
                )
                print0(f"Loaded {len(val_texts)} validation texts")
                kv_results = {}
                partial_path = None
                if settings.out_dir:
                    partial_path = (
                        Path(settings.out_dir)
                        / "eval"
                        / f"{settings.eval_name}.json"
                    )
                    if partial_path.exists():
                        try:
                            prior = json.loads(partial_path.read_text())
                            kv_results.update(
                                prior.get("kv_cache_val", {})
                            )
                        except (OSError, json.JSONDecodeError):
                            pass
                for strategy in settings.tasks.kv_cache_val.strategies:
                    if strategy in kv_results:
                        print0(
                            f"\nKV-Cache val loss: strategy={strategy} "
                            "(already complete)"
                        )
                        continue
                    print0(f"\nKV-Cache val loss: strategy={strategy}")
                    if strategy == "full":
                        # full + token-by-token is numerically equivalent to a single parallel
                        # forward pass; skip the wrapper to avoid the N-fold slowdown.
                        eval_module = model
                    else:
                        eval_module = KVCacheModelWrapper(
                            model, lookup_strategy=strategy,
                            num_steps=settings.override_mean_recurrence, mode="token_by_token",
                        )
                    r = run_val_loss(
                        eval_module, tokenizer, settings, val_texts,
                        max_samples=settings.tasks.kv_cache_val.max_samples,
                    )
                    kv_results[strategy] = {
                        "loss": r.loss, "perplexity": r.perplexity,
                        "bits_per_byte": r.bits_per_byte, "num_tokens": r.num_tokens,
                    }
                    if settings._is_main_process:
                        print0(f"  {strategy}: loss={r.loss:.4f}  ppl={r.perplexity:.2f}  bpb={r.bits_per_byte:.4f}")
                    if settings._is_main_process and partial_path is not None:
                        partial_path.parent.mkdir(
                            parents=True,
                            exist_ok=True,
                        )
                        tmp_path = partial_path.with_suffix(
                            ".json.tmp"
                        )
                        tmp_path.write_text(
                            json.dumps(
                                {
                                    **all_results,
                                    "kv_cache_val": kv_results,
                                },
                                indent=2,
                                default=str,
                            )
                        )
                        tmp_path.replace(partial_path)
                all_results["kv_cache_val"] = kv_results

        print0(f"Task {task} completed in {time.time() - t0:.2f}s")

    if settings._is_main_process and settings.out_dir:
        eval_dir = Path(settings.out_dir) / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        output_path = eval_dir / f"{settings.eval_name}.json"
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print0(f"\nResults saved to: {output_path}")

    if settings.eval_backend != "sglang":
        cleanup_distributed(settings)
    print0("\nEvaluation complete.")


if __name__ == "__main__":
    main()
