"""Glue functions to enable/disable monitoring."""

import torch
from torch.utils.flop_counter import FlopCounterMode


import copy
import time


def enable_monitoring_on_step(module, extreme=False):
    if hasattr(module, "monitoring"):
        module.monitoring = True
    if extreme and hasattr(module, "extreme_monitoring"):
        module.extreme_monitoring = True


def disable_monitoring_and_retrieve_metrics(module, metrics):
    if hasattr(module, "monitoring"):
        if len(module.latest_metrics) > 0:
            metrics |= module.latest_metrics
            module.latest_metrics = {}
        module.monitoring = False
    if hasattr(module, "extreme_monitoring"):
        module.extreme_monitoring = False


@torch.compile()
def _reverse_engineer_adam_effective_lr(param, param_state, group):
    """Recompute select values to compute Adam effective lr. Ignoring bias correction because ...
    Ignoring actual lr, because its schedule otherwise dominates the effect. Plot the 'actual lr' as this value multiplied by the schedule
    """
    exp_avg = param_state["exp_avg"].float()
    denom = param_state["exp_avg_sq"].float().sqrt().add_(group["eps"])
    effective_lr = torch.where(
        param.grad.float().abs() > group["eps"],
        exp_avg / denom / param.grad.float(),
        exp_avg / denom / group["eps"],  # needs a more stable impl for small values
    )
    return effective_lr


@torch.no_grad()
def track_gradient_metrics(model, optimizer, metrics):
    optim_metrics = {}
    _, H = model.config.padded_vocab_size, model.config.n_embd
    dim_Q, dim_KV = H, model.config.head_size * model.config.n_query_groups
    # Get specific gradient norms:
    qkv_layer_counter, mlp_layer_counter = 0, 0
    qkv_param_ids = []  # store references to check against optim params in 2nd loop
    proj_param_ids = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            if "qkv" in name and "weight" in name:
                qkv_param_ids += [param]
                if (~torch.isfinite(param.grad)).sum() == 0:
                    if param.grad.numel() % H == 0:  # need to find a better solution for sharded grads
                        q_grad = param.grad.view(-1, H)[:dim_Q, :]
                        optim_metrics[f"query_grad_{qkv_layer_counter}"] = q_grad.norm()
                else:
                    optim_metrics[f"query_grad_{qkv_layer_counter}"] = torch.as_tensor(float("NaN"))
                qkv_layer_counter += 1
            if "mlp" in name and "proj" in name and "weight" in name:
                proj_param_ids += [param]
                if (~torch.isfinite(param.grad)).sum() == 0:
                    optim_metrics[f"ffn2_grad_{mlp_layer_counter}"] = param.grad.norm()
                else:
                    optim_metrics[f"ffn2_grad_{mlp_layer_counter}"] = torch.as_tensor(float("NaN"))
                mlp_layer_counter += 1

    # 2nd moment quality and qkv learning rate
    total_rms = 0.0
    num_params_with_grad = 0
    qkv_layer_counter, mlp_layer_counter = 0, 0
    params_with_finite_grad = []
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None and (~torch.isfinite(param.grad)).sum() == 0:
                params_with_finite_grad.append(param)
                if param in optimizer.state and "exp_avg_sq" in optimizer.state[param]:
                    exp_avg_sq = optimizer.state[param]["exp_avg_sq"]
                    if exp_avg_sq.shape == param.grad.shape:
                        rms = (
                            param.grad.float()
                            .pow(2)
                            .div_(exp_avg_sq.float().clamp_(min=group["eps"] ** 2))
                            .mean()
                            .sqrt()
                        )
                        total_rms += rms
                        num_params_with_grad += 1
                        if param is model.transformer.wte.weight:
                            optim_metrics["embed_RMS"] = rms

                        if any(param is p for p in qkv_param_ids):  # comprehension because "in" breaks
                            qkv_lr = _reverse_engineer_adam_effective_lr(param, optimizer.state[param], group)
                            if qkv_lr.numel() % H == 0:  # need to find a better solution for sharded grads
                                optim_metrics[f"q_effective_lr_{qkv_layer_counter}"] = qkv_lr.view(-1, H)[
                                    :dim_Q, :
                                ].mean()
                                optim_metrics[f"k_effective_lr_{qkv_layer_counter}"] = qkv_lr.view(-1, H)[
                                    dim_Q : dim_Q + dim_KV, :
                                ].mean()
                                optim_metrics[f"v_effective_lr_{qkv_layer_counter}"] = qkv_lr.view(-1, H)[
                                    dim_Q + dim_KV :, :
                                ].mean()
                            qkv_layer_counter += 1

                        if any(param is p for p in proj_param_ids):
                            proj_lr = _reverse_engineer_adam_effective_lr(param, optimizer.state[param], group)
                            optim_metrics[f"ffn2_effective_lr_{mlp_layer_counter}"] = proj_lr.mean()
                            mlp_layer_counter += 1

    if num_params_with_grad > 0:
        optim_metrics["avg_RMS"] = total_rms / num_params_with_grad

    # Finally, a quick local L1 norm
    if len(params_with_finite_grad) > 0:
        l1_grad_norm = torch.mean(torch.stack([torch.norm(p.grad.detach(), 1.0) for p in params_with_finite_grad]))
        optim_metrics["local_l1_grad_norm"] = l1_grad_norm

    # And param norms
    l2_param_norm = torch.norm(torch.stack([torch.norm(p.detach()) for p in model.parameters()]))
    l1_param_norm = torch.mean(torch.stack([torch.norm(p.detach(), 1.0) for p in model.parameters()]))

    optim_metrics["l2_param_norm"] = l2_param_norm
    optim_metrics["l1_param_norm"] = l1_param_norm

    # sub-group param norms
    block_module = model.transformer.core_block if hasattr(model.transformer, 'core_block') else model.transformer.h
    optim_metrics["core_block_l2_param_norm"] = torch.norm(
        torch.stack([torch.norm(p.detach()) for p in block_module.parameters()])
    )
    optim_metrics["word_embed_l2_param_norm"] = torch.norm(
        torch.stack([torch.norm(p.detach()) for p in model.transformer.wte.parameters()])
    )
    optim_metrics["model_l2_param_norm"] = torch.norm(
        torch.stack([torch.norm(p.detach()) for n, p in model.named_parameters() if "wte" not in name])
    )

    # finalize if all metrics were recorded successfully
    metrics |= optim_metrics


# Bucket parameter grad norms by submodule for grad-spike attribution (block / attn|mlp|norm / param).
def _spike_bucket_key(parts: list[str]) -> str:
    for i, p in enumerate(parts):
        if p in ("core_block", "prelude", "coda") and i + 1 < len(parts):
            block = f"{p}.{parts[i + 1]}"
            rest = parts[i + 2:]
            if len(rest) >= 2 and rest[0] == "attn":
                return f"{block}.attn.{rest[1]}"
            if len(rest) >= 2 and rest[0] == "mlp":
                return f"{block}.mlp.{rest[1]}"
            if rest and rest[0].startswith("norm"):
                return f"{block}.{rest[0]}"
            return block
        if p == "adapter":
            return f"adapter.{parts[i + 1]}" if i + 1 < len(parts) else "adapter"
        if p == "C":
            return "C"
        if p in ("wte", "lm_head", "embedding"):
            return "embedding"
        if p == "ln_f":
            return "norm"
    return "other"


@torch.no_grad()
def compute_grad_norm_buckets(model):
    buckets: dict[str, list] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        clean = name
        for prefix in ("_orig_mod.", "_forward_module.", "_fsdp_wrapped_module.", "module."):
            clean = clean.replace(prefix, "")
        key = _spike_bucket_key(clean.split("."))
        buckets.setdefault(key, []).append(param.grad.detach().norm())
    bucket_norms = [(k, torch.stack(v).norm()) for k, v in buckets.items()]
    total_sq = sum(n.item() ** 2 for _, n in bucket_norms)
    if total_sq > 0:
        bucket_norms = [(k, n, n.item() / total_sq**0.5) for k, n in bucket_norms]
        return bucket_norms
    return [(k, n, 0.0) for k, n in bucket_norms]


def _get_num_params(model: torch.nn.Module, only_trainable: bool = False) -> int:
    """
    Get the total model params
    Args : only_trainable: whether to only count trainable params
    """
    param_list = list(model.parameters())
    if only_trainable:
        param_list = [p for p in param_list if p.requires_grad]
    # unique_params = {p.data_ptr(): p for p in param_list}.values()
    return sum(p.numel() for p in param_list)


def _estimate_num_flop_per_token(num_params: int, model_config, model=None) -> int:
    t = model_config.block_size
    
    # Check if this is a recurrent model with proper layer counting
    if hasattr(model_config, 'mean_recurrence') and hasattr(model_config, 'n_layers_in_recurrent_block'):
        mean_recurrence = model_config.mean_recurrence
        prelude_layers = getattr(model_config, 'n_layers_in_prelude', 1)
        coda_layers = getattr(model_config, 'n_layers_in_coda', 1)
        recurrent_layers = model_config.n_layers_in_recurrent_block
        
        # Calculate static vs recurrent parameters
        # Static params: embedding, prelude, coda, head
        # Recurrent params: core_block + adapter (used mean_recurrence times)
        if model is not None:
            rec_params = sum([p.numel() for p in model.transformer.core_block.parameters()])
            if hasattr(model.transformer, 'adapter') and not isinstance(model.transformer.adapter, torch.nn.Identity):
                rec_params += sum([p.numel() for p in model.transformer.adapter.parameters()])
            static_params = num_params - rec_params
        else:
            # Rough estimate: recurrent block is roughly (n_layers_in_recurrent_block / total_layers) of params
            total_layers = prelude_layers + coda_layers + recurrent_layers
            rec_params = int(num_params * recurrent_layers / total_layers)
            static_params = num_params - rec_params
        
        # Parameter FLOPs: static params used once, recurrent params used mean_recurrence times
        param_flops = 6 * static_params + 6 * rec_params * mean_recurrence
        
        # Attention FLOPs (quadratic in sequence length)
        h_main = getattr(model_config, "n_heads", model_config.num_attention_heads)
        q_main = model_config.head_size
        static_attn_flops = 12 * (prelude_layers + coda_layers) * h_main * q_main * t
        
        # Recurrent block attention FLOPs
        if hasattr(model_config, 'recurrent_block_config'):
            rec_config = model_config.recurrent_block_config
            h_rec = getattr(rec_config, "n_heads", rec_config.num_attention_heads)
            q_rec = rec_config.head_size
        else:
            h_rec = h_main
            q_rec = q_main
        recurrent_attn_flops = 12 * (recurrent_layers * mean_recurrence) * h_rec * q_rec * t
        
        flop_per_token = param_flops + static_attn_flops + recurrent_attn_flops
    else:
        # Standard transformer estimation
        l, h, q = (
            model_config.n_layer,
            getattr(model_config, "n_heads", model_config.num_attention_heads),
            model_config.head_size,
        )
        flop_per_token = 6 * num_params + 12 * l * h * q * t

    # `6 * params` and `12 * l * h * q * t` already count forward and backward.
    return flop_per_token


def _actually_measure_flops(model_config, objective, gradient_checkpointing, micro_batch_size=1) -> int:
    """Measure FLOP usage for a single sequence."""
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode
        
        config_copy = copy.deepcopy(model_config)
        # Annoying special rules for improper triton implementations
        config_copy.simple_ops = True
        config_copy.use_fused_head = False
        # Also update recurrent_block_config if it exists
        if hasattr(config_copy, '_recurrent_block_config') and config_copy._recurrent_block_config is not None:
            config_copy._recurrent_block_config.simple_ops = True
            config_copy._recurrent_block_config.use_fused_head = False
        
        # Use FakeTensorMode for proper FLOP counting (meta tensors don't support all ops).
        # Build the input on meta, the model's device, so FakeTensorMode does not fall back to the estimator.
        with FakeTensorMode(allow_non_fake_inputs=True):
            with torch.device("meta"):
                meta_model = config_copy.construct_model(objective=objective, gradient_checkpointing=gradient_checkpointing)
                x = torch.randint(0, config_copy.padded_vocab_size, (micro_batch_size, model_config.block_size))

            flop_counter = FlopCounterMode(display=not torch.distributed.is_initialized())
            with flop_counter:
                meta_model(input_ids=x, labels=x)["loss"].backward()
            measured_flops = flop_counter.get_total_flops()
            del meta_model, x
    except (NotImplementedError, AssertionError, RuntimeError) as e:
        _rank0 = (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0
        if _rank0:
            print(
                "Cannot trace model with meta tensors for flop calculation, falling back on estimated flop count. "
                f"This may be (very) inaccurate for exotic archs. Original error: {e}"
            )
        with torch.device("meta"):
            meta_model = model_config.construct_model(
                objective=objective, gradient_checkpointing=gradient_checkpointing
            )
        num_params = sum(p.numel() for p in meta_model.parameters())
        # _estimate_num_flop_per_token returns per-token FLOPs, but caller expects total sequence FLOPs
        # (it will divide by block_size), so multiply here to match the traced path
        flops_per_token = _estimate_num_flop_per_token(num_params, model_config, model=meta_model)
        measured_flops = flops_per_token * model_config.block_size
        if _rank0:
            print(f"FLOP estimate for this model is {measured_flops:,} ({flops_per_token:,} per token)")
    return measured_flops


def _get_peak_flops(fabric_precision, device_name: str) -> int:
    """Assuming tensor core usage for all nvidia cards"""
    if "32" in fabric_precision:
        multiplier = 0.5
    elif "64" in fabric_precision:
        multiplier = 0.25
    elif "8" in fabric_precision:
        multiplier = 2
    else:
        multiplier = 1

    if "MI250" in device_name:
        # https://www.amd.com/en/products/accelerators/instinct/mi200/mi250x.html
        flops = 383e12 / 2 * multiplier  # only one die counted

    elif "A100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/a100/
        flops = 312e12 * multiplier
    elif "H100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h100/
        # NOTE: Specifications are one-half lower without sparsity.
        if "NVL" in device_name:
            flops = 1979e12 * multiplier
        elif "PCIe" in device_name:
            flops = 756e12 * multiplier
        else:  # for SXM and other variants
            flops = 989e12 * multiplier
    elif "B200" in device_name or "B300" in device_name:
        # NVIDIA Blackwell B200/B300
        # NOTE: Specifications are one-half lower without sparsity.
        flops = 2250e12 * multiplier
    elif "V100" in device_name:
        flops = 125e12 * multiplier  # sxm
    elif "RTX 6000 Ada" in device_name:
        flops = 210.6e12 * multiplier  # or 364.25 = 1457.0 / 2 / 2? thanks nvidia
    elif "A4000" in device_name:
        # 4000 performance is actually less clear
        # from this whitepaper, and estimated to be 88.45 TFLOP/s, based on it containing 192 tensor cores,
        # compared to 336 for the A6000.
        flops = 88.45e12 * multiplier
    elif "A5000" in device_name:
        flops = 111.1e12 * multiplier
    elif "A6000" in device_name:
        flops = 154.8e12 * multiplier
    elif "2080 Ti" in device_name:
        flops = 53.8e12 * multiplier
    elif "RTX 3050 Ti Laptop" in device_name:
        flops = 21.2e12  # 5.299 / 9.098 * 36.4 # :)
    else:  # for other GPU types, raise
        raise ValueError(f"Could not retrieve flops for device {device_name}.")
    return int(flops)  # ok up to 1e18


# Cache dictionary
model_cache = {}


def get_MFU_metrics(tokens_per_second, fabric, model, precision, measure_flops=True):
    if id(model) not in model_cache:
        model_param_count = _get_num_params(model)
        if measure_flops:
            measured_flops = _actually_measure_flops(model.config, model.objective, model.gradient_checkpointing)
            num_flop_per_token = measured_flops / model.config.block_size
        else:
            num_flop_per_token = _estimate_num_flop_per_token(model_param_count, model.config, model=model)

        peak_flops = _get_peak_flops(precision, torch.cuda.get_device_name(device=fabric.device))
        model_cache[id(model)] = {
            "num_params": model_param_count,
            "num_flop_per_token": num_flop_per_token,
            "peak_flops": peak_flops,
        }
    cache = model_cache[id(model)]
    flops = cache["num_flop_per_token"] * tokens_per_second
    mfu = flops / fabric.world_size / cache["peak_flops"]
    return cache["num_flop_per_token"], flops / 1e12, mfu


def standalone_measure_peak_flops(A100_PEAK_TFLOPS_FP16=312, dtype=torch.float16):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True  # Should be true anyway

    bs = 16

    for max_size in [2048, 4096, 8192, 16384]:
        # Determine the largest matrix size that fits in memory
        # while True:
        #     try:
        #         torch.cuda.empty_cache()
        #         a = torch.randn(max_size, max_size, dtype=torch.float16, device="cuda")
        #         b = torch.randn(max_size, max_size, dtype=torch.float16, device="cuda")
        #         del a, b
        #         max_size *= 2
        #     except RuntimeError:
        #         max_size //= 2
        #         break

        # Create matrices
        a = torch.randn(bs, max_size, max_size, dtype=dtype, device="cuda")
        b = torch.randn(bs, max_size, max_size, dtype=dtype, device="cuda")

        # Warm-up run
        torch.bmm(a, b)
        torch.cuda.synchronize()

        # Measure time for matrix multiplication
        print("Starting new measurement cycle")
        num_runs = 100
        cooldown_time = 0.1

        times = []
        for _ in range(num_runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()  # type: ignore
            torch.bmm(a, b)
            end.record()  # type: ignore

            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) / 1000)  # Convert to seconds

            # Cooldown step
            torch.cuda.empty_cache()  # Clear CUDA cache
            time.sleep(cooldown_time)  # Wait for cooldown_time seconds

        mean_time, std_time = torch.mean(torch.as_tensor(times)), torch.std(torch.as_tensor(times))

        # Calculate FLOPs
        flops = 2 * max_size**3 * bs / mean_time  # 2n^3 FLOPs for matrix multiplication
        tflops = flops / 1e12

        # Calculate MFU
        mfu = (tflops / A100_PEAK_TFLOPS_FP16) * 100

        print(f"Matrix size: {bs}x{max_size}x{max_size}")
        print(f"Time per multiplication: {mean_time:.6f} seconds, std={std_time:.6f}")
        print(f"Measured peak performance: {tflops:.2f} TFLOPS ({dtype})")
        print(f"A100 theoretical peak in float16: {A100_PEAK_TFLOPS_FP16:.2f} TFLOPS ({dtype})")
        print(f"Model FLOPs Utilization (MFU): {mfu:.2f}%")


if __name__ == "__main__":
    standalone_measure_peak_flops()
