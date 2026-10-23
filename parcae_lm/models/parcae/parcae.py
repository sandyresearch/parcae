import math
from contextlib import contextmanager
from typing import Any, Optional, TYPE_CHECKING

import torch
from torch import Tensor
import torch.nn as nn

if TYPE_CHECKING:
    from parcae_lm.utils.cache import ParcaeDynamicCache

from parcae_lm.models.parcae.config import ParcaeConfig
from parcae_lm.ops import LinearCrossEntropyLoss
from parcae_lm.modules.utils import precompute_freqs_cis
from parcae_lm.modules.injection import _get_injection_method
from parcae_lm.modules.mixer import has_ve


# Liger's Triton kernel has signatures that torch.compile's dynamo can't trace through,
# plus an internal `.item()` that forces a graph break anyway. Wrap the call with
# `torch._dynamo.disable` so dynamo treats it as opaque: compile the trunk up to here,
# hand off to eager, resume compilation after. Without this, autotune crashes with
# illegal-memory-access during inductor's coord_desc tuning.
@torch._dynamo.disable(recursive=True)
def _call_liger_eager(loss_fn, weight, x, labels):
    return loss_fn(weight, x, labels)


class Parcae(torch.nn.Module):
    _default_objective = {"ignore_index": -100, "z_regularization": 0.0}

    def __init__(
        self,
        config: ParcaeConfig,
        objective=None,
        gradient_checkpointing=False,
    ) -> None:
        super().__init__()
        objective = objective or self._default_objective
        assert config.padded_vocab_size is not None
        self.config = config

        # Dynamical Systems Components
        adapter = _get_injection_method(config)
        C = config.Linear(
            config.recurrent_embedding_dimension,
            config.n_embd,
            bias=config.bias,
            init_method=config.init.fn("C", config.n_layers_in_prelude),
        )
        C.weight._no_weight_decay = True

        recurrent_config = config.recurrent_block_config

        # Transformer layers
        prelude = torch.nn.ModuleList(config.Block(config, layer_id=i) for i in range(config.n_layers_in_prelude))
        
        core_block = torch.nn.ModuleList(
            recurrent_config.Block(recurrent_config, layer_id=i + config.n_layers_in_prelude)
            for i in range(config.n_layers_in_recurrent_block)
        )

        o = config.n_layers_in_prelude + config.n_layers_in_recurrent_block * config.mean_recurrence
        coda = torch.nn.ModuleList(config.Block(config, layer_id=i + o) for i in range(config.n_layers_in_coda))

        transformer_dict = dict(
            wte=torch.nn.Embedding(config.padded_vocab_size, config.n_embd),
            prelude=prelude,
            adapter=adapter,
            core_block=core_block,
            C=C,
            coda=coda,
            ln_f=config.Norm(config.n_embd, eps=config.norm_eps),
        )
        if config.prelude_norm:
            transformer_dict["ln_prelude"] = config.Norm(config.n_embd, eps=config.norm_eps)
        self.transformer = torch.nn.ModuleDict(transformer_dict)

        if config.is_mla:
            # ve rides on the materialized per-head V; every query head has one under MLA.
            kv_dim = config.num_attention_heads * config.v_head_dim
            rec = config.recurrent_block_config
            assert rec.num_attention_heads * rec.v_head_dim == kv_dim, (
                "value_embeds is built once and shared across prelude/core/coda, so the outer and "
                f"recurrent configs must agree on n_head*v_head_dim ({kv_dim} vs "
                f"{rec.num_attention_heads * rec.v_head_dim})."
            )
        else:
            kv_dim = config.num_key_value_heads * (config.n_embd // config.num_attention_heads)
        n_effective_layers = config.n_layers_in_prelude + config.n_layers_in_recurrent_block + config.n_layers_in_coda
        self.value_embeds = torch.nn.ModuleDict({
            str(i): torch.nn.Embedding(config.padded_vocab_size, kv_dim)
            for i in range(n_effective_layers) if has_ve(i, n_effective_layers)
        })

        self.emb_scale = config.init.embedding_scale
        if config.use_fused_head == "cce":
            self.lm_head = config.Linear(
                config.n_embd, config.padded_vocab_size, bias=False, init_method=config.init.fn("head")
            )
        elif config.use_fused_head == "hhe":
            from parcae_lm.ops import LinearCrossEntropyLoss as LCE

            self.lm_head = LCE(
                config.n_embd,
                config.padded_vocab_size,
                ignore_index=objective["ignore_index"],
                init_method=config.init.fn("head"),
            )
        elif self.config.use_fused_head == "full-triton":
            self.lm_head = LinearCrossEntropyLoss(
                config.n_embd,
                config.padded_vocab_size,
                ignore_index=objective["ignore_index"],
                z_regularization=objective["z_regularization"],
                logit_scale=config.init.logit_scale,
                init_method=config.init.fn("head"),
                transposed_weight=not self.config.tie_embeddings,
            )
        elif self.config.use_fused_head == "liger":
            self.lm_head = config.Linear(
                config.n_embd, config.padded_vocab_size, bias=False, init_method=config.init.fn("head")
            )
            from liger_kernel.transformers.fused_linear_cross_entropy import (
                LigerFusedLinearCrossEntropyLoss,
            )
            self._liger_loss_fn = LigerFusedLinearCrossEntropyLoss(
                ignore_index=objective["ignore_index"],
                lse_square_scale=objective["z_regularization"],
                label_smoothing=0.0,
                reduction="mean",
                softcap=float(self.config.logit_softcap) if self.config.logit_softcap else None,
            )
        else:
            self.lm_head = config.Linear(
                config.n_embd, config.padded_vocab_size, bias=False, init_method=config.init.fn("head")
            )
        if self.config.tie_embeddings:
            self.lm_head.weight = self.transformer.wte.weight
        self.objective = objective

        self.max_seq_length = self.config.block_size
        self.gradient_checkpointing = gradient_checkpointing
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=True)

        self.step = 0
        self.monitoring = False
        self.latest_metrics = {}
        self.extreme_monitoring = False
        self.extreme_metrics = {}
        self.recurrent_iteration_method = config.recurrent_iteration_method

        self.reset_parameters()

    def _precompute_freqs_cis(self):
        # Trigger resetting the rope-cache
        dim = self.config.intermediate_size if self.transformer.core_block[0].expanded else self.config.n_embd
        if self.config.randomize_positions_from is not None:
            max_length = self.config.randomize_positions_from
        else:
            max_length = self.config.block_size
        # MLA rotates only the decoupled qk_rope slice
        rope_dim = self.config.qk_rope_head_dim if self.config.is_mla else dim // self.config.num_attention_heads
        freqs_cis = precompute_freqs_cis(
            rope_dim,
            max_length,
            self.config.rope_settings.rope_base,  # 50k in the newer models
            self.config.rope_settings.rope_condense_ratio,
        )  # can actually be a buffer now, and remains in fp32! (at least in the settings I tested)
        return freqs_cis

    def reset_parameters(self) -> None:
        self.config.init.apply(self.transformer.wte, "embedding")
        self.config.init.apply(self.transformer.ln_f, "normalization")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_logits: bool = False,
        num_steps_pair: Optional[torch.Tensor] = None,
        *,
        n_per_sample: Optional[torch.Tensor] = None,
        k_per_sample: Optional[torch.Tensor] = None,
        past_key_values=None,
    ) -> dict[str, Optional[torch.Tensor]]:
        if self.config.randomize_positions_from is not None and self.training:
            position_ids = torch.sort(  # need to fork rng for distributed
                torch.randint(0, self.config.randomize_positions_from, (input_ids.shape[1],), device=input_ids.device)
            )[0]

        if position_ids is None:
            freqs_cis = self.freqs_cis[:, : input_ids.shape[1]]
        else:
            freqs_cis = self.freqs_cis.index_select(1, position_ids)

        input_embeds = self.transformer.wte(input_ids)
        if self.emb_scale != 1:
            input_embeds = input_embeds * self.emb_scale
        
        self._current_input_ids = input_ids

        for i, block in enumerate(self.transformer.prelude):
            ve = self.value_embeds[str(i)](input_ids) if str(i) in self.value_embeds else None
            input_embeds = block(input_embeds, freqs_cis, attention_mask, ve=ve)

        if self.config.prelude_norm:
            input_embeds = self.transformer.ln_prelude(input_embeds)

        x, num_steps_no_grad, num_steps_with_grad, xk, max_steps_no_grad, max_steps_with_grad, min_steps_no_grad, min_steps_with_grad, min_total_recurrence = self.iterate_forward(
            input_embeds,  # type: ignore
            freqs_cis,
            attention_mask,
            num_steps_pair,
            n_per_sample=n_per_sample,
            k_per_sample=k_per_sample,
            past_key_values=past_key_values,
        )
        x_rec_output = x

        x = self.transformer.C(x)
        x_rec_projected = x

        coda_ve_offset = self.config.n_layers_in_prelude + self.config.n_layers_in_recurrent_block
        for i, block in enumerate(self.transformer.coda):
            ve_idx = str(coda_ve_offset + i)
            ve = self.value_embeds[ve_idx](input_ids) if ve_idx in self.value_embeds else None
            if self.gradient_checkpointing and "in-coda" in self.config.activation_checkpoint_impl:
                x = self.config.checkpoint(block, x, freqs_cis, attention_mask, ve=ve)
            else:
                x = block(x, freqs_cis, attention_mask, ve=ve)
        if self.gradient_checkpointing and "in-coda" in self.config.activation_checkpoint_impl:
            x = self.config.checkpoint(self.transformer.ln_f, x)
        else:
            x = self.transformer.ln_f(x)

        if self.monitoring:
            self.monitor_module(
                x, x_rec_output, xk, input_embeds, num_steps_no_grad, num_steps_with_grad, x_rec_projected,
                max_steps_no_grad, max_steps_with_grad, min_steps_no_grad, min_steps_with_grad, min_total_recurrence,
            )

        if labels is not None:
            logits = None
            if self.config.use_fused_head == "cce":
                from cut_cross_entropy import linear_cross_entropy  # type: ignore[unusal import]

                # cce kernel requires bf16/fp16 inputs and weights. Under bf16-mixed the
                # final norm typically leaves x in fp32 and the fp32 master weight is what's
                # stored on lm_head; cast both to bf16 just for the cce call.
                x_bf16 = (x * self.config.init.logit_scale).to(torch.bfloat16)
                weight_bf16 = self.lm_head.weight.to(torch.bfloat16)
                loss = linear_cross_entropy(
                    x_bf16, weight_bf16, labels, filter_eps="auto"
                )
            elif self.config.use_fused_head == "hhe":
                loss = self.lm_head(x * self.config.init.logit_scale, labels)
            elif self.config.use_fused_head == "full-triton":
                loss = self.lm_head(x, labels)  # LinearCrossEntropyLoss applies logit_scale itself
            elif self.config.use_fused_head == "liger":
                # Liger expects flat (N, H) embeddings and (N,) labels.
                # Apply logit_scale on x (matmul is linear, equivalent to scaling logits).
                x_flat = (x * self.config.init.logit_scale).reshape(-1, x.shape[-1])
                labels_flat = labels.reshape(-1)
                loss = _call_liger_eager(self._liger_loss_fn, self.lm_head.weight, x_flat, labels_flat)
            else:
                logits = self.lm_head(x).float() * self.config.init.logit_scale
                if self.config.logit_softcap is not None:
                    softcap = self.config.logit_softcap
                    logits = softcap * torch.tanh(logits / softcap)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.shape[-1]), labels.view(-1),
                    ignore_index=self.objective.get("ignore_index", -100)
                )
            log_ppl = loss.clone().detach()
        else:
            if self.config.use_fused_head == "cce":
                logits = self.lm_head(x).float() * self.config.init.logit_scale
            elif self.config.use_fused_head == "full-triton":
                logits = (
                    torch.matmul(
                        x, self.lm_head.weight.T if self.config.tie_embeddings else self.lm_head.weight
                    ).float()
                    * self.config.init.logit_scale
                )
            else:
                logits = self.lm_head(x).float() * self.config.init.logit_scale
            if self.config.logit_softcap is not None:
                softcap = self.config.logit_softcap
                logits = softcap * torch.tanh(logits / softcap)
            loss, log_ppl = torch.as_tensor(0.0), torch.as_tensor(0.0)

        return {
            "loss": loss,
            "logits": logits if return_logits else None,
            "log_ppl": log_ppl,
        }
    
    def update_recurrent_state(self, x, input_embeds, freq_cis, mask, step: Tensor, total_steps: Tensor, **kwargs):
        return self.core_block_forward(x, input_embeds, freq_cis, mask, step, total_steps, **kwargs)

    def _recurrent_step_kwargs(
        self,
        step: int,
        *,
        past_key_values=None,
    ) -> dict:
        if past_key_values is None:
            return {}
        n_prelude = len(self.transformer.prelude)
        n_core = len(self.transformer.core_block)
        return {
            "past_key_values": past_key_values,
            "step_idx_base": n_prelude + step * n_core,
        }

    @staticmethod
    @contextmanager
    def _no_grad_prefix():
        # Autocast caches each weight's low-precision cast at its first use. Casts made under
        # no_grad have no autograd history, so if the with-grad steps reused them the core
        # weights would get no gradient. Keep those casts out of the cache.
        if torch.compiler.is_compiling():
            with torch.no_grad():
                yield
            return
        cache_enabled = torch.is_autocast_cache_enabled()
        torch.set_autocast_cache_enabled(False)
        try:
            with torch.no_grad():
                yield
        finally:
            torch.set_autocast_cache_enabled(cache_enabled)

    @torch._dynamo.disable(recursive=False)  # type: ignore
    def iterate_forward(
        self,
        input_embeds,
        freqs_cis,
        mask,
        num_steps_pair: Optional[torch.Tensor] = None,
        *,
        n_per_sample: Optional[torch.Tensor] = None,
        k_per_sample: Optional[torch.Tensor] = None,
        past_key_values=None,
    ):
        if (
            past_key_values is not None
            and self.gradient_checkpointing
            and "per-iteration" in self.config.activation_checkpoint_impl
        ):
            raise ValueError("past_key_values is not supported with per-iteration activation checkpointing.")
        x = self.initialize_state(input_embeds)
        per_token = self.recurrent_iteration_method == "per-token"

        if self.recurrent_iteration_method in ["per-batch"]:
            if n_per_sample is not None or k_per_sample is not None:
                raise ValueError("n_per_sample/k_per_sample require per-sequence or per-token iteration")
            if num_steps_pair is None:
                num_steps_no_grad, num_steps_with_grad = self.randomized_iteration_sampler()  # type: ignore
            elif len(num_steps_pair) > 1:
                num_steps_no_grad, num_steps_with_grad = num_steps_pair
            else:
                num_steps_no_grad, num_steps_with_grad = num_steps_pair, torch.tensor(0)

            num_steps_no_grad_int = int(num_steps_no_grad) if isinstance(num_steps_no_grad, torch.Tensor) else num_steps_no_grad
            num_steps_with_grad_int = int(num_steps_with_grad) if isinstance(num_steps_with_grad, torch.Tensor) else num_steps_with_grad
            n_per_sample = None
            k_per_sample = None
        elif self.recurrent_iteration_method in ["per-sequence", "per-token"]:
            t = max(self.config.mean_recurrence - self.config.mean_backprop_depth, 0)
            s = self.config.mean_backprop_depth

            if input_embeds.is_meta:
                num_steps_no_grad_int = t
                num_steps_with_grad_int = s
                num_steps_no_grad = t
                num_steps_with_grad = s
                n_per_sample = None
                k_per_sample = None
            elif n_per_sample is not None and k_per_sample is not None:
                n_per_sample = n_per_sample.to(input_embeds.device)
                k_per_sample = k_per_sample.to(input_embeds.device)
                num_steps_no_grad_int = int(n_per_sample.max().item())
                num_steps_with_grad_int = int(k_per_sample.max().item())
                num_steps_no_grad = n_per_sample.float().mean().to(torch.long)
                num_steps_with_grad = k_per_sample.float().mean().to(torch.long)
            elif num_steps_pair is None:
                seq_len = input_embeds.shape[1] if per_token else None
                n_per_sample, k_per_sample = self._sample_batch_depths(input_embeds.shape[0], seq_len)
                n_per_sample = n_per_sample.to(input_embeds.device)
                k_per_sample = k_per_sample.to(input_embeds.device)
                num_steps_no_grad_int = int(n_per_sample.max().item())
                num_steps_with_grad_int = int(k_per_sample.max().item())
                num_steps_no_grad = n_per_sample.float().mean().to(torch.long)
                num_steps_with_grad = k_per_sample.float().mean().to(torch.long)
            else:
                n_per_sample = None
                k_per_sample = None
                if len(num_steps_pair) > 1:
                    num_steps_no_grad, num_steps_with_grad = num_steps_pair
                else:
                    num_steps_no_grad, num_steps_with_grad = num_steps_pair, torch.tensor(0)
                num_steps_no_grad_int = int(num_steps_no_grad) if isinstance(num_steps_no_grad, torch.Tensor) else num_steps_no_grad
                num_steps_with_grad_int = int(num_steps_with_grad) if isinstance(num_steps_with_grad, torch.Tensor) else num_steps_with_grad
        else:
            raise ValueError(f"Invalid recurrent iteration method: {self.recurrent_iteration_method}")

        total_steps_int = num_steps_no_grad_int + num_steps_with_grad_int
        total_steps = torch.tensor(total_steps_int, device=input_embeds.device)
        # Exposed for RecFT endpoint capture / logging; None on scalar paths.
        self._current_n_per_sample = n_per_sample
        self._current_k_per_sample = k_per_sample

        if self.recurrent_iteration_method in ["per-batch"]:
            with self._no_grad_prefix():
                # ultra annoying in ddp due to
                # https://discuss.pytorch.org/t/does-distributeddataparallel-work-with-torch-no-grad-and-find-unused-parameters-false/122594
                # for now running with find_unused_params=True enabled even though the graph structure is (technically) clear
                # and all parameters are always used
                for step in range(num_steps_no_grad_int):
                    xk = x
                    step_t = torch.tensor(step, device=input_embeds.device)
                    x = self.update_recurrent_state(
                        xk,
                        input_embeds,
                        freqs_cis,
                        mask,
                        step_t,
                        total_steps,
                        **self._recurrent_step_kwargs(step, past_key_values=past_key_values),
                    )


            for step in range(num_steps_with_grad_int):
                xk = x
                abs_step = num_steps_no_grad_int + step
                step_t = torch.tensor(abs_step, device=input_embeds.device)
                step_kwargs = self._recurrent_step_kwargs(
                    abs_step, past_key_values=past_key_values
                )
                if self.gradient_checkpointing and "per-iteration" in self.config.activation_checkpoint_impl:
                    x = self.config.checkpoint(
                        self.update_recurrent_state, xk, input_embeds, freqs_cis, mask, step_t, total_steps
                    )
                else:
                    x = self.update_recurrent_state(
                        xk, input_embeds, freqs_cis, mask, step_t, total_steps, **step_kwargs
                    )

        elif self.recurrent_iteration_method in ["per-sequence", "per-token"]:
            if n_per_sample is None:
                with self._no_grad_prefix():
                    for step in range(num_steps_no_grad_int):
                        xk = x
                        step_t = torch.tensor(step, device=input_embeds.device)
                        x = self.update_recurrent_state(
                            xk,
                            input_embeds,
                            freqs_cis,
                            mask,
                            step_t,
                            total_steps,
                            **self._recurrent_step_kwargs(step, past_key_values=past_key_values),
                        )

                for step in range(num_steps_with_grad_int):
                    xk = x
                    abs_step = num_steps_no_grad_int + step
                    step_t = torch.tensor(abs_step, device=input_embeds.device)
                    step_kwargs = self._recurrent_step_kwargs(
                        abs_step, past_key_values=past_key_values
                    )
                    if self.gradient_checkpointing and "per-iteration" in self.config.activation_checkpoint_impl:
                        x = self.config.checkpoint(
                            self.update_recurrent_state, xk, input_embeds, freqs_cis, mask, step_t, total_steps
                        )
                    else:
                        x = self.update_recurrent_state(
                            xk, input_embeds, freqs_cis, mask, step_t, total_steps, **step_kwargs
                        )
            else:
                # Normal mode - use torch.where to gate updates per sample (or per token)
                with self._no_grad_prefix():
                    for step in range(num_steps_no_grad_int):
                        xk = x
                        active_mask = (step < n_per_sample)  # [batch_size] or [batch_size, seq_len]
                        step_t = torch.tensor(step, device=input_embeds.device)
                        x_new = self.update_recurrent_state(
                            xk,
                            input_embeds,
                            freqs_cis,
                            mask,
                            step_t,
                            total_steps,
                            **self._recurrent_step_kwargs(step, past_key_values=past_key_values),
                        )
                        mask_expanded = active_mask[..., None] if per_token else active_mask[:, None, None]
                        x = torch.where(mask_expanded, x_new, x)

                for step in range(num_steps_with_grad_int):
                    xk = x
                    active_mask = (step < k_per_sample)  # [batch_size] or [batch_size, seq_len]
                    abs_step = num_steps_no_grad_int + step
                    step_t = torch.tensor(abs_step, device=input_embeds.device)
                    step_kwargs = self._recurrent_step_kwargs(
                        abs_step, past_key_values=past_key_values
                    )
                    if self.gradient_checkpointing and "per-iteration" in self.config.activation_checkpoint_impl:
                        x_new = self.config.checkpoint(
                            self.update_recurrent_state, xk, input_embeds, freqs_cis, mask, step_t, total_steps
                        )
                    else:
                        x_new = self.update_recurrent_state(
                            xk, input_embeds, freqs_cis, mask, step_t, total_steps, **step_kwargs
                        )
                    mask_expanded = active_mask[..., None] if per_token else active_mask[:, None, None]
                    x = torch.where(mask_expanded, x_new, x)
        else:
            raise ValueError(f"Invalid recurrent iteration method: {self.recurrent_iteration_method}")

        max_steps_no_grad = torch.tensor(num_steps_no_grad_int, device=input_embeds.device, dtype=torch.long)
        max_steps_with_grad = torch.tensor(num_steps_with_grad_int, device=input_embeds.device, dtype=torch.long)
        if n_per_sample is not None and k_per_sample is not None:
            min_steps_no_grad = n_per_sample.min().to(device=input_embeds.device, dtype=torch.long)
            min_steps_with_grad = k_per_sample.min().to(device=input_embeds.device, dtype=torch.long)
            min_total_recurrence = (n_per_sample + k_per_sample).min().to(device=input_embeds.device, dtype=torch.long)
        else:
            min_steps_no_grad = max_steps_no_grad
            min_steps_with_grad = max_steps_with_grad
            min_total_recurrence = max_steps_no_grad + max_steps_with_grad
        return (
            x,
            num_steps_no_grad,
            num_steps_with_grad,
            xk.detach(),
            max_steps_no_grad,
            max_steps_with_grad,
            min_steps_no_grad,
            min_steps_with_grad,
            min_total_recurrence,
        )
    
    def core_block_forward(
        self, x, input_embeds, freqs_cis, mask, step: Tensor, total_steps: Tensor, **kwargs
    ):
        x = self.transformer.adapter(x,input_embeds)

        past_key_values = kwargs.get("past_key_values")
        # Honor explicit cache/step indexing during training so RecFT can reuse
        # iterate_forward while still capturing recurrence endpoints.
        if "step_idx_base" in kwargs or past_key_values is not None:
            step_idx_base = kwargs.get("step_idx_base", 0)
        elif not self.training:
            step_idx_base = kwargs.get("step_idx_base", 0)
        else:
            past_key_values = None
            step_idx_base = 0

        core_ve_offset = self.config.n_layers_in_prelude
        for layer_idx, block in enumerate(self.transformer.core_block):
            ve_idx = str(core_ve_offset + layer_idx)
            ve = self.value_embeds[ve_idx](self._current_input_ids) if ve_idx in self.value_embeds else None
            block_kwargs = {"ve": ve}
            if past_key_values is not None:
                block_kwargs["past_key_values"] = past_key_values
                # MLA read-time ve reconstruction needs the layer's value_embeds table. Ignored for MHA / non-cache paths.
                block_kwargs["ve_table"] = self.value_embeds[ve_idx] if ve_idx in self.value_embeds else None
                block_kwargs["step_idx"] = torch.tensor(step_idx_base + layer_idx, dtype=torch.long)
            if self.gradient_checkpointing and "per-block" in self.config.activation_checkpoint_impl:
                x = self.config.checkpoint(block, x, freqs_cis, mask, **block_kwargs)
            else:
                x = block(x, freqs_cis, mask, **block_kwargs)
        return x

    def _truncated_full_depths(
        self, total: torch.Tensor, s: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k = torch.clamp(total, max=s)
        n = total - k
        return n, k

    @torch._dynamo.disable(recursive=False)  # type: ignore
    def randomized_iteration_sampler(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Outputs are long tensors so that they can be passed through compiled functions"""
        if torch.rand((1,)).is_meta:  # annoying clause to make meta-tensor-based flop counting work
            # these values are only approximate, not all schemes exactly target a mean of n and k
            # they overvalue the compute done when curricula are turned on, but that may be considered
            # a feature, given that it is a valid form of training acceleration
            return self.config.mean_recurrence - self.config.mean_backprop_depth, self.config.mean_backprop_depth  # type: ignore

        seed_n = 514229 + self.step  # easiest way to make the sampler re-runnable in checkpointing
        seed_k = 317811 + self.step
        if not self.config.lockstep_n and torch.distributed.is_initialized():
            seed_n = seed_n * (torch.distributed.get_rank() + 1)
        if not self.config.lockstep_k and torch.distributed.is_initialized():
            seed_k = seed_k * (torch.distributed.get_rank() + 1)

        n_generator = torch.Generator(device="cpu")
        n_generator.manual_seed(seed_n % (2**31 - 1))
        k_generator = torch.Generator(device="cpu")
        k_generator.manual_seed(seed_k % (2**31 - 1))

        if "curriculum-" in self.config.sampling_scheme:
            # Parse curriculum type and length
            # Formats: "...-curriculum-{length}" (linear) or "...-curriculum-sqrt-{length}" (1-sqrt)
            curriculum_part = self.config.sampling_scheme.split("curriculum-")[1]
            if curriculum_part.startswith("sqrt-"):
                # 1-sqrt schedule: f(step) = ceil(tgt * (1 - sqrt(1 - step/ramp_length)))
                ramp_length = int(curriculum_part.split("sqrt-")[1])
                use_sqrt_schedule = True
            else:
                # Linear schedule: f(step) = ceil(tgt * step/ramp_length)
                ramp_length = int(curriculum_part)
                use_sqrt_schedule = False
            
            t_full = max(self.config.mean_recurrence - self.config.mean_backprop_depth, 0)
            s_full = self.config.mean_backprop_depth
            
            if self.step > ramp_length:
                t, s = t_full, s_full
            else:
                if use_sqrt_schedule:
                    progress = 1 - math.sqrt(1 - self.step / ramp_length)
                else:
                    progress = self.step / ramp_length
                
                # Apply curriculum based on target (default: forward only)
                if self.config.curriculum_target == "forward":
                    t = max(math.ceil(progress * t_full), 0)
                    s = s_full
                elif self.config.curriculum_target == "backward":
                    t = t_full
                    s = max(math.ceil(progress * s_full), 1)
                else:  # "both"
                    t = max(math.ceil(progress * t_full), 0)
                    s = max(math.ceil(progress * s_full), 1)
        else:
            t = max(self.config.mean_recurrence - self.config.mean_backprop_depth, 0)
            s = self.config.mean_backprop_depth

        if self.training:
            if "poisson-unbounded" in self.config.sampling_scheme:
                n = torch.poisson(torch.tensor([t], dtype=torch.float), generator=n_generator)
                k = torch.randint(low=1, high=2 * s + 1, size=(1,), generator=k_generator)
            elif "poisson-fill" in self.config.sampling_scheme:
                n = torch.poisson(torch.tensor([t], dtype=torch.float), generator=n_generator)
                k = torch.as_tensor(s)
            elif "poisson-truncated-full" in self.config.sampling_scheme:
                total = torch.clamp(torch.poisson(torch.tensor([t + s], dtype=torch.float), generator=n_generator), min=1)
                n, k = self._truncated_full_depths(total, int(s))
            elif "poisson-full" in self.config.sampling_scheme:
                n = torch.as_tensor(0)
                k = torch.clamp(torch.poisson(torch.tensor([t + s], dtype=torch.float), generator=k_generator), min=1)
            elif "poisson-bounded" in self.config.sampling_scheme:
                n = torch.minimum(
                    torch.poisson(torch.tensor([t], dtype=torch.float), generator=n_generator),
                    torch.as_tensor(2 * t - 1),
                )
                k = torch.randint(low=1, high=2 * s + 1, size=(1,), generator=k_generator)
            elif "fixed" in self.config.sampling_scheme:
                n, k = torch.as_tensor(t), torch.as_tensor(s)
            else:
                # Default: poisson-unbounded
                n = torch.poisson(torch.tensor([t], dtype=torch.float), generator=n_generator)
                k = torch.randint(low=1, high=2 * s + 1, size=(1,), generator=k_generator)
        else:
            n, k = torch.as_tensor(self.config.mean_recurrence), torch.as_tensor(0)

        return n.to(dtype=torch.long), k.to(dtype=torch.long)

    @torch._dynamo.disable(recursive=False)  # type: ignore
    def _sample_batch_depths(self, batch_size: int, seq_len: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (batch_size, seq_len) if seq_len is not None else (batch_size,)
        if not self.training:
            # Eval: every sequence runs mean_recurrence steps, as in randomized_iteration_sampler.
            return (
                torch.full(shape, self.config.mean_recurrence, dtype=torch.long),
                torch.zeros(shape, dtype=torch.long),
            )
        if "curriculum-" in self.config.sampling_scheme:
            curriculum_part = self.config.sampling_scheme.split("curriculum-")[1]
            if curriculum_part.startswith("sqrt-"):
                ramp_length = int(curriculum_part.split("sqrt-")[1])
                use_sqrt_schedule = True
            else:
                ramp_length = int(curriculum_part)
                use_sqrt_schedule = False
            
            t_full = max(self.config.mean_recurrence - self.config.mean_backprop_depth, 0)
            s_full = self.config.mean_backprop_depth
            
            if self.step > ramp_length:
                t, s = t_full, s_full
            else:
                if use_sqrt_schedule:
                    progress = 1 - math.sqrt(1 - self.step / ramp_length)
                else:
                    progress = self.step / ramp_length
                if self.config.curriculum_target == "forward":
                    t = max(math.ceil(progress * t_full), 0)
                    s = s_full
                elif self.config.curriculum_target == "backward":
                    t = t_full
                    s = max(math.ceil(progress * s_full), 1)
                else:  # "both"
                    t = max(math.ceil(progress * t_full), 0)
                    s = max(math.ceil(progress * s_full), 1)
        else:
            t = max(self.config.mean_recurrence - self.config.mean_backprop_depth, 0)
            s = self.config.mean_backprop_depth
        
        # Use CPU generator with proper seeding (matches randomized_iteration_sampler)
        seed = 514229 + self.step
        if not self.config.lockstep_n and torch.distributed.is_initialized():
            seed = seed * (torch.distributed.get_rank() + 1)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed % (2**31 - 1))
        
        # Sample on CPU - per batch element (or per token if seq_len provided)
        if "poisson-truncated-full" in self.config.sampling_scheme:
            total = torch.clamp(torch.poisson(torch.full(shape, float(t + s)), generator=generator), min=1)
            n, k = self._truncated_full_depths(total, int(s))
            # Avoid a rank-local all-zero no-grad prefix. In eager DDP this can
            # desynchronize gradient collectives when it lands on a sync step.
            while bool(torch.all(n == 0)):
                total = torch.clamp(
                    torch.poisson(torch.full(shape, float(t + s)), generator=generator),
                    min=1,
                )
                n, k = self._truncated_full_depths(total, int(s))
        elif "poisson-full" in self.config.sampling_scheme:
            n = torch.zeros(shape, dtype=torch.long)
            k = torch.clamp(torch.poisson(torch.full(shape, float(t + s)), generator=generator), min=1)
        elif "fixed" in self.config.sampling_scheme:
            n = torch.full(shape, t, dtype=torch.long)
            k = torch.full(shape, s, dtype=torch.long)
        else:
            # Default: poisson for n, fixed for k
            n = torch.poisson(torch.full(shape, float(t)), generator=generator).to(torch.long)
            k = torch.full(shape, s, dtype=torch.long)
        
        return n.to(dtype=torch.long), k.to(dtype=torch.long)

    def initialize_state(self, input_embeds):
        shape = list(input_embeds.size())
        shape[-1] = self.config.recurrent_embedding_dimension
        if self.config.state_init == "normal":
            x = torch.randn(tuple(shape), device=input_embeds.device, dtype=input_embeds.dtype)
        elif self.config.state_init == "embed":  # initialized like a scaled embedding:
            x = torch.randn(tuple(shape), device=input_embeds.device, dtype=input_embeds.dtype).mul(1 / math.sqrt(input_embeds.shape[-1]))
        elif self.config.state_init == "like-init":
            x = torch.randn(tuple(shape), device=input_embeds.device, dtype=input_embeds.dtype)
            std = self.config.init.get_std("embedding")
            torch.nn.init.trunc_normal_(x, mean=0.0, std=std, a=-3 * std, b=3 * std)
            if self.emb_scale != 1:
                x = x * self.emb_scale
        elif self.config.state_init == "zero":
            x = torch.zeros(tuple(shape), device=input_embeds.device, dtype=input_embeds.dtype)
        elif self.config.state_init == "unit":
            x = torch.randn(tuple(shape), device=input_embeds.device, dtype=input_embeds.dtype)
            std, mean = torch.std_mean(x, dim=-1, keepdim=True)
            x = (x - mean) / (std + 1e-8)
        return x

    @torch.no_grad()
    def monitor_module(
        self,
        x_out: torch.Tensor,
        x_rec: torch.Tensor,
        xk: torch.Tensor,
        input_embeds: torch.Tensor,
        num_steps_no_grad: torch.Tensor,
        num_steps_with_grad: torch.Tensor,
        x_rec_projected: torch.Tensor,
        max_steps_no_grad: torch.Tensor,
        max_steps_with_grad: torch.Tensor,
        min_steps_no_grad: torch.Tensor,
        min_steps_with_grad: torch.Tensor,
        min_total_recurrence: torch.Tensor,
    ):
        x_out_c = x_out - x_out.mean(dim=-1, keepdim=True)
        normed_x = x_out_c / x_out_c.norm(dim=-1, keepdim=True)
        token_corr = (normed_x @ normed_x.transpose(1, 2)).mean() - 1 / x_out.shape[1]

        x_rec_c = x_rec - x_rec.mean(dim=-1, keepdim=True)
        normed_x = x_rec_c / x_rec_c.norm(dim=-1, keepdim=True)
        token_corr_rec = (normed_x @ normed_x.transpose(1, 2)).mean() - 1 / x_rec.shape[1]
        prelude_output_norm = input_embeds.norm(dim=-1).mean()
        recurrent_state_norm = x_rec.norm(dim=-1).mean()

        metrics = {
            "last_hidden_token_corr": token_corr,
            "recurrent_state_token_corr": token_corr_rec,
            "last_hidden_norm": x_out.norm(dim=-1).mean(),
            "prelude_output_norm": prelude_output_norm,
            "recurrent_state_norm": recurrent_state_norm,
            "recurrent_to_prelude_norm_ratio": (
                recurrent_state_norm / prelude_output_norm.clamp_min(1e-8)
            ),
            "recurrent_diff": (x_rec_projected - input_embeds).norm(dim=-1).mean(),
            "num_steps_no_grad": num_steps_no_grad,
            "num_steps_with_grad": num_steps_with_grad,
            "max_steps_no_grad": max_steps_no_grad,
            "max_steps_with_grad": max_steps_with_grad,
            "max_total_recurrence": max_steps_no_grad + max_steps_with_grad,
            "min_steps_no_grad": min_steps_no_grad,
            "min_steps_with_grad": min_steps_with_grad,
            "min_total_recurrence": min_total_recurrence,
            "recurrent_residual": (x_rec - xk).norm(dim=-1).mean(),
            "rel_residual": ((x_rec - xk).norm(dim=-1) / x_rec.norm(dim=-1)).mean(),
        }
        adapter = self.transformer.adapter
        if all(hasattr(adapter, name) for name in ("A_log", "dt_bias", "B")):
            dt = torch.nn.functional.softplus(adapter.dt_bias)
            decay = torch.exp(-dt * torch.exp(adapter.A_log))
            injection = dt * (input_embeds @ adapter.B.T)
            metrics.update(
                {
                    "adapter_injection_norm": injection.norm(dim=-1).mean(),
                    "adapter_decay_mean": decay.mean(),
                    "adapter_decay_max": decay.max(),
                }
            )
        self.latest_metrics = metrics
        
        if self.extreme_monitoring:
            extreme = self.compute_extreme_metrics()
            self.latest_metrics.update(extreme)
            self.extreme_metrics = extreme

    @torch.no_grad()
    @torch._dynamo.disable
    def compute_extreme_metrics(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        
        adapter = self.transformer.adapter
        if isinstance(adapter, torch.nn.Identity):
            pass
        elif isinstance(adapter, torch.nn.Sequential):
            for module in adapter:
                if hasattr(module, 'weight') and module.weight.dim() == 2:
                    W = module.weight.float()
                    d_in, d_out = W.shape[1], W.shape[0]
                    if d_in == 2 * d_out:
                        S_x = torch.linalg.svdvals(W[:, :d_out])
                        S_e = torch.linalg.svdvals(W[:, d_out:])
                        metrics["extreme/adapter_Wx_spectral_norm"] = S_x[0].item()
                        metrics["extreme/adapter_We_spectral_norm"] = S_e[0].item()
                    else:
                        metrics["extreme/adapter_spectral_norm"] = torch.linalg.svdvals(W)[0].item()
                    break
        elif hasattr(adapter, 'weight'):
            W = adapter.weight.float()
            d_in, d_out = W.shape[1], W.shape[0]
            if d_in == 2 * d_out:
                S_x = torch.linalg.svdvals(W[:, :d_out])
                S_e = torch.linalg.svdvals(W[:, d_out:])
                metrics["extreme/adapter_Wx_spectral_norm"] = S_x[0].item()
                metrics["extreme/adapter_We_spectral_norm"] = S_e[0].item()
            else:
                metrics["extreme/adapter_spectral_norm"] = torch.linalg.svdvals(W)[0].item()
        
        return metrics
