import dataclasses
import json
import torch
from pathlib import Path
from typing import Optional, Union
from parcae_lm.models.parcae import Parcae, ParcaeConfig

class ModelingParcae(Parcae):
    def __init__(self, config: ParcaeConfig, objective=None, gradient_checkpointing=False) -> None:
        if objective is None:
            objective = {"ignore_index": -100, "z_regularization": 0.0}
        super().__init__(config, objective, gradient_checkpointing)
        self._generation_config = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, Path], device=None, dtype=None, **kwargs):
        path = Path(pretrained_model_name_or_path)
        if not path.exists():
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(repo_id=str(pretrained_model_name_or_path), allow_patterns=["*.json", "*.bin", "*.safetensors"]))
        from parcae_lm.models.config import RoPESettings
        with open(path / "config.json") as f:
            config_dict = json.load(f)
        if "rope_settings" in config_dict and isinstance(config_dict["rope_settings"], dict):
            config_dict["rope_settings"] = RoPESettings(**config_dict["rope_settings"])
        config_dict.update(kwargs)
        for key in ["_class_name", "init"]:
            config_dict.pop(key, None)
        # Released configs also carry rve_*, gdn_* and recurrence_depth_variance; they are dropped when inert.
        if config_dict.get("rve_rank", 1) > 1:
            raise ValueError("rve checkpoints are not supported")
        if "GDN" in str(config_dict.get("attention_class_name", "")):
            raise ValueError("GDN checkpoints are not supported")
        if "fixed-var" in str(config_dict.get("sampling_scheme", "")):
            raise ValueError("fixed-var sampling schemes are not supported")
        known = {f.name for f in dataclasses.fields(ParcaeConfig)}
        inert = {"rve_rank", "rve_bias", "rve_scale_by_sqrt_rank", "recurrence_depth_variance"}
        for key in list(config_dict):
            if key in known:
                continue
            if key in inert or key.startswith("gdn_"):
                config_dict.pop(key)
            else:
                raise ValueError(f"unknown config key {key!r}")
        model = cls(ParcaeConfig(**config_dict))
        weights_path = None
        for name in ["pytorch_model.bin", "model.safetensors", "model.bin"]:
            if (path / name).exists():
                weights_path = path / name
                break
        if weights_path is None:
            raise FileNotFoundError(f"No weights found in {path}")
        if weights_path.suffix == ".safetensors":
            from safetensors.torch import load_file
            state_dict = load_file(weights_path)
        else:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        cleaned = {}
        for k, v in state_dict.items():
            for prefix in ["module.", "_orig_mod.", "model."]:
                if k.startswith(prefix):
                    k = k[len(prefix):]
            cleaned[k] = v
        model.load_state_dict(cleaned, strict=False)
        if dtype:
            model = model.to(dtype=dtype)
        if device:
            model = model.to(device=device)
        model.eval()
        return model

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        config_dict = self.config.__getstate__()
        for key in ["_recurrent_block_config", "init"]:
            config_dict.pop(key, None)
        with open(save_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
        torch.save(self.state_dict(), save_dir / "pytorch_model.bin")

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def generation_config(self):
        if self._generation_config is None:
            from transformers import GenerationConfig
            self._generation_config = GenerationConfig(max_length=self.config.block_size, do_sample=True, temperature=1.0)
        return self._generation_config

    @generation_config.setter
    def generation_config(self, value):
        self._generation_config = value

    def create_cache(self, lookup_strategy: str = "full", num_steps: Optional[int] = None):
        from parcae_lm.utils.cache import ParcaeDynamicCache
        if num_steps is None:
            num_steps = self.config.mean_recurrence
        n_prelude = len(self.transformer.prelude)
        n_core = len(self.transformer.core_block)
        return ParcaeDynamicCache(lookup_strategy=lookup_strategy, core_step_range=(n_prelude, n_prelude + num_steps * n_core), n_core=n_core)

    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> dict:
        return {"input_ids": input_ids, "attention_mask": attention_mask, "num_steps": kwargs.get("num_steps", self.config.mean_recurrence)}

    def forward_for_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        past_key_values=None,
        grad_loops=None,
        **kwargs,
    ) -> dict:
        # grad_loops: if not None, only those recurrence indices keep autograd; every
        # other recurrence runs under torch.no_grad() (detaching x). This matches RecFT
        # last4first / pretraining-style truncated BPTT without materializing the full
        # T-loop graph. Default None => all loops grad => original eval/generate behavior.
        if num_steps is None:
            num_steps = self.config.mean_recurrence
        seq_len = input_ids.shape[1]
        cache_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        if past_key_values is not None:
            past_key_values._current_input_ids = input_ids
        freqs_cis = self.freqs_cis[:, cache_len:cache_len + seq_len]
        self._current_input_ids = input_ids
        x = self.transformer.wte(input_ids)
        if self.emb_scale != 1:
            x = x * self.emb_scale
        for i, block in enumerate(self.transformer.prelude):
            ve_embed = self.value_embeds[str(i)] if str(i) in self.value_embeds else None
            ve = ve_embed(input_ids) if ve_embed is not None else None
            x = block(
                x, freqs_cis, attention_mask, past_key_values=past_key_values,
                step_idx=torch.tensor(i, dtype=torch.long), ve=ve, ve_table=ve_embed,
            )
        if self.config.prelude_norm:
            x = self.transformer.ln_prelude(x)
        input_embeds = x
        n_prelude = len(self.transformer.prelude)
        n_core = len(self.transformer.core_block)
        x = self.initialize_state(input_embeds)
        total_steps = torch.tensor(num_steps)
        for recurrence in range(num_steps):
            step_t = torch.tensor(recurrence)
            step_idx_base = n_prelude + recurrence * n_core
            if grad_loops is not None and recurrence not in grad_loops:
                with torch.no_grad():
                    x = self.core_block_forward(
                        x,
                        input_embeds,
                        freqs_cis,
                        attention_mask,
                        step=step_t,
                        total_steps=total_steps,
                        past_key_values=past_key_values,
                        step_idx_base=step_idx_base,
                    )
            else:
                x = self.core_block_forward(
                    x,
                    input_embeds,
                    freqs_cis,
                    attention_mask,
                    step=step_t,
                    total_steps=total_steps,
                    past_key_values=past_key_values,
                    step_idx_base=step_idx_base,
                )
        x = self.transformer.C(x)
        coda_base = n_prelude + num_steps * n_core
        coda_ve_offset = n_prelude + n_core
        for i, block in enumerate(self.transformer.coda):
            key = str(coda_ve_offset + i)
            ve_embed = self.value_embeds[key] if key in self.value_embeds else None
            ve = ve_embed(input_ids) if ve_embed is not None else None
            x = block(
                x, freqs_cis, attention_mask, past_key_values=past_key_values,
                step_idx=torch.tensor(coda_base + i, dtype=torch.long), ve=ve, ve_table=ve_embed,
            )
        x = self.transformer.ln_f(x)
        if self.config.use_fused_head == "full-triton":
            weight = self.lm_head.weight.T if self.config.tie_embeddings else self.lm_head.weight
            logits = torch.matmul(x, weight).float() * self.config.init.logit_scale
        else:
            logits = self.lm_head(x).float() * self.config.init.logit_scale
        if getattr(self.config, 'logit_softcap', None) is not None:
            softcap = self.config.logit_softcap
            logits = softcap * torch.tanh(logits / softcap)
        return {"logits": logits, "past_key_values": past_key_values}

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100, temperature: float = 1.0, top_k: Optional[int] = None, top_p: Optional[float] = None, do_sample: bool = True, num_steps: Optional[int] = None, use_cache: bool = True, streamer=None, **kwargs) -> torch.Tensor:
        import torch.nn.functional as F
        if num_steps is None:
            num_steps = self.config.mean_recurrence
        past_key_values = self.create_cache(num_steps=num_steps) if use_cache else None
        generated = input_ids.clone()
        current_input = input_ids
        for _ in range(max_new_tokens):
            outputs = self.forward_for_generation(current_input, num_steps=num_steps, past_key_values=past_key_values)
            logits = outputs["logits"][:, -1, :] / max(temperature, 1e-8)
            past_key_values = outputs.get("past_key_values")
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cumulative_probs > top_p
                mask[:, 1:] = mask[:, :-1].clone()
                mask[:, 0] = False
                logits[mask.scatter(1, sorted_indices, mask)] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) if do_sample else logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            current_input = next_token if use_cache else generated
            if streamer is not None:
                streamer.put(next_token.cpu())
        if streamer is not None:
            streamer.end()
        return generated
