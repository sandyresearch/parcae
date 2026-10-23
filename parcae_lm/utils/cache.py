from typing import Optional
import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache


class KVCache(nn.Module):

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        n_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        shape = (batch_size, max_seq_len, n_heads, head_dim)
        self.register_buffer("k_cache", torch.zeros(shape, dtype=dtype, device=device))
        self.register_buffer("v_cache", torch.zeros(shape, dtype=dtype, device=device))
        self.max_seq_len = max_seq_len
        self.seq_len = 0

    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = k.shape[1]
        
        if tok_idx is None:
            tok_idx = torch.arange(self.seq_len, self.seq_len + seq_len, device=k.device)
        
        self.k_cache.index_copy_(1, tok_idx, k)
        self.v_cache.index_copy_(1, tok_idx, v)
        
        self.seq_len = max(self.seq_len, tok_idx.max().item() + 1)
        
        return self.k_cache[:, :self.seq_len], self.v_cache[:, :self.seq_len]

    def reset(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.seq_len = 0

    def get_seq_length(self) -> int:
        return self.seq_len


def _remap_step_idx(lookup_strategy: str, step_idx: int, core_start: int, n_core: int, num_recurrences: int) -> int:
    relative_idx = step_idx - core_start
    recurrence_idx = relative_idx // n_core
    layer_in_rec = relative_idx % n_core

    if "-except" in lookup_strategy:  # compress-s1-except{X}: core layer X keeps a cache for every recurrence
        lookup_strategy, layer = lookup_strategy.split("-except")
        if layer_in_rec == int(layer):
            return step_idx
    elif "-first-l" in lookup_strategy:  # compress-s1-first-l{X}: core layer X uses compress-first1
        lookup_strategy, layer = lookup_strategy.split("-first-l")
        if layer_in_rec == int(layer):
            lookup_strategy = "compress-first1"

    if "compress-first" in lookup_strategy:
        n_keep_rec = int(lookup_strategy.split("compress-first")[1])
        if recurrence_idx >= n_keep_rec:
            new_step_idx = core_start + n_keep_rec * n_core + layer_in_rec
        else:
            new_step_idx = step_idx
    elif "compress-last" in lookup_strategy:
        n_keep_rec = int(lookup_strategy.split("compress-last")[1])
        threshold_rec = num_recurrences - n_keep_rec
        if recurrence_idx < threshold_rec:
            new_step_idx = core_start + layer_in_rec
        else:
            kept_rec_idx = recurrence_idx - threshold_rec + 1
            new_step_idx = core_start + kept_rec_idx * n_core + layer_in_rec
    elif "compress-stride" in lookup_strategy:
        stride = int(lookup_strategy.split("compress-stride")[1])
        aligned_rec = (recurrence_idx // stride)
        new_step_idx = core_start + aligned_rec * n_core + layer_in_rec
    elif "compress-boundaries" in lookup_strategy:
        if recurrence_idx == num_recurrences - 1:
            new_step_idx = core_start + n_core + layer_in_rec
        else:
            new_step_idx = core_start + layer_in_rec
    elif "compress-s" in lookup_strategy:
        compression_stage = int(lookup_strategy.split("compress-")[1][1:])
        new_step_idx = (relative_idx % (compression_stage * n_core)) + core_start
    elif "compress-anchor" in lookup_strategy:
        if step_idx - core_start < 4 * 8:
            new_step_idx = step_idx
        else:
            new_step_idx = 34 + (step_idx - 34) % 4
    else:  # compress-r
        compression_stage = int(lookup_strategy.split("compress-")[1][1:])
        new_step_idx = (step_idx - core_start) // compression_stage + core_start
    return new_step_idx


class ParcaeDynamicCache(DynamicCache):

    def __init__(
        self, 
        lookup_strategy: str = "full", 
        core_step_range: Optional[tuple[int, int]] = None,
        n_core: int = 1,
    ) -> None:
        super().__init__()
        self._seen_tokens = 0
        self.key_cache: dict[int, dict[int, torch.Tensor]] = {}
        self.value_cache: dict[int, dict[int, torch.Tensor]] = {}
        self.latent_cache: dict[int, dict[int, torch.Tensor]] = {}
        self.token_cache: dict[int, torch.Tensor] = {}
        self.kda_state_cache: dict[int, dict[str, object]] = {}
        self.lookup_strategy = lookup_strategy
        self.core_step_range = core_step_range
        self.n_core = n_core

    def _cache_step(self, step_idx: int, lookup_strategy: str) -> int:
        if "compress-" not in lookup_strategy:
            return step_idx
        if self.core_step_range is None:
            raise ValueError(
                f"lookup_strategy {lookup_strategy!r} needs core_step_range=(n_prelude, n_prelude + num_steps * n_core) "
                "and n_core; without them every core and coda layer would share one cache slot."
            )
        core_start, core_end = self.core_step_range
        if core_start <= step_idx < core_end:
            num_recurrences = (core_end - core_start) // self.n_core
            step_idx = _remap_step_idx(lookup_strategy, step_idx, core_start, self.n_core, num_recurrences)
        return step_idx

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        step_idx_tensor: torch.Tensor,
        lookup_strategy: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        step_idx: int = int(step_idx_tensor)
        lookup_strategy = self.lookup_strategy if lookup_strategy is None else lookup_strategy
        step_idx = self._cache_step(step_idx, lookup_strategy)

        if step_idx not in self.key_cache:
            self.key_cache[step_idx] = {}
            self.value_cache[step_idx] = {}

        if step_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        for idx, entry in enumerate(key_states.unbind(dim=-2)):
            if "compress-" not in self.lookup_strategy:
                assert step_idx < 0 or self._seen_tokens - key_states.shape[-2] + idx not in self.key_cache[step_idx]
            self.key_cache[step_idx][self._seen_tokens - key_states.shape[-2] + idx] = entry
        for idx, entry in enumerate(value_states.unbind(dim=-2)):
            self.value_cache[step_idx][self._seen_tokens - value_states.shape[-2] + idx] = entry

        return (
            torch.stack(list(self.key_cache[step_idx].values()), dim=-2),
            torch.stack(list(self.value_cache[step_idx].values()), dim=-2),
        )

    def update_latent_scalar(
        self,
        latent_states: torch.Tensor,
        step_idx_tensor: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cache scalar-path MLA latents using the standard recurrence remap."""
        step_idx = int(step_idx_tensor)
        cache_step = self._cache_step(step_idx, self.lookup_strategy)

        if step_idx == 0:
            self._seen_tokens += latent_states.shape[1]
            if token_ids is None:
                raise ValueError("MLA scalar cache requires token_ids at step_idx 0.")
            start = self._seen_tokens - token_ids.shape[1]
            for offset, ids in enumerate(token_ids.unbind(dim=1)):
                self.token_cache[start + offset] = ids

        positions = self.latent_cache.setdefault(cache_step, {})
        start = self._seen_tokens - latent_states.shape[1]
        for offset, latent in enumerate(latent_states.unbind(dim=1)):
            position = start + offset
            if self.lookup_strategy == "full":
                assert position not in positions
            positions[position] = latent

        latents = torch.stack(list(positions.values()), dim=1)
        tokens = torch.stack(list(self.token_cache.values()), dim=1)
        return latents, tokens

    def _route_kda_state(self, step_idx: int) -> tuple[int, bool, bool]:
        """Route scalar KDA state and decide whether this recurrence may commit it.

        Under ``compress-*`` every recurrence that maps to the same cache slot reads the same state from
        the previous token/chunk. Only the last recurrence mapped to that slot commits its resulting
        state, matching last-write-wins K/V sharing without folding the current token into the state
        multiple times.
        """
        if "compress-" not in self.lookup_strategy:
            return step_idx, True, False
        if self.core_step_range is None:
            raise ValueError("KDA caching with a compress-* lookup strategy requires core_step_range.")

        core_start, core_end = self.core_step_range
        if not (core_start <= step_idx < core_end):
            return step_idx, True, False

        cache_step = self._cache_step(step_idx, self.lookup_strategy)
        later_steps = range(step_idx + self.n_core, core_end, self.n_core)
        is_last_writer = all(self._cache_step(step, self.lookup_strategy) != cache_step for step in later_steps)
        return cache_step, is_last_writer, True

    def get_kda_state(self, step_idx_tensor: torch.Tensor) -> Optional[dict[str, object]]:
        """Read a scalar KDA temporal state for one logical Parcae layer."""
        if step_idx_tensor.ndim != 0:
            raise NotImplementedError("KDA cache supports only scalar step_idx decoding.")
        cache_step, _, clone_state = self._route_kda_state(int(step_idx_tensor))
        state = self.kda_state_cache.get(cache_step)
        if state is None or not clone_state:
            return state

        conv_state = state["conv_state"]
        assert isinstance(conv_state, tuple)
        recurrent_state = state["recurrent_state"]
        assert isinstance(recurrent_state, torch.Tensor)
        return {
            "conv_state": tuple(
                value.clone() if isinstance(value, torch.Tensor) else None
                for value in conv_state
            ),
            "recurrent_state": recurrent_state.clone(),
        }

    def update_kda_state(
        self,
        state: dict[str, object],
        step_idx_tensor: torch.Tensor,
        sequence_length: int,
    ) -> None:
        """Commit KDA state, with last-writer semantics under ``compress-*``."""
        if step_idx_tensor.ndim != 0:
            raise NotImplementedError("KDA cache supports only scalar step_idx decoding.")
        step_idx = int(step_idx_tensor)
        cache_step, should_write, _ = self._route_kda_state(step_idx)
        if step_idx == 0:
            self._seen_tokens += sequence_length
        if should_write:
            self.kda_state_cache[cache_step] = state

    def reset(self) -> None:
        self._seen_tokens = 0
        self.key_cache.clear()
        self.value_cache.clear()
        self.latent_cache.clear()
        self.token_cache.clear()
        self.kda_state_cache.clear()

    def get_seq_length(self) -> int:
        return self._seen_tokens


class GPTKVCache:
    def __init__(self, batch_size, n_layers, n_heads, head_dim, max_seq_len, dtype, device):
        self.n_layers = n_layers
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self.caches = [
            KVCache(batch_size, max_seq_len, n_heads, head_dim, dtype, device)
            for _ in range(n_layers)
        ]

    def get_layer_cache(self, layer_id):
        cache = self.caches[layer_id]
        return cache.k_cache, cache.v_cache

    def advance(self, seq_len):
        self.cache_seqlens += seq_len

    def get_seq_length(self):
        return int(self.cache_seqlens[0].item())
