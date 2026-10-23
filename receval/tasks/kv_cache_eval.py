import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from receval.tasks.core_eval import print0, run_core_eval


class KVCacheModelWrapper(nn.Module):
    def __init__(
        self,
        model,
        lookup_strategy="compress-s1",
        num_steps=None,
        mode="token_by_token",
    ):
        super().__init__()
        self.model = model
        self.lookup_strategy = lookup_strategy
        self._num_steps = num_steps
        self.mode = mode

    @property
    def config(self):
        return self.model.config

    @property
    def num_steps(self):
        return self._num_steps if self._num_steps is not None else self.model.config.mean_recurrence

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def forward(self, input_ids, **kwargs):
        return self._run(input_ids)

    def forward_for_generation(self, input_ids, **kwargs):
        return self._run(input_ids)

    @torch.no_grad()
    def forward_tokens(self, input_ids, prefix_lengths=None):
        if self.mode != "prefix_decode" or prefix_lengths is None:
            logits = self._run(input_ids)["logits"]
            targets = torch.roll(input_ids.to(logits.device), -1, 1)
            losses = F.cross_entropy(
                logits.flatten(0, 1).float(),
                targets.flatten(),
                reduction="none",
            ).view_as(targets)
            losses[:, -1] = float("nan")
            return losses.cpu(), logits.argmax(-1).cpu()
        return self._prefix_decode(input_ids, prefix_lengths)

    @torch.no_grad()
    def _run(self, input_ids):
        input_ids = input_ids.to(next(self.model.parameters()).device)
        cache = self._create_cache()
        if self.mode == "prefill":
            return self._prefill(input_ids, cache)
        return self._token_by_token(input_ids, cache)

    def _create_cache(self):
        return self.model.create_cache(
            lookup_strategy=self.lookup_strategy,
            num_steps=self.num_steps,
        )

    def _token_by_token(self, input_ids, cache):
        logits = []
        for t in range(input_ids.shape[1]):
            out = self.model.forward_for_generation(input_ids[:, t:t+1], num_steps=self.num_steps, past_key_values=cache)
            logits.append(out["logits"])
        return {"logits": torch.cat(logits, dim=1)}

    def _prefix_decode(self, input_ids, prefix_lengths):
        batch_size, sequence_length = input_ids.shape
        device = next(self.model.parameters()).device
        losses = torch.full(
            (batch_size, sequence_length),
            float("nan"),
            dtype=torch.float32,
        )
        predictions = torch.zeros_like(input_ids)

        groups = defaultdict(list)
        for row_index, prefix_length in enumerate(prefix_lengths):
            prefix_length = max(0, min(int(prefix_length), sequence_length))
            groups[prefix_length].append(row_index)

        for prefix_length, row_indices in groups.items():
            rows = input_ids[row_indices].to(device)
            cache = self._create_cache()

            if prefix_length > 0:
                prefix_rows = rows[:, :prefix_length]
                shared_prefix = (
                    len(row_indices) > 1
                    and torch.equal(prefix_rows, prefix_rows[:1].expand_as(prefix_rows))
                )
                output = self.model.forward_for_generation(
                    prefix_rows[:1] if shared_prefix else prefix_rows,
                    num_steps=self.num_steps,
                    past_key_values=cache,
                )
                if shared_prefix:
                    for store in (
                        cache.key_cache,
                        cache.value_cache,
                        cache.latent_cache,
                        cache.token_cache,
                    ):
                        for key, positions in list(store.items()):
                            if isinstance(positions, dict):
                                for position, tensor in positions.items():
                                    positions[position] = tensor.expand(
                                        len(row_indices), *tensor.shape[1:]
                                    )
                            else:
                                store[key] = positions.expand(
                                    len(row_indices), *positions.shape[1:]
                                )
                    for state in cache.kda_state_cache.values():
                        state["conv_state"] = tuple(
                            value.expand(len(row_indices), *value.shape[1:])
                            if isinstance(value, torch.Tensor)
                            else None
                            for value in state["conv_state"]
                        )
                        recurrent_state = state["recurrent_state"]
                        state["recurrent_state"] = recurrent_state.expand(
                            len(row_indices), *recurrent_state.shape[1:]
                        )
                    prefix_logits = output["logits"][:, -1].expand(
                        len(row_indices), -1
                    )
                else:
                    prefix_logits = output["logits"][:, -1]
                if prefix_length < sequence_length:
                    self._record_prediction(
                        prefix_logits,
                        rows[:, prefix_length],
                        losses,
                        predictions,
                        row_indices,
                        prefix_length - 1,
                    )

            # Candidate suffixes must advance one token at a time. With shared
            # recurrence slots, a later token must read the final cache written
            # by each earlier token, which a multi-token forward cannot expose.
            for position in range(prefix_length, sequence_length - 1):
                output = self.model.forward_for_generation(
                    rows[:, position : position + 1],
                    num_steps=self.num_steps,
                    past_key_values=cache,
                )
                self._record_prediction(
                    output["logits"][:, 0],
                    rows[:, position + 1],
                    losses,
                    predictions,
                    row_indices,
                    position,
                )

        return losses, predictions

    @staticmethod
    def _record_prediction(
        logits,
        target,
        losses,
        predictions,
        row_indices,
        prediction_index,
    ):
        losses[row_indices, prediction_index] = F.cross_entropy(
            logits.float(),
            target,
            reduction="none",
        ).cpu()
        predictions[row_indices, prediction_index] = logits.argmax(-1).cpu()

    def _prefill(self, input_ids, cache):
        if input_ids.shape[1] <= 1:
            return {"logits": self.model.forward_for_generation(input_ids, num_steps=self.num_steps, past_key_values=cache)["logits"]}
        prefix = self.model.forward_for_generation(input_ids[:, :-1], num_steps=self.num_steps, past_key_values=cache)
        decode = self.model.forward_for_generation(input_ids[:, -1:], num_steps=self.num_steps, past_key_values=cache)
        return {"logits": torch.cat([prefix["logits"], decode["logits"]], dim=1)}


def run_kv_cache_core_eval(model, tokenizer, device, max_seq_len=None, max_per_task=-1, seeds=None, strategies=None, num_steps=None):
    strategies = strategies or ["full", "compress-s1"]
    seeds = seeds or [1234]
    all_results = {}
    for strategy in strategies:
        print0(f"\n{'='*60}\nKV-Cache CORE eval  strategy={strategy}  num_steps={num_steps}\n{'='*60}")
        wrapped = KVCacheModelWrapper(model, lookup_strategy=strategy, num_steps=num_steps, mode="prefill")
        all_results[strategy] = run_core_eval(wrapped, tokenizer, device, max_seq_len=max_seq_len, max_per_task=max_per_task, seeds=seeds)
    if len(strategies) > 1:
        print0(f"\n{'='*60}\nKV-Cache CORE comparison\n{'='*60}")
        for s in strategies:
            print0(f"  {s}: CORE={all_results[s]['core_metric']:.4f}")
    return all_results
