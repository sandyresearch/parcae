from __future__ import annotations

import queue
import random
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch


def parquet_batches(
    data_dir: str,
    tokenizer: Any,
    sequence_length: int,
    batch_size: int,
    max_batches: int,
    *,
    rank: int = 0,
    world_size: int = 1,
    skip_batches: int = 0,
    seed: int | None = None,
) -> Iterator[torch.Tensor]:
    """Yield rank-sharded, pinned CPU batches."""
    import pyarrow.parquet as pq

    files = sorted(Path(data_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files found under {data_dir}")
    rng = random.Random(seed)
    if seed is not None:
        rng.shuffle(files)

    encoder = (
        tokenizer.processor
        if hasattr(tokenizer, "processor")
        else tokenizer
    )
    batch: list[list[int]] = []
    token_buffer: list[int] = []
    eos_id = getattr(tokenizer, "eos_id", None)
    if eos_id is None:
        eos_id = getattr(tokenizer, "eos_token_id", None)
    yielded = 0
    local_batch_index = 0
    global_row_index = 0

    for path in files:
        texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
        if seed is not None:
            rng.shuffle(texts)
        for text in texts:
            mine = global_row_index % world_size == rank
            global_row_index += 1
            if not mine:
                continue
            encoded = encoder.encode(text)
            if hasattr(encoded, "tolist"):
                encoded = encoded.tolist()
            else:
                encoded = list(encoded)
            token_buffer.extend(encoded)
            if eos_id is not None:
                token_buffer.append(eos_id)

            while len(token_buffer) >= sequence_length:
                batch.append(token_buffer[:sequence_length])
                del token_buffer[:sequence_length]
                if len(batch) != batch_size:
                    continue

                if local_batch_index < skip_batches:
                    local_batch_index += 1
                    batch = []
                    continue

                yield torch.tensor(
                    batch,
                    dtype=torch.long,
                    pin_memory=True,
                )
                yielded += 1
                local_batch_index += 1
                batch = []
                if yielded >= max_batches:
                    return


class PrefetchIterator:
    _END = object()

    def __init__(self, source: Iterator[torch.Tensor], depth: int = 2) -> None:
        self.source = source
        self.items: queue.Queue[object] = queue.Queue(maxsize=max(1, depth))
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._fill, daemon=True)
        self.thread.start()

    def _fill(self) -> None:
        try:
            for item in self.source:
                self.items.put(item)
        except BaseException as exc:
            self.error = exc
        finally:
            self.items.put(self._END)

    def __iter__(self) -> "PrefetchIterator":
        return self

    def __next__(self) -> torch.Tensor:
        item = self.items.get()
        if item is self._END:
            if self.error is not None:
                raise self.error
            raise StopIteration
        if not isinstance(item, torch.Tensor):
            raise TypeError(f"unexpected prefetched item: {type(item)}")
        return item
