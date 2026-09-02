import copy
import torch

from typing import Optional, List
from torch.utils.data import IterableDataset, get_worker_info
from torch.utils.data._utils.collate import collate_tensor_fn
from parcae_lm.tokenizer import Tokenizer


class BestFitPackingCollator:
    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        add_bos: bool = True,
        add_eos: bool = True,
        buffer_size: int = 1000,
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.buffer_size = buffer_size
        self.doc_buffer: List[torch.Tensor] = []
        self.row_capacity = block_size
    
    def state_dict(self) -> dict:
        return {
            "doc_buffer": [doc.tolist() for doc in self.doc_buffer],
            "num_docs": len(self.doc_buffer),
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        if state_dict is None:
            return
        doc_buffer_lists = state_dict.get("doc_buffer", [])
        self.doc_buffer = [torch.tensor(doc, dtype=torch.long) for doc in doc_buffer_lists]

    def _refill(self, doc_iter) -> None:
        while len(self.doc_buffer) < self.buffer_size:
            try:
                row = next(doc_iter)
            except StopIteration:
                return
            tokens = self._tokenize_row(row)
            if tokens is not None and len(tokens) > 0:
                self.doc_buffer.append(tokens)

    def _emit(self, n_rows: int, doc_iter=None) -> tuple[torch.Tensor, torch.Tensor, list]:
        row_buffer = torch.empty((n_rows, self.row_capacity), dtype=torch.long)
        for row_idx in range(n_rows):
            pos = 0
            while pos < self.row_capacity:
                if doc_iter is not None:
                    self._refill(doc_iter)
                if not self.doc_buffer:
                    row_buffer[row_idx, pos:] = self.tokenizer.pad_id or 0
                    break

                remaining = self.row_capacity - pos
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(self.doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = self.doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = doc
                    pos += len(doc)
                else:
                    shortest_idx = min(range(len(self.doc_buffer)), key=lambda i: len(self.doc_buffer[i]))
                    doc = self.doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = doc[:remaining]
                    pos += remaining

        input_ids = row_buffer[:, :-1].contiguous()
        label_ids = row_buffer[:, 1:].contiguous()
        metadata = [None] * n_rows
        return input_ids, label_ids, metadata

    def __call__(self, batch) -> tuple[torch.Tensor, torch.Tensor, list]:
        for row in batch:
            tokens = self._tokenize_row(row)
            if tokens is not None and len(tokens) > 0:
                self.doc_buffer.append(tokens)
        return self._emit(len(batch))

    def _tokenize_row(self, row) -> Optional[torch.Tensor]:
        if isinstance(row, torch.Tensor):
            return row
        
        if isinstance(row, dict):
            data_sig = row.get("data_signature", {"format_fn": "pass_text", "keys": ["text"]})
            format_fn_name = data_sig.get("format_fn", "pass_text")
            
            local_add_bos = data_sig.get("add_bos", self.add_bos)
            local_add_eos = data_sig.get("add_eos", self.add_eos)
            
            if format_fn_name == "pass_text":
                text = row.get("text", "")
                if text:
                    return self.tokenizer.encode(text, bos=local_add_bos, eos=local_add_eos)
            elif format_fn_name == "concat_input_target":
                text = row.get("input", "") + row.get("target", "")
                if text:
                    return self.tokenizer.encode(text, bos=local_add_bos, eos=local_add_eos)
            else:
                text = row.get("text", "")
                if text:
                    return self.tokenizer.encode(text, bos=local_add_bos, eos=local_add_eos)
        
        return None


class PackedIterableDataset(IterableDataset):
    """Pull documents on demand so the packing buffer stays full."""

    def __init__(self, dataset, collator: BestFitPackingCollator, n_rows: int):
        super().__init__()
        self.dataset = dataset
        self.collator = collator
        self.n_rows = n_rows

    def __iter__(self):
        collator = self.collator
        if get_worker_info() is not None:
            collator = copy.copy(self.collator)
            collator.doc_buffer = []
        doc_iter = iter(self.dataset)
        while True:
            collator._refill(doc_iter)
            if not collator.doc_buffer:
                return
            yield collator._emit(self.n_rows, doc_iter=doc_iter)

    def __getattr__(self, name):
        try:
            dataset = object.__getattribute__(self, "dataset")
        except AttributeError as exc:
            raise AttributeError(name) from exc
        return getattr(dataset, name)

    @property
    def _state(self):
        dataset = object.__getattribute__(self, "dataset")
        if not hasattr(dataset, "_state"):
            raise AttributeError("_state")
        return dataset._state

    @_state.setter
    def _state(self, value):
        object.__getattribute__(self, "dataset")._state = value


def unwrap_singleton_collate(batch):
    return batch[0]


def pass_text(row, tokenizer, add_bos, add_eos):
    input_string = row["text"]
    input_tokens = tokenizer.encode(input_string, bos=add_bos, eos=add_eos)
    label_tokens = input_tokens.clone()
    return (input_tokens, label_tokens)


def concat_input_target(row, tokenizer, add_bos, add_eos):
    input_string = row["input"] + row["target"]
    input_tokens = tokenizer.encode(input_string, bos=add_bos, eos=add_eos)
    label_tokens = input_tokens.clone()
    return (input_tokens, label_tokens)


def condition_input_supervise_target(row, tokenizer, add_bos, add_eos):
    input_string = row["input"]
    joint_string = row["input"] + row["target"]
    input_tokens = tokenizer.encode(input_string, bos=add_bos, eos=False)
    joint_tokens = tokenizer.encode(joint_string, bos=add_bos, eos=add_eos)
    label_tokens = joint_tokens.clone()
    label_tokens[0 : len(input_tokens)] = tokenizer.pad_id
    input_tokens = joint_tokens
    return (input_tokens, label_tokens)


def apply_chat_template_supervise_all(row, tokenizer, add_bos, add_eos):
    assert len(row["data_signature"]["keys"]) == 1, (
        "Ambiguous row format for chat template call. data signature should spec the single intended key."
    )
    key = row["data_signature"]["keys"][0]
    input_string = tokenizer.processor.apply_chat_template(row[key], tokenize=False)
    input_tokens = tokenizer.encode(input_string, bos=add_bos, eos=add_eos)
    label_tokens = input_tokens.clone()
    return (input_tokens, label_tokens)


def apply_chat_template_supervise_assistant(row, tokenizer, add_bos, add_eos):
    tokenizer.processor.chat_template = """{% set loop_messages = messages %}{% for message in loop_messages %}{% set start_content = '<|begin_header|>' %}{% set end_content = message['content'] | trim  + '<|end_turn|>' %}{% if loop.index0 == 0 %}{% set start_content = bos_token + start_content %}{% endif %}{% if message['role'] == 'Huginn' or message['role'] == 'assistant' %}{% set start_content = start_content + 'Huginn<|end_header|>\n\n' %}{{ start_content }}{% generation %}{{ end_content }}{% endgeneration %}{% else %}{% set start_content = start_content + message['role'] + '<|end_header|>\n\n' %}{{ start_content }}{{ end_content }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|begin_header|>Huginn<|end_header|>\n\n' }}{% else %}{{ '<|end_text|>' }}{% endif %}"""

    assert len(row["data_signature"]["keys"]) == 1, (
        "Ambiguous row format for chat template call. data signature should spec the single intended key."
    )
    key = row["data_signature"]["keys"][0]

    assert isinstance(row[key], list), "is not in chat format"
    tokenized_string = tokenizer.processor.apply_chat_template(
        row[key],
        tokenize=True,
        add_generation_prompt=False,
        return_assistant_tokens_mask=True,
        return_dict=True,
        return_tensors="pt",
    )
    labels = torch.tensor(tokenized_string["assistant_masks"]) * tokenized_string["input_ids"]
    labels[labels == 0] = tokenizer.pad_id
    return (tokenized_string["input_ids"], labels)


format_fn_registry = {
    "pass_text": pass_text,
    "concat_input_target": concat_input_target,
    "condition_input_supervise_target": condition_input_supervise_target,
    "apply_chat_template_supervise_all": apply_chat_template_supervise_all,
    "apply_chat_template_supervise_assistant": apply_chat_template_supervise_assistant,
}


def apply_formatting(row, tokenizer, add_bos, add_eos):
    if isinstance(row, torch.Tensor):
        return row, row.clone()
    if isinstance(row, tuple):
        raise NotImplementedError("Tuple format not supported.")
    if isinstance(row, dict):
        if row["data_signature"].get("add_bos") is not None:
            add_bos = row["data_signature"]["add_bos"]
        if row["data_signature"].get("add_eos") is not None:
            add_eos = row["data_signature"]["add_eos"]

        return format_fn_registry[row["data_signature"]["format_fn"]](row, tokenizer, add_bos, add_eos)
    raise ValueError("Row format not recognized.")


def shift_inputs_and_labels(inputs_batch: torch.Tensor, labels_batch: torch.Tensor, tokenizer: Tokenizer):
    seq_len = inputs_batch.shape[1]

    input_ids = inputs_batch[:, 0 : (seq_len - 1)].contiguous().long()
    label_ids = labels_batch[:, 1:(seq_len)].contiguous().long()

    if tokenizer.eos_id is not None:
        input_ids[input_ids == tokenizer.pad_id] = tokenizer.eos_id  # type: ignore
    return input_ids, label_ids


def generic_collate_fn(
    batch,
    tokenizer: Tokenizer,
    block_size: Optional[int] = None,
    pad_to_block_size: bool = False,
    add_bos=True,
    add_eos=True,
    collate_checks_enabled=True,
    all_block_size_tensors=False,
):
    metadata = [None] * len(batch)
    for i, row in enumerate(batch):
        if isinstance(row, dict) and "data_id" in row:
            metadata[i] = row["data_id"]

    if all_block_size_tensors:
        inputs_batch = collate_tensor_fn(batch)
        labels_batch = inputs_batch.clone()
        input_ids, label_ids = shift_inputs_and_labels(inputs_batch, labels_batch, tokenizer)
        return input_ids, label_ids, metadata
    else:
        assert block_size is not None

    if collate_checks_enabled:
        assert isinstance(batch, list), "Batch must be a list."
        type_list = [type(x) for x in batch]
        allowed_types = [dict, torch.Tensor]
        types_found = set(type_list)
        assert types_found.issubset(allowed_types), "Batch must contain only expected types."

        if dict in types_found:
            assert tokenizer is not None, "If batch contains dicts, tokenizer must be provided."
            assert tokenizer.pad_id is not None, "Tokenizer must have pad token id since we are dynamically padding."

    batch = [apply_formatting(row, tokenizer, add_bos, add_eos) for row in batch]

    if pad_to_block_size:
        local_block_size = block_size
    else:
        all_lengths = [len(x) for row in batch for x in row]
        local_block_size = min(max(all_lengths), block_size)

    inputs_batch = torch.full((len(batch), local_block_size), tokenizer.pad_id or 0, dtype=torch.int)  # type: ignore
    labels_batch = torch.full((len(batch), local_block_size), tokenizer.pad_id or 0, dtype=torch.int)  # type: ignore
    for i, (input_tokens, label_tokens) in enumerate(batch):
        inputs_batch[i, : len(input_tokens)] = input_tokens[:local_block_size]
        labels_batch[i, : len(label_tokens)] = label_tokens[:local_block_size]

    all_eos = tokenizer.eos_id is not None and torch.all(labels_batch == tokenizer.eos_id)
    all_pad = tokenizer.pad_id is not None and torch.all(labels_batch == tokenizer.pad_id)
    if all_eos or all_pad:
        raise StopIteration("All tokens in batch are padding tokens.")

    input_ids, label_ids = shift_inputs_and_labels(inputs_batch, labels_batch, tokenizer)
    return input_ids, label_ids, metadata
