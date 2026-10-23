import os
from pathlib import Path
from typing import Optional, Union, List
import torch


class Tokenizer:
    def __init__(
        self,
        checkpoint_dir: Optional[Union[Path, str]] = None,
        processor=None,
        bos_id=None,
        eos_id=None,
        pad_id=None,
    ):
        if checkpoint_dir is not None:
            instance = self.from_checkpoint(checkpoint_dir)
            self.processor = instance.processor
            self.bos_id = instance.bos_id
            self.eos_id = instance.eos_id
            self.pad_id = instance.pad_id
            self.checkpoint_dir = Path(checkpoint_dir)
        else:
            self.processor = processor
            self.bos_id = bos_id
            self.eos_id = eos_id
            self.pad_id = pad_id
            self.checkpoint_dir = None

        self.cache_token_id = None
        self.eod_token_id = None

    @classmethod
    def from_pretrained(cls, model_name: str):
        from transformers import AutoTokenizer

        processor = AutoTokenizer.from_pretrained(
            model_name, add_bos_token=False, add_eos_token=False
        )
        instance = cls(
            processor=processor,
            bos_id=processor.bos_token_id,
            eos_id=processor.eos_token_id,
            pad_id=processor.pad_token_id,
        )
        return instance

    @classmethod
    def from_directory(cls, tokenizer_dir: Union[Path, str]):
        from transformers import AutoTokenizer, PreTrainedTokenizerFast

        tokenizer_dir = Path(tokenizer_dir)
        tokenizer_path = tokenizer_dir / "tokenizer.json"

        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

        try:
            processor = AutoTokenizer.from_pretrained(
                str(tokenizer_dir), add_bos_token=False, add_eos_token=False
            )
        except Exception:
            processor = PreTrainedTokenizerFast(
                tokenizer_file=str(tokenizer_path),
                add_bos_token=False,
                add_eos_token=False,
            )

        instance = cls(
            processor=processor,
            bos_id=processor.bos_token_id,
            eos_id=processor.eos_token_id,
            pad_id=processor.pad_token_id,
        )
        instance.checkpoint_dir = tokenizer_dir
        return instance

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: Union[Path, str]):
        from huggingface_hub import hf_hub_download

        checkpoint_dir = Path(checkpoint_dir)

        if not checkpoint_dir.exists():
            checkpoint_str = str(checkpoint_dir)
            if checkpoint_str.startswith("/") or checkpoint_str.startswith("."):
                raise NotADirectoryError(f"Tokenizer directory not found: {checkpoint_str}")
            try:
                tokenizer_file = hf_hub_download(
                    repo_id=checkpoint_str, filename="tokenizer.json"
                )
                for optional_file in ["special_tokens_map.json", "tokenizer_config.json"]:
                    try:
                        hf_hub_download(repo_id=checkpoint_str, filename=optional_file)
                    except Exception:
                        pass
                checkpoint_dir = Path(tokenizer_file).parent
            except (OSError, AssertionError):
                raise NotADirectoryError(
                    f"Tokenizer checkpoint {checkpoint_str} cannot be loaded (no tokenizer.json). "
                    "The parcae-* models use SandyResearch/parcae-tokenizer; "
                    "ClimbMix-trained models use SandyResearch/parcae-climbmix-tokenizer."
                )

        return cls.from_directory(checkpoint_dir)

    @classmethod
    def from_pretrained(cls, repo_id: str):
        return cls.from_checkpoint(repo_id)

    def get_vocab_size(self) -> int:
        return self.processor.vocab_size

    @property
    def vocab_size(self) -> int:
        return self.processor.vocab_size

    def __len__(self) -> int:
        return len(self.processor)

    def get_bos_token_id(self) -> int:
        if self.bos_id is None:
            raise ValueError("This tokenizer does not have a BOS token defined")
        return self.bos_id

    def get_eos_token_id(self) -> int:
        if self.eos_id is None:
            raise ValueError("This tokenizer does not have an EOS token defined")
        return self.eos_id

    def encode(
        self,
        text: Union[str, List[str]],
        device: Optional[torch.device] = None,
        bos: Optional[bool] = None,
        eos: bool = False,
        prepend: Optional[Union[str, int]] = None,
        append: Optional[Union[str, int]] = None,
        max_length: int = -1,
        return_tensors: bool = True,
    ) -> Union[torch.Tensor, List[int], List[List[int]]]:

        # Handle legacy bos/eos API
        if bos:
            if self.bos_id is None:
                raise NotImplementedError("This tokenizer does not have a BOS token defined")
            prepend = self.bos_id
        if eos:
            append = self.eos_id

        if isinstance(text, str):
            ids = self._encode_one(text, prepend=prepend, append=append)
            if max_length > 0:
                ids = ids[:max_length]
            if return_tensors:
                return torch.tensor(ids, dtype=torch.int, device=device)
            return ids
        elif isinstance(text, list):
            all_ids = [self._encode_one(t, prepend=prepend, append=append) for t in text]
            if max_length > 0:
                all_ids = [ids[:max_length] for ids in all_ids]
            if return_tensors:
                return [torch.tensor(ids, dtype=torch.int, device=device) for ids in all_ids]
            return all_ids
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def _encode_one(
        self,
        text: str,
        prepend: Optional[Union[str, int]] = None,
        append: Optional[Union[str, int]] = None,
    ) -> List[int]:
        ids = []

        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids.append(prepend_id)

        ids.extend(self.processor.encode(text, add_special_tokens=False))

        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            ids.append(append_id)

        return ids

    def encode_special(self, token: str) -> int:
        token_id = self.processor.convert_tokens_to_ids(token)
        if token_id == self.processor.unk_token_id:
            raise ValueError(f"Special token '{token}' not found in vocabulary")
        return token_id

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, tensor: Union[torch.Tensor, List[int]], skip_special_tokens: bool = False) -> str:
        if isinstance(tensor, torch.Tensor):
            tokens = [tensor.item()] if tensor.ndim == 0 else tensor.tolist()
        else:
            tokens = tensor
        return self.processor.decode(tokens, skip_special_tokens=skip_special_tokens)

    def id_to_token(self, token_id: int) -> str:
        return self.processor.convert_ids_to_tokens(token_id)

    def save(self, tokenizer_dir: Union[Path, str]):
        tokenizer_dir = Path(tokenizer_dir)
        os.makedirs(tokenizer_dir, exist_ok=True)
        self.processor.save_pretrained(str(tokenizer_dir))
        print(f"Saved tokenizer to {tokenizer_dir}")

    def state_dict(self):
        return {"dir": self.checkpoint_dir}

    def load_state_dict(self, state_dict):
        try:
            return Tokenizer(state_dict["dir"])
        except Exception as e:
            print(f"Warning: Could not reload tokenizer due to error {e}. Using current tokenizer.")
            return self

    def __reduce__(self):
        return (self.__class__, (self.checkpoint_dir,))

    def render_conversation(self, conversation, max_tokens=2048):
        import copy
        ids, mask = [], []

        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        # merge system message with first user message
        if conversation["messages"][0]["role"] == "system":
            conversation = copy.deepcopy(conversation)
            messages = conversation["messages"]
            assert messages[1]["role"] == "user"
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]
        else:
            messages = conversation["messages"]
        assert len(messages) >= 1

        bos = self.bos_id if self.bos_id is not None else 1

        def get_special_token(name):
            try:
                return self.encode_special(name)
            except (ValueError, KeyError):
                enc = self.encode(name, return_tensors=False)
                return enc[0] if enc else bos

        user_start = get_special_token("<|user|>")
        user_end = get_special_token("<|/user|>")
        assistant_start = get_special_token("<|assistant|>")
        assistant_end = get_special_token("<|/assistant|>")

        add_tokens(bos, 0)
        for i, message in enumerate(messages):
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == must_be_from
            content = message["content"]

            if message["role"] == "user":
                assert isinstance(content, str)
                add_tokens(user_start, 0)
                add_tokens(self.encode(content, return_tensors=False), 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    add_tokens(self.encode(content, return_tensors=False), 1)
                elif isinstance(content, list):
                    for part in content:
                        add_tokens(self.encode(part["text"], return_tensors=False), 1)
                else:
                    raise ValueError(f"Unknown content type: {type(content)}")
                add_tokens(assistant_end, 1)

        return ids[:max_tokens], mask[:max_tokens]
