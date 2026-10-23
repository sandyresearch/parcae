"""HF config class for Parcae, so AutoConfig can read the shim directory.
"""

from __future__ import annotations

from transformers import PretrainedConfig


class ParcaeConfig(PretrainedConfig):
    model_type = "parcae"

    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_attention_heads: int = 6,
        num_key_value_heads: int = 6,
        num_hidden_layers: int = 6,
        vocab_size: int = 32768,
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 50000.0,
        tie_word_embeddings: bool = True,
        **kwargs,
    ) -> None:
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


def register_config() -> None:
    """Register ParcaeConfig with AutoConfig, so that SGLang can read it from the shim directory.
    """
    from transformers import AutoConfig

    try:
        AutoConfig.register(ParcaeConfig.model_type, ParcaeConfig)
    except ValueError:
        pass  # already registered


register_config()
