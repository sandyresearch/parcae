# Parcae Config
from dataclasses import dataclass, replace, field
from typing import Literal, Optional
import torch

from parcae_lm.models.config import Config
from parcae_lm.models.parcae.init import ParcaeInit


@dataclass
class ParcaeConfig(Config):
    """Configuration for Parcae Models."""
    # Arch
    injection_type: Literal["diagonal", "linear", "add"] = "diagonal"
    n_layers_in_recurrent_block: int = 4
    n_layers_in_prelude: int = 1
    n_layers_in_coda: int = 1
    state_init: str = "like-init"
    recurrent_embedding_dimension: int = 1024
    recurrent_intermediation_embedding_dimension: int = 3520
    recurrent_num_attention_heads: Optional[int] = None  # defaults to num_attention_heads if None
    prelude_norm: bool = False
    # Sampling
    sampling_scheme: str = "poisson-truncated-full"
    mean_recurrence: int = 32
    mean_backprop_depth: int = 8
    lockstep_n: bool = False
    lockstep_k: bool = False
    curriculum_target: Literal["forward", "backward", "both"] = "forward"  # what to schedule in curriculum
    # Gradient flow control
    recurrent_iteration_method: Literal["per-batch", "per-sequence", "per-token"] = "per-batch"
    qk_norm: bool = False
    activation_checkpoint_impl: str = "per-iteration"
    tie_embeddings: bool = False
    # Model class selection
    model_class_name: Literal["Parcae"] = "Parcae"
    # Internal flag to prevent recursion when creating nested config
    _is_recurrent_block_config: bool = field(default=False, repr=False)

    def __post_init__(self):
        super().__post_init__()

        effective_expected_depth = (
            self.n_layers_in_prelude + self.n_layers_in_coda + self.n_layers_in_recurrent_block * self.mean_recurrence
        )
        self.n_layer = self.n_layers_in_recurrent_block * self.mean_backprop_depth
        
        self.init = ParcaeInit(
            self.init_strategy,
            self.n_embd,
            self.intermediate_size,
            self.head_size,
            effective_expected_depth,
            self.mup_model_scaling_factor,
            orthogonal=self.init_orthogonal,
            verbose=False,
            skip_reinitializing=self.skip_initialization,
        )
        
        self._recurrent_block_config = None
        if not self._is_recurrent_block_config:
            self._recurrent_block_config = self._create_recurrent_block_config()

    def _create_recurrent_block_config(self) -> "ParcaeConfig":
        """Create a config for recurrent blocks with proper dimensions and initialization."""
        # Determine number of attention heads for recurrent blocks
        recurrent_num_heads = self.recurrent_num_attention_heads
        if recurrent_num_heads is None:
            recurrent_num_heads = self.num_attention_heads
        
        # Validate divisibility
        assert self.recurrent_embedding_dimension % recurrent_num_heads == 0, (
            f"recurrent_embedding_dimension ({self.recurrent_embedding_dimension}) must be divisible by "
            f"recurrent_num_attention_heads ({recurrent_num_heads})"
        )
        
        recurrent_config = replace(
            self,
            n_embd=self.recurrent_embedding_dimension,
            intermediate_size=self.recurrent_intermediation_embedding_dimension,
            num_attention_heads=recurrent_num_heads,
            num_key_value_heads=recurrent_num_heads,
            _is_recurrent_block_config=True,
        )
        
        if not recurrent_config.is_mla:
            recurrent_config.head_size = recurrent_config.n_embd // recurrent_config.num_attention_heads
            recurrent_config.n_head = recurrent_config.num_attention_heads
            recurrent_config.n_query_groups = recurrent_config.num_key_value_heads
        
        effective_expected_depth = (
            self.n_layers_in_prelude + self.n_layers_in_coda + 
            self.n_layers_in_recurrent_block * self.mean_recurrence
        )
        recurrent_config.init = ParcaeInit(
            self.init_strategy,
            recurrent_config.n_embd,
            recurrent_config.intermediate_size,
            recurrent_config.head_size,
            effective_expected_depth,
            self.mup_model_scaling_factor,
            orthogonal=self.init_orthogonal,
            verbose=False,
            skip_reinitializing=self.skip_initialization,
        )
        
        return recurrent_config

    @property
    def recurrent_block_config(self) -> "ParcaeConfig":
        if self._recurrent_block_config is not None:
            return self._recurrent_block_config
        return self


    def construct_model(self, **kwargs) -> torch.nn.Module:
        from parcae_lm.models.parcae.parcae import Parcae
        return Parcae(self, **kwargs)