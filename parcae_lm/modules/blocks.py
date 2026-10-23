from typing import Optional

import torch
from torch import Tensor

from parcae_lm.models.config import AnyConfig


class TransformerPreNormBlock(torch.nn.Module):
    expanded = False

    def __init__(self, config: AnyConfig, layer_id: int) -> None:
        super().__init__()
        self.config = config
        self.norm_1 = config.Norm(config.n_embd, eps=config.norm_eps)
        self.attn = config.Attention(config, layer_id=layer_id)
        self.norm_2 = config.Norm(config.n_embd, eps=config.norm_eps)
        self.mlp = config.MLP(config, layer_id=layer_id)
        self.layer_id = layer_id

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Optional[Tensor] = None, **kwargs) -> Tensor:
        x = self.attn(self.norm_1(x), freqs_cis, mask, **kwargs) + x
        x = self.mlp(self.norm_2(x)) + x
        return x

    def reset_parameters(self) -> None:
        self.config.init.apply(self.norm_1, "normalization")
        self.config.init.apply(self.norm_2, "normalization")