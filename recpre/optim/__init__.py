# type: ignore
import torch
from functools import partial

from .simple_zero_redundancy import SimpleZeroRedundancyOptimizer
from .lionw import LionW
from .sophiag import SophiaG
from .lilith import Lilith
from .ellis_adam import ELLISAdam
from .ivon import IVON
from .zero_shampoo import ZeroShampooWithAdamGraftingOptimizer
from .orthogonal_nesterov import OrthogonalNesterov
from .soap import SOAP
from .muon_adamw import MuonAdamW, get_muon_param_groups


def compute_scaled_muon_wd(optim_config, d_model):
    """nanochat-style 1/channels² Muon WD scaling. Reference d_model is 768 (parcae-small).
    Set optim_config['muon_wd_dmodel_scale']=false to disable."""
    base = optim_config.get('muon_wd', 0.0)
    if not optim_config.get('muon_wd_dmodel_scale', True):
        return base
    d_ref = optim_config.get('muon_wd_dmodel_ref', 768)
    return base * (d_ref / d_model) ** 2


def get_muon_param_groups_from_config(named_parameters, optim_config, no_weight_decay_for_bias_and_norm_params=True, verbose=False):
    """Build MuonAdamW param groups from standard optim_config."""
    return get_muon_param_groups(
        named_parameters,
        adamw_lr=optim_config.get('lr', 3e-4),
        adamw_betas=optim_config.get('betas', (0.9, 0.95)),
        adamw_eps=optim_config.get('eps', 1e-8),
        adamw_wd=optim_config.get('weight_decay', 0.1) if no_weight_decay_for_bias_and_norm_params else 0.0,
        muon_lr=optim_config.get('muon_lr', 0.02),
        muon_momentum=optim_config.get('muon_momentum', 0.95),
        muon_wd=optim_config.get('muon_wd', 0.2),
        muon_ns_steps=optim_config.get('muon_ns_steps', 5),
        verbose=verbose,
    )


def get_param_groups(
    named_parameters,
    no_weight_decay_for_bias_and_norm_params=True,
    weight_lr_scale=1.0,
    no_wd_on_embedding=False,  # for tied models this needs to be false to have wd on the logit layer
    verbose=False,
):
    param_groups = []

    weights_group = []  # default group
    weights_no_wd_group = []  # weights with _no_weight_decay attribute
    embedding_group = []
    embedding_no_wd_group = []  # embeddings with _no_weight_decay attribute
    scale_and_norm_group = []
    # Memory value parameters grouped by their fixed learning rate
    memory_value_groups = {}  # Dict mapping fixed_lr -> list of params
    # readout_group = [] # unused
    
    for name, param in named_parameters:
        # Check if this parameter has _no_weight_decay attribute
        has_no_wd = getattr(param, "_no_weight_decay", False)
        
        # Check if this is a memory value parameter with fixed learning rate
        if hasattr(param, "pk_value_param") and param.pk_value_param and hasattr(param, "fixed_lr"):
            fixed_lr = param.fixed_lr
            if fixed_lr is not None:
                if fixed_lr not in memory_value_groups:
                    memory_value_groups[fixed_lr] = []
                memory_value_groups[fixed_lr].append(param)
                if verbose:
                    print(f"Matched {name} to memory value group with fixed_lr={fixed_lr}")
                continue  # Skip adding to other groups
        
        # Strip FSDP/compile wrapper prefixes for matching
        clean_name = name
        for prefix in ["_orig_mod.", "_forward_module.", "_fsdp_wrapped_module."]:
            clean_name = clean_name.replace(prefix, "")
        lname = clean_name.lower()
        if "wte" in lname or "embedding" in lname or "abacus" in lname or "lm_head" in lname:
            if has_no_wd:
                embedding_no_wd_group.append(param)
                if verbose:
                    print(f"Matched {name} to embedding group (no weight decay via attribute)")
            else:
                embedding_group.append(param)
                if verbose:
                    print(f"Matched {name} to embedding group")
        elif "ln_f" in lname or "norm" in lname or "bias" in lname or "diff_lmb" in name or "euler" in lname or "anchor" in lname or "momentum" in lname or "anderson" in lname:
            scale_and_norm_group.append(param)
            if verbose:
                print(f"Matched {name} to scale and norm_group")
        elif "proj" in lname or "qkv" in lname or "fc" in lname or "adapter" in lname or "mlp" in lname or param.ndim == 2:
            if has_no_wd:
                weights_no_wd_group.append(param)
                if verbose:
                    print(f"Matched {name} to main weight group (no weight decay via attribute)")
            else:
                weights_group.append(param)
                if verbose:
                    print(f"Matched {name} to main weight group")
        elif has_no_wd:
            # Parameters with _no_weight_decay that don't match other patterns
            weights_no_wd_group.append(param)
            if verbose:
                print(f"Matched {name} to no-weight-decay group (via attribute)")
        else:
            raise ValueError(f"param {name} could not be matched to an optim group in recpre/optim.py")

    param_groups.append({"params": weights_group, "base_lr": weight_lr_scale})
    if weights_no_wd_group:
        param_groups.append({"params": weights_no_wd_group, "base_lr": weight_lr_scale, "weight_decay": 0.0})
    param_groups.append({"params": embedding_group, "base_lr": 1.0})
    if no_wd_on_embedding:
        param_groups[-1]["weight_decay"] = 0.0
    if embedding_no_wd_group:
        param_groups.append({"params": embedding_no_wd_group, "base_lr": 1.0, "weight_decay": 0.0})
    param_groups.append({"params": scale_and_norm_group, "base_lr": 1.0})
    if no_weight_decay_for_bias_and_norm_params:
        param_groups[-1]["weight_decay"] = 0.0
    
    for fixed_lr, params in memory_value_groups.items():
        param_groups.append({
            "params": params, 
            "lr": fixed_lr, 
            "base_lr": fixed_lr,
            "fixed_lr": True  # Flag to indicate this group has a fixed learning rate
        })
        if verbose:
            print(f"Created memory value group with {len(params)} params and fixed_lr={fixed_lr}")

    return param_groups


def get_optimizer(
    optimizer_name,
    model=None,
    pytorch_optimizer_sharding: bool = False,
    allow_fusion: bool = True,
    use_apex_adamw: bool = False,
):
    if hasattr(torch.optim, optimizer_name):
        optim_class = getattr(torch.optim, optimizer_name)  # read all torch optimizers
    elif optimizer_name == "LionW":
        optim_class = LionW
    elif optimizer_name == "SophiaG":
        optim_class = SophiaG
    elif optimizer_name == "Lilith":
        optim_class = Lilith
    elif optimizer_name == "ELLISAdam":
        optim_class = ELLISAdam
    elif optimizer_name == "IVON":
        optim_class = IVON
    elif optimizer_name == "simo-shampoo":
        optim_class = ZeroShampooWithAdamGraftingOptimizer
    elif optimizer_name == "meta-shampoo":
        try:
            from distributed_shampoo.distributed_shampoo import DistributedShampoo
            from distributed_shampoo.shampoo_types import AdamGraftingConfig, CommunicationDType, DDPShampooConfig
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Run `pip install git+https://github.com/JonasGeiping/meta-shampoo` first!")

        optim_class = partial(
            DistributedShampoo,
            grafting_config=AdamGraftingConfig(
                beta2=0.95,
                epsilon=1e-8,
            ),
            distributed_config=DDPShampooConfig(
                communication_dtype=CommunicationDType.FP32,
                num_trainers_per_group=torch.cuda.device_count(),
                communicate_params=False,
            ),
        )
    elif optimizer_name == "SOAP":
        optim_class = SOAP
    elif optimizer_name == "Kellers":
        optim_class = OrthogonalNesterov
    elif optimizer_name == "MuonAdamW":
        return lambda param_groups, **kwargs: MuonAdamW(param_groups)
    else:
        raise ValueError(f"Invalid optimizer {optimizer_name} requested.")

    if optimizer_name == "AdamW" and use_apex_adamw:
        try:
            from apex.optimizers import FusedAdam
        except ModuleNotFoundError:
            raise ValueError("Need to install apex!")

        optim_class = FusedAdam
        print("Using apex.optimizers.FusedAdam")

    if allow_fusion:
        import inspect

        if "fused" in inspect.signature(optim_class).parameters:
            # llm.c trick to fish for fused implementations
            optim_class = partial(optim_class, fused=True)

    if pytorch_optimizer_sharding and torch.distributed.is_initialized():
        # from torch.distributed.optim import ZeroRedundancyOptimizer

        # return partial(ZeroRedundancyOptimizer, optimizer_class=optim_class, overlap_with_ddp=False)
        return partial(SimpleZeroRedundancyOptimizer, optimizer_class=optim_class)
    else:
        return optim_class


# Export all optimizer classes for direct access if needed
__all__ = [
    "get_param_groups",
    "get_optimizer",
    "get_muon_param_groups",
    "SimpleZeroRedundancyOptimizer",
    "LionW",
    "SophiaG",
    "Lilith",
    "ELLISAdam",
    "IVON",
    "ZeroShampooWithAdamGraftingOptimizer",
    "OrthogonalNesterov",
    "SOAP",
    "MuonAdamW",
]


