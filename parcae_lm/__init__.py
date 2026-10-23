def create_config(name: str, **kwargs):
    """Create a model config by name. Returns a GPTConfig or ParcaeConfig.

    >>> config = create_config("parcae-small-140m")
    >>> config.block_size = 2048
    >>> model = config.construct_model()
    """
    from parcae_lm.models.config import Config
    return Config.from_name(name, **kwargs)


def create_model(name: str, **kwargs):
    """Create a model by name, ready for forward passes.

    >>> model = create_model("parcae-small-140m")
    >>> model = create_model("gpt-small-140m")
    """
    config = create_config(name, **kwargs)
    return config.construct_model()


def from_pretrained(repo_id: str, device="cpu", dtype=None, **kwargs):
    """Load a pretrained model from a HuggingFace repository or a local directory.

    Expects the repo to contain:
      - config.json: serialized model config (with _class_name field)
      - pytorch_model.bin or model.pt: PyTorch state dict

    >>> model = parcae_lm.from_pretrained("SandyResearch/parcae-140m")
    >>> model = parcae_lm.from_pretrained("SandyResearch/parcae-140m", device="cuda")
    >>> model = parcae_lm.from_pretrained("SandyResearch/benice-140m")
    >>> model = parcae_lm.from_pretrained("./my-model")  # a directory written by save_pretrained
    """
    import json
    import os
    import torch
    from huggingface_hub import hf_hub_download

    def fetch(filename):
        if os.path.isdir(repo_id):
            path = os.path.join(repo_id, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            return path
        return hf_hub_download(repo_id, filename, **kwargs)

    config_path = fetch("config.json")
    with open(config_path) as f:
        config_dict = json.load(f)

    class_name = config_dict.pop("_class_name", "GPTConfig")

    # Lazy import to avoid circular deps — Config.from_name handles dispatch internally
    if class_name == "ParcaeConfig":
        from parcae_lm.models.parcae.config import ParcaeConfig as config_cls
    else:
        from parcae_lm.models.gpt.config import GPTConfig as config_cls

    config = config_cls(**config_dict)

    model = config.construct_model()

    for weight_file in ["pytorch_model.bin", "model.pt"]:
        try:
            weights_path = fetch(weight_file)
            break
        except Exception:
            continue
    else:
        raise FileNotFoundError(f"No weights found in {repo_id} (tried pytorch_model.bin, model.pt)")

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)

    if dtype is not None:
        model = model.to(dtype=dtype)
    model = model.to(device=device)
    model.eval()

    return model


def save_pretrained(model, path: str):
    """Save a model's config and weights for later loading with from_pretrained.

    >>> parcae_lm.save_pretrained(model, "./my-model")
    # Then upload the directory to HuggingFace
    """
    import json
    import torch
    from pathlib import Path
    from dataclasses import asdict

    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_dict = asdict(model.config)
    config_dict["_class_name"] = model.config.__class__.__name__
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    torch.save(model.state_dict(), out_dir / "model.pt")


def __getattr__(name):
    if name == "GPT":
        from parcae_lm.models.gpt.gpt import GPT
        return GPT
    if name == "GPTConfig":
        from parcae_lm.models.gpt.config import GPTConfig
        return GPTConfig
    if name == "Parcae":
        from parcae_lm.models.parcae.parcae import Parcae
        return Parcae
    if name == "ParcaeConfig":
        from parcae_lm.models.parcae.config import ParcaeConfig
        return ParcaeConfig
    raise AttributeError(f"module 'parcae_lm' has no attribute {name!r}")
