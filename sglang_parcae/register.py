from __future__ import annotations

import logging
import runpy

logger = logging.getLogger(__name__)

ARCHITECTURES = ("ParcaeForCausalLM", "ParcaeMLAForCausalLM")


def register() -> None:
    from sglang.srt.models.registry import ModelRegistry

    from sglang_parcae import modeling_parcae
    from sglang_parcae.configuration_parcae import register_config

    register_config()

    entries = modeling_parcae.EntryClass
    if not isinstance(entries, (list, tuple)):
        entries = [entries]
    for cls in entries:
        existing = ModelRegistry.models.get(cls.__name__)
        if existing is cls:
            continue
        if existing is not None:
            raise RuntimeError(
                f"{cls.__name__} is already registered to a different class ({existing!r}). "
                f"Refusing to overwrite -- resolve the collision explicitly."
            )
        ModelRegistry.models[cls.__name__] = cls
    logger.info("registered Parcae architectures: %s", ", ".join(ARCHITECTURES))


def verify(architecture: str = "ParcaeForCausalLM"):
    from sglang.srt.models.registry import ModelRegistry

    from sglang_parcae import modeling_parcae

    resolved, matched = ModelRegistry.resolve_model_cls(architecture)
    expected = {c.__name__: c for c in
                (modeling_parcae.EntryClass if isinstance(modeling_parcae.EntryClass, (list, tuple))
                 else [modeling_parcae.EntryClass])}
    if matched != architecture or resolved is not expected.get(architecture):
        raise RuntimeError(
            f"SGLang resolved {architecture!r} to {resolved.__module__}.{resolved.__name__} "
            f"(matched arch {matched!r}), not our model class. If that is "
            f"'TransformersForCausalLM', SGLang fell back silently; "
            f"call sglang_parcae.register.register() before the model is constructed."
        )
    return resolved


def main() -> None:
    register()
    verify()
    runpy.run_module("sglang.launch_server", run_name="__main__")


if __name__ == "__main__":
    main()
