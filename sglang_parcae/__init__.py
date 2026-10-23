"""Parcae for SGLang.

Importing this package registers `parcae` with transformers' AutoConfig and configures SGLang's
worker subprocesses as side effects. This must happen before any ServerArgs / Engine / ModelConfig
is constructed, because those resolve the model directory through AutoConfig.from_pretrained,
which raises on an unknown model_type.

The editable install makes this package importable without PYTHONPATH. Setting
SGLANG_EXTERNAL_MODEL_PACKAGE here covers SGLang's subprocesses; drivers still need to import this
package before constructing an Engine or ServerArgs so the parent process registers AutoConfig.
"""

import os

os.environ.setdefault("SGLANG_EXTERNAL_MODEL_PACKAGE", "sglang_parcae")

from sglang_parcae import configuration_parcae as _configuration_parcae  # noqa: F401  (registers)
