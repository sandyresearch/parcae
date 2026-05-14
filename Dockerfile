FROM nvcr.io/nvidia/pytorch:26.02-py3

WORKDIR /resource
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    "numpy<2.0" einops "lightning==2.3.0.dev20240328" \
    "jsonargparse[signatures]>=4.27.6" "torchmetrics>=1.3.1" "lm-eval>=0.4.2" \
    wandb "sentencepiece>=0.2.0" "tokenizers>=0.15.2" "safetensors>=0.4.3" \
    "datasets>=2.18.0" "transformers>=4.38.0" pandas plotly packaging ninja torchdata \
    "git+https://github.com/axonn-ai/axonn@c5ef65d3d4f329c20292a2d26681e8d9cbcccecf"
