<h1 align="center" style="fontsize:50em"><b>Parcae</b></h1>

![Parcae](assets/parcae.png "Parcae Looped Architecture")
> **Parcae: Scaling Laws For Stable Looped Language Models**\
> Hayden Prairie, Zachary Novack, Taylor Berg-Kirkpatrick, Daniel Y. Fu\
> Paper: https://arxiv.org/abs/2604.12946


![BeNice!](assets/benice.png "KV-Sharing Across Recurrences")
> **BeNice! Closing Looping's Inference-Memory Gap via KV-Sharing**\
> Hayden Prairie<sup>&ast;</sup>, Jacob M. Ryan<sup>&ast;</sup>, Taylor Berg-Kirkpatrick, Daniel Y. Fu\
> Paper:  https://arxiv.org/abs/2610.31633v1


# About this Codebase

Parcae is a new looped architecture that utilizes a handful of techniques to stabilize training. Parcae enables stable, hassle-free training of looped models, which we use to derive the first scaling laws for looping, finding that compute-optimal training scales looping and data in tandem.

BeNice! is a post-training technique that allows looped architectures to share KV-caches across recurrence with no quality degradations, a property we call *KV-shareability*. BeNice! allows looped architectures to maintain the same memory footprint as their unlooped counterparts.

# Installation

Just wanna use off-the-shelf models? We make things easy with a PyPI package to access models. Install the package with the following:
```bash
pip install parcae-lm
```
If you are training models, then please clone the GitHub repository 
```bash
git clone https://github.com/SandyResearch/parcae.git
cd parcae
```
and then follow the following:

### Docker (recommended)

Our launch scripts handle everything automatically. Set `PROJECT_DIR` and `DOCKER_IMAGE` at the top of `launch_job.slurm` or `launch_interactive.sh`, then:

```bash
# Interactive development shell
bash launch_interactive.sh

# Submit a training job
CONFIG=launch_configs/parcae/pretrain/parcae-small-140m.yaml sbatch launch_job.slurm
```

The Docker image is hosted publicly at `ghcr.io/sandyresearch/parcae` and will be pulled automatically.

### Local

Requires Python 3.11+ and PyTorch 2.4+. Install PyTorch first, following [pytorch.org](https://pytorch.org/get-started/locally/), then:

```bash
pip install -e .
```

# Usage

## Models

We provide three ways to instantiate models: load pretrained weights with `from_pretrained`, build from a built-in config with `create_model`, or customize a config before building with `create_config`.

```python
import parcae_lm

# Load a pretrained model from HuggingFace
model = parcae_lm.from_pretrained("SandyResearch/parcae-140m")

# Create a model from a built-in config
model = parcae_lm.create_model("parcae-small-140m")

# Or get the config, customize it, then build
config = parcae_lm.create_config("parcae-small-140m")
config.mean_recurrence = 16
model = config.construct_model()
```

## Training

### Downloading Data

```bash
python scripts/download_data.py fineweb-100bt   # FineWeb-Edu 100B tokens
python scripts/download_data.py fineweb-350bt   # FineWeb-Edu 350B tokens
python scripts/download_data.py huginn          # Huginn dataset
python scripts/download_data.py climbmix-100bt  # ClimbMix 100B tokens
python scripts/download_data.py climbmix-400bt  # ClimbMix 400B tokens
```

### Training a Tokenizer

Train a GPT-4 style BPE tokenizer on your data:

```bash
python scripts/tok_train.py --data-dir fineweb --output-dir tokenizer/ --vocab-size 32768
```

Evaluate compression ratios against GPT-2 and GPT-4 tokenizers:

```bash
python scripts/tok_eval.py --tokenizer tokenizer/parcae_tokenizer --data-dir fineweb
```

### Launching Training

All launch_configs are listed in `launch_configs/README.md`.
**Single node:**

```bash
bash runs/run_training.sh launch_configs/parcae/pretrain/parcae-small-140m.yaml parcae-small-140m parcae 8
```
or
```bash
bash runs/run_finetuning.sh launch_configs/parcae/finetune/benice-770m.yaml benice-770m benice 8
```

**Multi-node (Slurm):**

```bash
CONFIG=launch_configs/parcae/pretrain/parcae-large-770m.yaml sbatch launch_job.slurm
```
or

```bash
CONFIG=launch_configs/benice/pretrain/parcae-xlarge-1_3b.yaml sbatch launch_job.slurm
```

## Eval

Evaluate models using `scripts/eval.py`. Supports loading from HuggingFace or local checkpoints.

```bash
# Evaluate a pretrained model from HuggingFace
python scripts/eval.py --hf_repo SandyResearch/parcae-140m --eval_tasks core

# Evaluate a local checkpoint
bash runs/run_eval.sh outputs/parcae-small-140m eval_configs/eval-core.yaml 8

# Evaluate validation loss
python scripts/eval.py --hf_repo SandyResearch/parcae-140m --eval_tasks bpb \
    --tasks.bpb.val_data_dir /path/to/val/data
```

**Tokenizer.** `scripts/eval.py --hf_repo` and `scripts/finetune.py` read the tokenizer from the
`PARCAE_TOKENIZER` environment variable, which defaults to `SandyResearch/parcae-tokenizer`
(the FineWeb-Edu tokenizer used by the `parcae-*` models). The BeNice models, and every model
trained on ClimbMix, use a different vocabulary, so set it before evaluating or finetuning them:

```bash
export PARCAE_TOKENIZER=SandyResearch/parcae-climbmix-tokenizer
python scripts/eval.py --hf_repo SandyResearch/benice-140m --config eval_configs/eval-val-loss-compress-s1.yaml \
    --tasks.kv_cache_val.val_data_dir /resource/data/climbmix-shuffle/val
```

Available eval configs in `eval_configs/`:
- `eval-core.yaml` — Core benchmark suite
- `eval-core-extended.yaml` — Extended core benchmarks
- `eval-core-extended-compress-s1.yaml` - Extended core benchmarks under KV-sharing
- `eval-val-loss.yaml` — Validation loss / bits-per-byte
- `eval-lambada.yaml` — LAMBADA evaluation
- `eval-val-loss-compress-s1.yaml` - Validation loss / bits-per-byte under KV-sharing

# Pretrained Models

Pretrained models are uploaded to [Hugging Face](https://huggingface.co/SandyResearch): `parcae-140m`, `parcae-370m`, `parcae-770m`, `parcae-1.3b`, trained on the [FineWeb-Edu](https://huggingface.co/datasets/SandyResearch/fineweb-edu-shuffled) dataset. Models will be auto-downloaded when using `from_pretrained`.

These models dimensions are:

| Model | Parameters | Prelude | Core | Coda | Model dim. | Recurrence |
|-------|-----------|---------|------|------|-----------|------------|
| Parcae-140M | 140M | 2 | 2 | 2 | 768 | 8 |
| Parcae-370M | 370M | 4 | 4 | 4 | 1024 | 8 |
| Parcae-770M | 770M | 6 | 6 | 6 | 1280 | 8 |
| Parcae-1.3B | 1.3B | 8 | 8 | 8 | 1536 | 8 |

Note: these are base models without any form of downstream modification (instruction tuning, etc.). 

Finetuned models are uploaded to [Hugging Face](https://huggingface.co/SandyResearch): `benice-140m`, `benice-370m`, `benice-770m`, `benice-1.3b`, pretrained and finetuned on the [ClimbMix](https://huggingface.co/datasets/SandyResearch/climbmix-shuffle) dataset. Models will be auto-downloaded when using `from_pretrained`.

| Model | Parameters | Prelude | Core | Coda | Model dim. | Recurrence |
|-------|-----------|---------|------|------|-----------|------------|
| BeNice-140M | 140M | 2 | 2 | 2 | 768 | 8 |
| BeNice-370M | 370M | 4 | 4 | 4 | 1024 | 8 |
| BeNice-770M | 770M | 6 | 6 | 6 | 1280 | 8 |
| BeNice-1.3B | 1.3B | 8 | 8 | 8 | 1536 | 8 |

Note: these are models only received BeNice! finetuning and no other downstream modification (instruction tuning, etc.). 

# Replicating Scaling Laws

The sweep scripts in `runs/` reproduce the scaling law experiments from the paper. See `runs/sweep_recurrence.sh` for recurrence scaling and `runs/sweep_flops.sh` for compute-optimal scaling.

# Citations

```bibtex
@misc{prairie2026parcaescalinglawsstable,
      title={Parcae: Scaling Laws For Stable Looped Language Models}, 
      author={Hayden Prairie and Zachary Novack and Taylor Berg-Kirkpatrick and Daniel Y. Fu},
      year={2026},
      eprint={2604.12946},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.12946}, 
}
```

```bibtex
@misc{prairie2026beniceclosingloopingsinferencememory,
      title={BeNice! Closing Looping's Inference-Memory Gap via KV-Sharing}, 
      author={Hayden Prairie and Jacob M. Ryan and Taylor Berg-Kirkpatrick and Daniel Y. Fu},
      year={2026},
      eprint={2610.31633},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2610.31633}, 
}
```

# References

This code-base was built on `karpathy/nanochat`, `seal-rg/recurrent-pretraining`, and `Lightning-AI/litgpt`. While most code has been thoroughly adapted, we greatly appreciate the work that went into developing each of these training libraries.
