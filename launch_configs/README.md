### Training and Finetuning Launch Configs
Training is configured via YAML files in `launch_configs/`. We have two subdirectories: `parcae/` and `benice/`. `parcae/` contains the pretraining and BeNice! finetuning YAML files for the original Parcae models released in **Parcae: Scaling Laws For Stable Looped Language Models**. `benice/` contains the pretraining and BeNice! finetuning YAML files for the new Parcae models shown in **BeNice! Closing Looping's Inference-Memory Gap via KV-Sharing**, including the versions with memory efficient sequence mixers (multi-latent attention (MLA) and Kimi delta attention (KDA)).

Available pretraining configs:

| Config | Architecture | Parameters |
|--------|-------------|------------|
| `parcae/pretrain/parcae-small-140m.yaml` | Parcae (MHA) | 140M |
| `parcae/pretrain/parcae-medium-370m.yaml` | Parcae (MHA) | 370M |
| `parcae/pretrain/parcae-large-770m.yaml` | Parcae (MHA) | 770M |
| `parcae/pretrain/parcae-xlarge-1_3b.yaml` | Parcae (MHA) | 1.3B |
| `parcae/pretrain/gpt-small-140m.yaml` | GPT (MHA) | 140M |
| `parcae/pretrain/gpt-medium-370m.yaml` | GPT (MHA) | 370M |
| `parcae/pretrain/gpt-large-770m.yaml` | GPT (MHA) | 770M |
| `parcae/pretrain/gpt-xlarge-1_3b.yaml` | GPT (MHA) | 1.3B |
| `benice/pretrain/parcae-140m-kda.yaml` | Parcae (KDA) | 140M |
| `benice/pretrain/parcae-370m-kda.yaml` | Parcae (KDA) | 370M |
| `benice/pretrain/parcae-770m-kda.yaml` | Parcae (KDA) | 770M |
| `benice/pretrain/parcae-140m-mla.yaml` | Parcae (MLA) | 140M |
| `benice/pretrain/parcae-370m-mla.yaml` | Parcae (MLA) | 370M |
| `benice/pretrain/parcae-770m-mla.yaml` | Parcae (MLA) | 770M |
| `benice/pretrain/parcae-small-140m.yaml` | Parcae (MHA) | 140M |
| `benice/pretrain/parcae-medium-370m.yaml` | Parcae (MHA) | 370M |
| `benice/pretrain/parcae-large-770m.yaml` | Parcae (MHA) | 770M |
| `benice/pretrain/parcae-xlarge-1_3b.yaml` | Parcae (MHA) | 1.3B |

Available finetuning configs:
| Config | Architecture | Parameters |
|--------|-------------|------------|
| `parcae/finetune/benice-770m.yaml` | Parcae (MHA) | 770M |
| `parcae/finetune/benice-1.3b.yaml` | Parcae (MHA) | 1.3B |
| `parcae/finetune/standard-ft-770m.yaml` | Parcae (MHA) | 770M |
| `parcae/finetune/standard-ft-1.3b.yaml` | Parcae (MHA) | 1.3B |
| `parcae/finetune/benice-lora-770m.yaml` | Parcae (MHA) | 770M |
| `parcae/finetune/benice-lora-1.3b.yaml` | Parcae (MHA) | 1.3B |
| `benice/finetune/benice-140m-kda.yaml` | Parcae (KDA) | 140M |
| `benice/finetune/benice-370m-kda.yaml` | Parcae (KDA) | 370M |
| `benice/finetune/benice-770m-kda.yaml` | Parcae (KDA) | 770M |
| `benice/finetune/benice-140m-mla.yaml` | Parcae (MLA) | 140M |
| `benice/finetune/benice-370m-mla.yaml` | Parcae (MLA) | 370M |
| `benice/finetune/benice-770m-mla.yaml` | Parcae (MLA) | 770M |
| `benice/finetune/benice-140m.yaml` | Parcae (MHA) | 140M |
| `benice/finetune/benice-370m.yaml` | Parcae (MHA) | 370M |
| `benice/finetune/benice-770m.yaml` | Parcae (MHA) | 770M |
| `benice/finetune/benice-1.3b.yaml` | Parcae (MHA) | 1.3B |
