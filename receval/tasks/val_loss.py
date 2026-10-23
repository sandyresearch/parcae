import math
import torch
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class ValLossResults:
    loss: float
    perplexity: float
    bits_per_byte: float
    num_tokens: int
    num_bytes: int


def build_token_bytes(tokenizer, vocab_size: int, device: torch.device) -> torch.Tensor:
    token_bytes = torch.zeros(vocab_size, dtype=torch.int64, device=device)
    special_ids = set()
    for attr in ['bos_id', 'eos_id', 'pad_id', 'bos_token_id', 'eos_token_id', 'pad_token_id']:
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            special_ids.add(tid)
    for token_id in range(vocab_size):
        if token_id in special_ids:
            continue
        try:
            if hasattr(tokenizer, 'decode'):
                text = tokenizer.decode([token_id])
            else:
                text = tokenizer.id_to_piece(token_id) if hasattr(tokenizer, 'id_to_piece') else ""
            token_bytes[token_id] = len(text.encode('utf-8'))
        except Exception:
            token_bytes[token_id] = 0
    return token_bytes


@torch.no_grad()
def run_val_loss(model, tokenizer, settings, val_texts: list[str], max_samples: int = 1000) -> ValLossResults:
    device = settings.device
    seq_len = settings.sequence_length or 2048
    vocab_size = model.config.vocab_size if hasattr(model, 'config') else 32768
    token_bytes = build_token_bytes(tokenizer, vocab_size, device)
    total_nats = 0.0
    total_bytes = 0
    total_tokens = 0
    samples_processed = 0
    model.eval()
    selected_texts = val_texts[:max_samples]
    if settings.ddp_world_size > 1:
        selected_texts = selected_texts[settings.ddp_rank::settings.ddp_world_size]
    pbar = tqdm(selected_texts, desc="val_loss", disable=settings.ddp_rank != 0)
    for text in pbar:
        tokens = tokenizer.encode(text, return_tensors=False)
        if not isinstance(tokens, list):
            tokens = tokens.tolist()
        if len(tokens) < 2:
            continue
        tokens = tokens[:seq_len + 1]
        input_ids = torch.tensor([tokens[:-1]], device=device)
        targets = torch.tensor([tokens[1:]], device=device)
        with settings._get_autocast_context():
            outputs = model.forward_for_generation(input_ids)
            logits = outputs["logits"]
        loss_per_token = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction='none'
        )
        targets_flat = targets.view(-1)
        num_bytes_per_token = token_bytes[targets_flat]
        valid_mask = num_bytes_per_token > 0
        total_nats += (loss_per_token * valid_mask).sum().item()
        total_bytes += num_bytes_per_token.sum().item()
        total_tokens += valid_mask.sum().item()
        samples_processed += 1
        if samples_processed % 100 == 0:
            avg_loss = total_nats / total_tokens if total_tokens > 0 else 0
            pbar.set_postfix(loss=f"{avg_loss:.4f}", ppl=f"{math.exp(avg_loss):.2f}")
    if settings.ddp_world_size > 1:
        total_nats_t = torch.tensor(total_nats, dtype=torch.float64, device=device)
        total_tokens_t = torch.tensor(total_tokens, dtype=torch.int64, device=device)
        total_bytes_t = torch.tensor(total_bytes, dtype=torch.int64, device=device)
        torch.distributed.all_reduce(total_nats_t)
        torch.distributed.all_reduce(total_tokens_t)
        torch.distributed.all_reduce(total_bytes_t)
        total_nats = total_nats_t.item()
        total_tokens = int(total_tokens_t.item())
        total_bytes = int(total_bytes_t.item())
    avg_loss = total_nats / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    bits_per_byte = total_nats / (math.log(2) * total_bytes) if total_bytes > 0 else float('inf')
    return ValLossResults(loss=avg_loss, perplexity=perplexity, bits_per_byte=bits_per_byte, num_tokens=total_tokens, num_bytes=total_bytes)

def load_val_texts_from_parquet(data_dir: str, max_files: int = 5) -> list[str]:
    import pyarrow.parquet as pq
    from pathlib import Path
    texts = []
    parquet_files = sorted(Path(data_dir).glob("*.parquet"))[:max_files]
    for pf in parquet_files:
        table = pq.read_table(pf, columns=["text"])
        texts.extend(table["text"].to_pylist())
    return texts


