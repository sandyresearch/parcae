import os
import csv
import json
import time
import random
import shutil
import zipfile
import tempfile
import urllib.request
import filelock
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from jinja2 import Template
import torch
import torch.distributed as dist

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

def print0(*args, **kwargs):
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print(*args, **kwargs, flush=True)

def get_cache_dir():
    return Path(os.environ.get("PARCAE_CACHE", Path.home() / ".cache" / "parcae"))

def download_eval_bundle():
    cache_dir = get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    eval_bundle_dir = cache_dir / "eval_bundle"
    if eval_bundle_dir.exists():
        return eval_bundle_dir
    lock_path = cache_dir / "eval_bundle.lock"
    with filelock.FileLock(lock_path):
        if eval_bundle_dir.exists():
            return eval_bundle_dir
        print0(f"Downloading CORE eval bundle...")
        zip_path = cache_dir / "eval_bundle.zip"
        urllib.request.urlretrieve(EVAL_BUNDLE_URL, zip_path)
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(tmpdir)
            shutil.move(os.path.join(tmpdir, "eval_bundle"), eval_bundle_dir)
        zip_path.unlink()
        print0(f"Eval bundle extracted to {eval_bundle_dir}")
    return eval_bundle_dir

def get_bos_id(tokenizer):
    if hasattr(tokenizer, 'bos_id') and tokenizer.bos_id is not None:
        return tokenizer.bos_id
    if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
        return tokenizer.bos_token_id
    return 1

def get_pad_id(tokenizer):
    """Get pad token ID matching training configuration (pad_id or 0)."""
    if hasattr(tokenizer, 'pad_id') and tokenizer.pad_id is not None:
        return tokenizer.pad_id
    if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    return 0  # Default to 0, matching training (scripts/train.py line 325)

def encode_with_bos(tokenizer, text):
    bos = get_bos_id(tokenizer)
    if hasattr(tokenizer, 'encode'):
        ids = tokenizer.encode(text)
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
    else:
        ids = tokenizer(text)
    if not ids or ids[0] != bos:
        ids = [bos] + ids
    return ids

def render_prompts_mc(item, delim, fewshot):
    tpl = Template("""
{%- for ex in fewshot -%}
{{ ex.query }}{{ delim }}{{ ex.choices[ex.gold] }}

{% endfor -%}
{{ item.query }}{{ delim }}{{ choice }}""".strip())
    return [tpl.render(fewshot=fewshot, delim=delim, item=item, choice=c) for c in item['choices']]

def render_prompts_schema(item, delim, fewshot):
    tpl = Template("""
{%- for ex in fewshot -%}
{{ ex.context_options[ex.gold] }}{{ delim }}{{ ex.continuation }}

{% endfor -%}
{{ ctx }}{{ delim }}{{ item.continuation }}""".strip())
    return [tpl.render(fewshot=fewshot, delim=delim, item=item, ctx=c) for c in item['context_options']]

def render_prompts_lm(item, delim, fewshot):
    tpl = Template("""
{%- for ex in fewshot -%}
{{ ex.context | trim }}{{ delim }}{{ ex.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ delim }}{% if inc %}{{ item.continuation }}{% endif %}""".strip())
    ctx = {'fewshot': fewshot, 'delim': delim, 'item': item}
    return [tpl.render(inc=False, **ctx).strip(), tpl.render(inc=True, **ctx)]

def find_common_length(seqs, direction='left'):
    min_len = min(len(s) for s in seqs)
    idxs = range(min_len) if direction == 'left' else range(-1, -min_len-1, -1)
    for i, idx in enumerate(idxs):
        if not all(s[idx] == seqs[0][idx] for s in seqs):
            return i
    return min_len

def stack_sequences(tokens, pad_id):
    bsz, seq_len = len(tokens), max(len(t) for t in tokens)
    ids = torch.full((bsz, seq_len), pad_id, dtype=torch.long)
    for i, t in enumerate(tokens):
        ids[i, :len(t)] = torch.tensor(t, dtype=torch.long)
    return ids

def batch_mc(tokenizer, prompts):
    tokens = [encode_with_bos(tokenizer, p) for p in prompts]
    start = find_common_length(tokens, 'left')
    return tokens, [start] * len(prompts), [len(t) for t in tokens]

def batch_schema(tokenizer, prompts):
    tokens = [encode_with_bos(tokenizer, p) for p in prompts]
    suffix_len = find_common_length(tokens, 'right')
    ends = [len(t) for t in tokens]
    return tokens, [e - suffix_len for e in ends], ends

def batch_lm(tokenizer, prompts):
    t_without = encode_with_bos(tokenizer, prompts[0])
    t_with = encode_with_bos(tokenizer, prompts[1])
    assert t_without == t_with[:len(t_without)], "prompt without is supposed to be a prefix of prompt with"
    return [t_with], [len(t_without)], [len(t_with)]

@torch.no_grad()
def forward_model(model, input_ids, device, prefix_lengths=None):
    if hasattr(model, "forward_tokens"):
        return model.forward_tokens(input_ids, prefix_lengths=prefix_lengths)
    input_ids = input_ids.to(device)
    try:
        import inspect
        sig = inspect.signature(model.forward)
        kwargs = {}
        if 'return_logits' in sig.parameters:
            kwargs['return_logits'] = True
        if 'position_ids' in sig.parameters:
            kwargs['position_ids'] = None
        if 'attention_mask' in sig.parameters:
            kwargs['attention_mask'] = None
        # For Parcae models: force deterministic recurrence depth (no stochastic sampling)
        if 'num_steps_pair' in sig.parameters and hasattr(model, 'config') and hasattr(model.config, 'mean_recurrence'):
            kwargs['num_steps_pair'] = torch.tensor([model.config.mean_recurrence, 0], device=device)
        outputs = model(input_ids, **kwargs)
    except Exception:
        outputs = model(input_ids)
    if isinstance(outputs, dict):
        logits = outputs.get('logits')
        if logits is None:
            for v in outputs.values():
                if isinstance(v, torch.Tensor) and v.dim() >= 2:
                    logits = v
                    break
        outputs = logits
    elif hasattr(outputs, 'logits'):
        outputs = outputs.logits
    if outputs is None:
        raise ValueError(f"Could not extract logits from model output. Keys: {outputs.keys() if isinstance(outputs, dict) else type(outputs)}")
    bsz, seq_len = input_ids.size()
    targets = torch.roll(input_ids, -1, 1)
    losses = torch.nn.functional.cross_entropy(
        outputs.view(bsz * seq_len, -1), targets.view(bsz * seq_len), reduction='none'
    ).view(bsz, seq_len)
    losses[:, -1] = float('nan')
    return losses, outputs.argmax(-1)

@torch.no_grad()
def evaluate_example(idx, model, tokenizer, data, device, task_meta, max_seq_len, seed=1234):
    item = data[idx]
    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']
    delim = task_meta['continuation_delimiter']
    fewshot = []
    if num_fewshot > 0:
        rng = random.Random(seed + idx)
        avail = [i for i in range(len(data)) if i != idx]
        fewshot = [data[i] for i in rng.sample(avail, num_fewshot)]
    if task_type == 'multiple_choice':
        prompts = render_prompts_mc(item, delim, fewshot)
        tokens, starts, ends = batch_mc(tokenizer, prompts)
    elif task_type == 'schema':
        prompts = render_prompts_schema(item, delim, fewshot)
        tokens, starts, ends = batch_schema(tokenizer, prompts)
    elif task_type == 'language_modeling':
        prompts = render_prompts_lm(item, delim, fewshot)
        tokens, starts, ends = batch_lm(tokenizer, prompts)
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    if max_seq_len:
        new_tokens, new_starts, new_ends = [], [], []
        for t, s, e in zip(tokens, starts, ends):
            if len(t) > max_seq_len:
                crop = len(t) - max_seq_len
                new_tokens.append(t[-max_seq_len:])
                new_starts.append(s - crop)
                new_ends.append(e - crop)
            else:
                new_tokens.append(t)
                new_starts.append(s)
                new_ends.append(e)
        tokens, starts, ends = new_tokens, new_starts, new_ends
    pad_id = get_pad_id(tokenizer)  # use pad token matching training (0 by default)
    input_ids = stack_sequences(tokens, pad_id)
    losses, preds = forward_model(
        model, input_ids, device, prefix_lengths=starts,
    )
    if task_type == 'language_modeling':
        si, ei = starts[0], ends[0]
        pred_toks = preds[0, si-1:ei-1]
        actual_toks = input_ids[0, si:ei].to(preds.device)
        return torch.all(pred_toks == actual_toks).item()
    else:
        mean_losses = [losses[i, s-1:e-1].mean().item() for i, (s, e) in enumerate(zip(starts, ends))]
        return mean_losses.index(min(mean_losses)) == item['gold']


def evaluate_task(model, tokenizer, data, device, task_meta, max_seq_len=None, seed=1234):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    correct = torch.zeros(len(data), dtype=torch.float32, device=device)
    indices = list(range(rank, len(data), world_size))
    def evaluate(idx):
        return evaluate_example(
            idx, model, tokenizer, data, device, task_meta, max_seq_len, seed=seed
        )
    results = map(evaluate, indices)
    if workers := getattr(model, "batch_size", None):
        with ThreadPoolExecutor(workers) as executor:
            results = list(executor.map(evaluate, indices))
    for idx, is_correct in zip(indices, results):
        correct[idx] = float(is_correct)
    if world_size > 1:
        dist.barrier(device_ids=[torch.device(device).index] if torch.device(device).type == "cuda" else None)
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    return correct.mean().item()

def run_core_eval(model, tokenizer, device, max_seq_len=None, max_per_task=-1, seeds=None):
    import yaml
    if seeds is None:
        seeds = [1234]
    eval_bundle = download_eval_bundle()
    config_path = eval_bundle / "core.yaml"
    data_path = eval_bundle / "eval_data"
    meta_path = eval_bundle / "eval_meta_data.csv"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    baselines = {}
    with open(meta_path, 'r') as f:
        for row in csv.DictReader(f):
            baselines[row['Eval Task']] = float(row['Random baseline'])
    all_seed_results = {}
    for seed in seeds:
        if len(seeds) > 1:
            print0(f"\n{'='*60}")
            print0(f"Running CORE eval with seed={seed}")
            print0(f"{'='*60}")
        results, centered = {}, {}
        for task in config['icl_tasks']:
            label = task['label']
            task_meta = {
                'task_type': task['icl_task_type'],
                'num_fewshot': task['num_fewshot'][0],
                'continuation_delimiter': task.get('continuation_delimiter', ' ')
            }
            print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... ", end='')
            t0 = time.time()
            with open(data_path / task['dataset_uri'], 'r') as f:
                data = [json.loads(line) for line in f]
            rng = random.Random(1337)
            rng.shuffle(data)
            if max_per_task > 0:
                data = data[:max_per_task]
            acc = evaluate_task(model, tokenizer, data, device, task_meta, max_seq_len, seed=seed)
            results[label] = acc
            baseline = baselines[label]
            centered[label] = (acc - 0.01 * baseline) / (1.0 - 0.01 * baseline)
            print0(f"accuracy: {acc:.4f} | centered: {centered[label]:.4f} | time: {time.time()-t0:.2f}s")
        core_score = sum(centered.values()) / len(centered)
        all_seed_results[seed] = {'results': results, 'centered_results': centered, 'core_metric': core_score}
        if len(seeds) > 1:
            print0(f"\nSeed {seed} CORE metric: {core_score:.4f}")
    if len(seeds) == 1:
        return all_seed_results[seeds[0]]
    task_labels = list(all_seed_results[seeds[0]]['results'].keys())
    avg_results, std_results, avg_centered, std_centered = {}, {}, {}, {}
    for label in task_labels:
        accs = [all_seed_results[s]['results'][label] for s in seeds]
        cents = [all_seed_results[s]['centered_results'][label] for s in seeds]
        avg_results[label] = sum(accs) / len(accs)
        avg_centered[label] = sum(cents) / len(cents)
        std_results[label] = (sum((x - avg_results[label])**2 for x in accs) / len(accs)) ** 0.5
        std_centered[label] = (sum((x - avg_centered[label])**2 for x in cents) / len(cents)) ** 0.5
    core_scores = [all_seed_results[s]['core_metric'] for s in seeds]
    avg_core = sum(core_scores) / len(core_scores)
    std_core = (sum((x - avg_core)**2 for x in core_scores) / len(core_scores)) ** 0.5
    print0(f"\n{'='*60}")
    print0(f"AGGREGATED RESULTS (across {len(seeds)} seeds)")
    print0(f"{'='*60}")
    for label in task_labels:
        print0(f"  {label}: acc={avg_results[label]:.4f}±{std_results[label]:.4f} "
               f"centered={avg_centered[label]:.4f}±{std_centered[label]:.4f}")
    print0(f"\n  CORE metric: {avg_core:.4f} ± {std_core:.4f}")
    return {
        'per_seed': all_seed_results,
        'aggregated': {
            'results': avg_results, 'results_std': std_results,
            'centered_results': avg_centered, 'centered_results_std': std_centered,
            'core_metric': avg_core, 'core_metric_std': std_core,
        },
        'seeds': seeds,
        'results': avg_results,
        'centered_results': avg_centered,
        'core_metric': avg_core,
    }

