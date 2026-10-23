import argparse
import os
from pathlib import Path
from huggingface_hub import snapshot_download

DATA_DIR = os.environ.get("DATA_DIR", "/resource/data")
CACHE_DIR = os.environ.get("HF_CACHE_DIR", os.path.join(DATA_DIR, ".cache/huggingface"))

DATASETS = {
    "fineweb": {
        "repo_id": "SandyResearch/fineweb-edu-shuffled",
        "local_dir": f"{DATA_DIR}/fineweb-edu-shuffled",
        "splits": ["sample-100BT", "sample-350BT", "val"],
        "split_prefix": "data/",
    },
    "fineweb-100bt": {
        "repo_id": "SandyResearch/fineweb-edu-shuffled",
        "local_dir": f"{DATA_DIR}/fineweb-edu-shuffled",
        "splits": ["sample-100BT", "val"],
        "split_prefix": "data/",
    },
    "fineweb-350bt": {
        "repo_id": "SandyResearch/fineweb-edu-shuffled",
        "local_dir": f"{DATA_DIR}/fineweb-edu-shuffled",
        "splits": ["sample-350BT", "val"],
        "split_prefix": "data/",
    },
    "fineweb-val": {
        "repo_id": "SandyResearch/fineweb-edu-shuffled",
        "local_dir": f"{DATA_DIR}/fineweb-edu-shuffled",
        "splits": ["val"],
        "split_prefix": "data/",
    },
    "climbmix": {
        "repo_id": "SandyResearch/climbmix-shuffle",
        "local_dir": f"{DATA_DIR}/climbmix-shuffle",
        "splits": ["100b", "400b", "val"],
    },
    "climbmix-100bt": {
        "repo_id": "SandyResearch/climbmix-shuffle",
        "local_dir": f"{DATA_DIR}/climbmix-shuffle",
        "splits": ["100b", "val"],
    },
    "climbmix-400bt": {
        "repo_id": "SandyResearch/climbmix-shuffle",
        "local_dir": f"{DATA_DIR}/climbmix-shuffle",
        "splits": ["400b", "val"],
    },
    "climbmix-val": {
        "repo_id": "SandyResearch/climbmix-shuffle",
        "local_dir": f"{DATA_DIR}/climbmix-shuffle",
        "splits": ["val"],
    },
    "huginn": {
        "repo_id": "tomg-group-umd/huginn-dataset",
        "local_dir": f"{DATA_DIR}/huginn-dataset",
        "val_percent": 1.0,
    },
}


def create_train_val_split(data_dir: str, val_percent: float = 1.0):
    data_path = Path(data_dir)
    train_dir = data_path / "train"
    val_dir = data_path / "val"
    
    parquet_files = sorted(data_path.glob("*.parquet"))
    
    if not parquet_files:
        print(f"No parquet files found in {data_dir}")
        return
    
    if train_dir.exists() and val_dir.exists():
        train_count = len(list(train_dir.glob("*.parquet")))
        val_count = len(list(val_dir.glob("*.parquet")))
        if train_count > 0 and val_count > 0:
            print(f"Split already exists: {train_count} train, {val_count} val shards")
            return
    
    total_shards = len(parquet_files)
    val_shards = max(1, int(total_shards * val_percent / 100))
    train_shards = total_shards - val_shards
    
    print(f"Creating split: {train_shards} train, {val_shards} val shards")
    
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    for i, pq_file in enumerate(parquet_files[:train_shards]):
        dest = train_dir / pq_file.name
        if not dest.exists():
            dest.symlink_to(pq_file.resolve())
    
    for pq_file in parquet_files[train_shards:]:
        dest = val_dir / pq_file.name
        if not dest.exists():
            dest.symlink_to(pq_file.resolve())
    
    print(f"Created train/val split in {data_dir}")
    print(f"  Train: {train_dir} ({train_shards} shards)")
    print(f"  Val: {val_dir} ({val_shards} shards)")


def download_with_splits(cfg: dict):
    repo_id = cfg["repo_id"]
    base_dir = Path(cfg["local_dir"])
    splits = cfg["splits"]
    prefix = cfg.get("split_prefix", "")

    for split in splits:
        split_dir = base_dir / f"{prefix}{split}"
        print(f"\nDownloading split '{split}' to {split_dir}...")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            cache_dir=CACHE_DIR,
            local_dir=str(base_dir),
            allow_patterns=[f"{prefix}{split}/*.parquet"],
            local_dir_use_symlinks=True,
        )
        print(f"Downloaded (symlinked) {split} to {split_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download datasets from HuggingFace")
    parser.add_argument("dataset", choices=DATASETS.keys(), help="Dataset to download")
    parser.add_argument("--no-split", action="store_true", help="Skip creating train/val split")
    parser.add_argument("--val-percent", type=float, default=None, 
                        help="Percentage of shards for validation (default: dataset-specific)")
    parser.add_argument("--split-only", action="store_true", 
                        help="Only create split (skip download)")
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    local_dir = cfg["local_dir"]
    
    if not args.split_only:
        print(f"Downloading {args.dataset}...")
        
        if "splits" in cfg:
            download_with_splits(cfg)
        else:
            snapshot_download(
                repo_id=cfg["repo_id"],
                repo_type="dataset",
            cache_dir=CACHE_DIR,
            local_dir=local_dir,
                allow_patterns=["**.parquet"],
                local_dir_use_symlinks=True,
            )
            print(f"Downloaded (symlinked) to {local_dir}")
    
    if not args.no_split and "splits" not in cfg:
        val_percent = args.val_percent if args.val_percent is not None else cfg.get("val_percent", 1.0)
        print(f"\nCreating train/val split ({val_percent}% validation)...")
        create_train_val_split(local_dir, val_percent)


if __name__ == "__main__":
    main()
