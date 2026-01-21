from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

from datasets import load_dataset
from tqdm import tqdm

# Ensure repo root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from math_eval.dataset_registry import DatasetConfig, load_dataset_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and export datasets to JSONL.")
    parser.add_argument(
        "--registry",
        type=str,
        default="configs/datasets.yaml",
        help="Path to dataset registry YAML.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Specific dataset keys to download (default: all).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing exported files.",
    )
    return parser.parse_args()


def ensure_output_path(cfg: DatasetConfig) -> Path:
    resolved = cfg.resolved_path()
    if resolved is None:
        raise ValueError(f"No output_path or local_path provided for dataset {cfg.key}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def export_gsm8k(cfg: DatasetConfig, *, force: bool) -> Path:
    output_path = ensure_output_path(cfg)
    if output_path.exists() and not force:
        print(f"[skip] {cfg.key} already exists at {output_path}")
        return output_path

    ds = load_dataset(cfg.source, cfg.subset, split=cfg.split)
    with output_path.open("w") as handle:
        for idx, row in enumerate(tqdm(ds, desc=f"Exporting {cfg.key}")):
            raw_answer = row["answer"]
            final_answer = raw_answer.split("####")[-1].strip()
            record = {
                "id": f"{cfg.key}-{idx:05d}",
                "question": row["question"],
                "answer": final_answer,
                "solution": raw_answer,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[done] wrote {len(ds)} rows to {output_path}")
    return output_path


def export_pairwise(cfg: DatasetConfig, *, force: bool) -> Path:
    output_path = ensure_output_path(cfg)
    if output_path.exists() and not force:
        print(f"[skip] {cfg.key} already exists at {output_path}")
        return output_path

    ds = load_dataset(cfg.source, cfg.subset, split=cfg.split)
    with output_path.open("w") as handle:
        for idx, row in enumerate(tqdm(ds, desc=f"Exporting {cfg.key}")):
            record = {
                "id": f"{cfg.key}-{idx:05d}",
                "prompt": row["prompt"],
                "chosen": row.get("selected") or row.get("chosen"),
                "rejected": row.get("rejected"),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[done] wrote {len(ds)} rows to {output_path}")
    return output_path


def download_dataset(cfg: DatasetConfig, *, force: bool) -> Path:
    if cfg.local_path:
        print(f"[skip] {cfg.key}: local_path set to {cfg.local_path}")
        return cfg.local_path

    if cfg.format == "gsm8k":
        return export_gsm8k(cfg, force=force)
    if cfg.format == "pairwise":
        return export_pairwise(cfg, force=force)
    raise ValueError(f"Unsupported dataset format '{cfg.format}' for {cfg.key}")


def main() -> None:
    args = parse_args()
    registry = load_dataset_registry(args.registry)
    selected_keys = set(args.datasets) if args.datasets else set(registry.keys())

    missing = selected_keys - registry.keys()
    if missing:
        raise KeyError(f"Dataset keys not found: {', '.join(sorted(missing))}")

    for key in sorted(selected_keys):
        cfg = registry[key]
        download_dataset(cfg, force=args.force)


if __name__ == "__main__":
    main()
