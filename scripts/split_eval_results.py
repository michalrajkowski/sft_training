from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple


def parse_name(path: Path) -> Tuple[str, str]:
    """Parse dataset/model from a filename like <dataset>__<model>.jsonl."""
    stem = path.stem
    if "__" in stem:
        dataset, model = stem.split("__", 1)
    else:
        dataset, model = "unknown_dataset", stem
    return dataset, model


def split_file(path: Path, out_root: Path) -> None:
    dataset, model = parse_name(path)
    correct_path = out_root / model / f"{dataset}__correct.jsonl"
    wrong_path = out_root / model / f"{dataset}__wrong.jsonl"
    correct_path.parent.mkdir(parents=True, exist_ok=True)
    wrong_path.parent.mkdir(parents=True, exist_ok=True)

    with path.open() as src, correct_path.open("w") as cor, wrong_path.open("w") as wrg:
        for line in src:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("is_correct"):
                cor.write(json.dumps(rec, ensure_ascii=False) + "\n")
            else:
                wrg.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split eval JSONL results into correct/wrong per model.")
    parser.add_argument("--input-dir", default="results/mult_tasks_runs", help="Directory with eval JSONL files.")
    parser.add_argument("--output-dir", default="results/analysis_split", help="Where to write split files.")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    if not in_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {in_dir}")

    files = list(in_dir.glob("*.jsonl"))
    if not files:
        print(f"No JSONL files found in {in_dir}")
        return

    for f in files:
        split_file(f, out_dir)
    print(f"Processed {len(files)} files into {out_dir}")


if __name__ == "__main__":
    main()
