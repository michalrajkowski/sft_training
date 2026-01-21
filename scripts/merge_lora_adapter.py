from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure repo root on path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a base model.")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter directory.")
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model repo/path. If omitted, will try training_meta.json in the adapter.",
    )
    parser.add_argument("--output-dir", default=None, help="Where to save the merged model (default: <adapter>/merged).")
    parser.add_argument("--device-map", default="auto", help="Device map for loading the base model.")
    parser.add_argument("--dtype", default=None, help="torch dtype override (float16, bfloat16, float32).")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True when loading.")
    return parser.parse_args()


def resolve_base_model(adapter_dir: Path, explicit_base: Optional[str]) -> str:
    if explicit_base:
        return explicit_base
    meta_path = adapter_dir / "training_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        base_model = meta.get("base_model")
        if base_model:
            return base_model
    raise ValueError("Base model not provided and training_meta.json missing or lacks base_model.")


def resolve_dtype(dtype: Optional[str]):
    if dtype is None:
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    out = mapping.get(dtype.lower())
    if out is None:
        raise ValueError(f"Unsupported dtype '{dtype}'")
    return out


def main() -> None:
    args = parse_args()
    adapter_dir = Path(args.adapter).expanduser().resolve()
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    base_model = resolve_base_model(adapter_dir, args.base_model)
    dtype = resolve_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=args.trust_remote_code,
        device_map=args.device_map,
        torch_dtype=dtype,
    )

    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else adapter_dir / "merged"
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    meta = {
        "base_model": base_model,
        "adapter": str(adapter_dir),
        "output_dir": str(output_dir),
        "method": "merge_lora",
    }
    (output_dir / "merge_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] merged model saved to {output_dir}")


if __name__ == "__main__":
    main()
