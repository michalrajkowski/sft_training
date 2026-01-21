from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DPO finetuning on a preference dataset.")
    parser.add_argument("--model-id", type=str, required=True, help="Base model repo or local path.")
    parser.add_argument("--adapter", type=str, required=True, help="Path to the LoRA adapter to start from (policy init).")
    parser.add_argument("--dataset", type=str, required=True, help="Path to JSONL with prompt/chosen/rejected fields.")
    parser.add_argument("--output-dir", type=str, default="outputs/dpo_run", help="Where to save the DPO adapter.")
    parser.add_argument("--limit", type=int, default=None, help="Optionally limit number of examples.")
    parser.add_argument("--use-4bit", action="store_true", help="Use 4-bit loading for policy/ref models.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta (strength of KL).")
    parser.add_argument("--max-length", type=int, default=1024, help="Max total sequence length.")
    parser.add_argument("--max-prompt-length", type=int, default=512, help="Max prompt length.")
    return parser.parse_args()


def load_models(model_id: str, adapter_path: Path, use_4bit: bool):
    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        )

    policy_base = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto",
        quantization_config=quant_config,
        trust_remote_code=True,
    )
    policy_model = PeftModel.from_pretrained(policy_base, adapter_path)

    ref_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto",
        quantization_config=quant_config,
        trust_remote_code=True,
    )
    return policy_model, ref_model


def prepare_dataset(path: Path, limit: Optional[int]):
    ds = load_dataset("json", data_files=str(path), split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    def _map(ex):
        return {
            "prompt": ex["prompt"].strip(),
            "chosen": ex["chosen"].strip(),
            "rejected": ex["rejected"].strip(),
        }

    return ds.map(_map, remove_columns=[c for c in ds.column_names if c not in {"prompt", "chosen", "rejected"}])


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    policy_model, ref_model = load_models(args.model_id, adapter_path, args.use_4bit)

    ds = prepare_dataset(Path(args.dataset), args.limit)

    dpo_args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        report_to="none",
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        beta=args.beta,
    )

    trainer = DPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "train_logs.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    print(f"[done] Saved DPO adapter to: {output_dir}")


if __name__ == "__main__":
    main()
