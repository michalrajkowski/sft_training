from __future__ import annotations

import argparse
from pathlib import Path
import json
from typing import List

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_TARGET_MODULES: List[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "down_proj",
    "gate_proj",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LoRA SFT using TRL's SFTTrainer.")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL, help="Base model repo or local path.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to JSONL with question/answer/solution fields.")
    parser.add_argument("--output-dir", type=str, default="outputs/sft_trl", help="Where to save the adapter.")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--use-4bit", action="store_true", help="Use 4-bit QLoRA.")
    parser.add_argument(
        "--target-modules",
        nargs="*",
        default=DEFAULT_TARGET_MODULES,
        help="Target modules for LoRA.",
    )
    parser.add_argument("--packing", action="store_true", help="Enable packing (disable for safety on math solutions).")
    return parser.parse_args()


def make_text(example, tokenizer: AutoTokenizer) -> str:
    question = example["question"].strip()
    completion = (example.get("solution") or f"#### {example['answer']}").strip()
    messages = [
        {"role": "system", "content": "You are a helpful assistant that solves math word problems."},
        {"role": "user", "content": f"Solve the problem and give the final answer.\n\nProblem: {question}"},
        {"role": "assistant", "content": completion},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("json", data_files=args.dataset, split="train")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=True)

    quant_config = None
    if args.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype="auto",
        quantization_config=quant_config,
        trust_remote_code=True,
    )

    def _map(ex):
        return {"text": make_text(ex, tokenizer)}

    ds = ds.map(_map, remove_columns=ds.column_names)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=args.target_modules,
    )

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        report_to="none",
        max_length=args.max_seq_length,
        packing=args.packing,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        peft_config=peft_config,
        args=sft_config,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    # Persist training log history for inspection.
    logs_path = output_dir / "train_logs.json"
    logs_path.write_text(json.dumps(trainer.state.log_history, indent=2))
    print(f"[done] Saved adapter to: {output_dir}")


if __name__ == "__main__":
    main()
