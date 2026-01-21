from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_model_registry
from .dataset_loader import load_tasks
from .evaluator import MathEvaluator
from .model_loader import load_text_generation_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate math tasks with a selected LLM.")
    parser.add_argument("--model", type=str, help="Model key from configs/models.yaml")
    parser.add_argument("--dataset", type=str, default="data/sample_tasks.jsonl", help="Path to JSONL dataset.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples.")
    parser.add_argument("--registry", type=str, default="configs/models.yaml", help="Path to the model registry.")
    parser.add_argument("--output", type=str, default=None, help="Where to write JSONL results.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override max_new_tokens.")
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Override top_p.")
    parser.add_argument("--dtype", type=str, default=None, help="Override torch dtype (float32, float16, bfloat16).")
    parser.add_argument("--device", type=str, default=None, help="Override device map (e.g., 'auto', 'cuda:0', 'cpu').")
    parser.add_argument("--list-models", action="store_true", help="List available models and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = load_model_registry(args.registry)

    if args.list_models:
        print("Available models:")
        for key, cfg in registry.items():
            print(f"- {key}: {cfg.display_name} ({cfg.repo_id})")
        return

    if not args.model:
        print("Please provide --model. Use --list-models to see options.")
        sys.exit(1)

    if args.model not in registry:
        print(f"Unknown model '{args.model}'. Available: {', '.join(registry.keys())}")
        sys.exit(1)

    model_cfg = registry[args.model]
    dataset_path = Path(args.dataset)
    tasks = load_tasks(dataset_path, limit=args.limit)
    if not tasks:
        print(f"No tasks loaded from {dataset_path}")
        sys.exit(1)

    pipeline, gen_kwargs = load_text_generation_pipeline(
        model_cfg,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        dtype_override=args.dtype,
        device_override=args.device,
    )

    evaluator = MathEvaluator(pipeline, gen_kwargs)
    summary = evaluator.run(tasks)

    print(f"Accuracy: {summary['accuracy']*100:.1f}% ({summary['correct']}/{summary['total']})")

    if args.output:
        MathEvaluator.write_results(args.output, summary["results"])
        print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
