from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from tqdm import tqdm

from .config import DEFAULT_SYSTEM_PROMPT
from .dataset_loader import MathTask

try:
    from math_verify import parse as mv_parse
    from math_verify import verify as mv_verify
except ImportError:
    mv_parse = mv_verify = None  # type: ignore


@dataclass
class EvalResult:
    task_id: str
    question: str
    reference_answer: str
    model_output: str
    extracted_answer: str
    is_correct: bool
    latency_s: float


def build_prompt(tokenizer, question: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system_prompt}\n\nQuestion: {question}\n\nAnswer:"


def extract_final_answer(text: str) -> str:
    patterns = [
        r"Final Answer\s*[:：]\s*(.+)",
        r"Answer\s*[:：]\s*(.+)",
        r"=+\s*([-+*/()0-9\\.\\s]+)",
    ]
    candidate = None
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if matches:
            candidate = matches[-1].strip()
            break
    if candidate:
        return _cleanup_answer(candidate)
    # Fallback to last non-empty line.
    for line in reversed(text.strip().splitlines()):
        stripped = line.strip()
        if stripped:
            return _cleanup_answer(stripped)
    return text.strip()


def _cleanup_answer(answer: str) -> str:
    cleaned = answer.strip()
    cleaned = cleaned.rstrip(".")
    return cleaned


def normalize_answer(answer: str) -> str:
    return re.sub(r"\s+", "", answer).strip().rstrip(".")


def verify_answer(reference: str, predicted: str) -> bool:
    if mv_parse and mv_verify:
        try:
            return bool(mv_verify(mv_parse(reference), mv_parse(predicted)))
        except Exception:
            # Fallback to simple normalization if parsing fails.
            pass
    return normalize_answer(reference) == normalize_answer(predicted)


class MathEvaluator:
    def __init__(
        self,
        pipeline,
        generation_kwargs: Dict,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        batch_size: int = 1,
    ):
        self.pipeline = pipeline
        self.generation_kwargs = generation_kwargs
        self.system_prompt = system_prompt
        self.batch_size = max(1, batch_size)

    def run(self, tasks: List[MathTask]) -> Dict:
        results: List[EvalResult] = []
        correct = 0
        iterator = tqdm(tasks, desc="Evaluating", leave=False)
        for batch in _batched(tasks, self.batch_size):
            prompts = [build_prompt(self.pipeline.tokenizer, t.question, self.system_prompt) for t in batch]
            start = time.perf_counter()
            outputs = self.pipeline(prompts, batch_size=self.batch_size, **self.generation_kwargs)
            batch_latency = time.perf_counter() - start
            # pipeline returns list[list[dict]] when batched
            for task, output in zip(batch, outputs):
                choice = output[0] if isinstance(output, list) else output
                generated_text = choice["generated_text"]
                extracted = extract_final_answer(generated_text)
                is_correct = verify_answer(task.answer, extracted)
                correct += int(is_correct)
                results.append(
                    EvalResult(
                        task_id=task.task_id,
                        question=task.question,
                        reference_answer=task.answer,
                        model_output=generated_text,
                        extracted_answer=extracted,
                        is_correct=is_correct,
                        latency_s=batch_latency / max(1, len(batch)),
                    )
                )
                iterator.update(1)
                iterator.set_postfix(acc=f"{correct}/{len(results)}")

        total = len(results)
        accuracy = correct / total if total else 0.0
        return {"accuracy": accuracy, "total": total, "correct": correct, "results": results}

    @staticmethod
    def write_results(path: str | Path, results: List[EvalResult]) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as handle:
            for record in results:
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def _batched(items: List[MathTask], batch_size: int) -> Iterable[List[MathTask]]:
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]
