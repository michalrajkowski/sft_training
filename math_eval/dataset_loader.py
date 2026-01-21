from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class MathTask:
    task_id: str
    question: str
    answer: str


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as handle:
        for line in handle:
            content = line.strip()
            if not content:
                continue
            yield json.loads(content)


def load_tasks(path: str | Path, limit: Optional[int] = None) -> List[MathTask]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    tasks: List[MathTask] = []
    for idx, record in enumerate(_read_jsonl(dataset_path)):
        if limit is not None and idx >= limit:
            break
        try:
            tasks.append(
                MathTask(
                    task_id=str(record["id"]),
                    question=str(record["question"]),
                    answer=str(record["answer"]),
                )
            )
        except KeyError as exc:
            raise KeyError(f"Malformed record missing {exc} in {dataset_path}") from exc
    return tasks
