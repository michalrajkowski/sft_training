from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import yaml


@dataclass
class DatasetConfig:
    key: str
    display_name: str
    source: str
    split: str = "train"
    subset: Optional[str] = None
    format: str = "gsm8k"  # supported: gsm8k, pairwise
    output_path: Optional[Path] = None
    local_path: Optional[Path] = None

    @classmethod
    def from_dict(cls, key: str, payload: Dict, base_dir: Path) -> "DatasetConfig":
        def resolve(path_val: Optional[str]) -> Optional[Path]:
            if path_val is None:
                return None
            return (base_dir / path_val).expanduser().resolve()

        return cls(
            key=key,
            display_name=payload.get("display_name", key),
            source=payload["source"],
            split=payload.get("split", "train"),
            subset=payload.get("subset"),
            format=payload.get("format", "gsm8k"),
            output_path=resolve(payload.get("output_path")),
            local_path=resolve(payload.get("local_path")),
        )

    def resolved_path(self) -> Optional[Path]:
        if self.local_path:
            return self.local_path
        return self.output_path


def load_dataset_registry(path: str | Path) -> Dict[str, DatasetConfig]:
    registry_path = Path(path)
    raw = yaml.safe_load(registry_path.read_text())
    datasets = raw.get("datasets", {})
    return {
        key: DatasetConfig.from_dict(key, cfg, base_dir=registry_path.parent)
        for key, cfg in datasets.items()
    }
