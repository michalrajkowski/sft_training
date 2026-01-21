from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import yaml

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful mathematician. Show concise reasoning, then end your response "
    "with a single line of the form 'Final Answer: <value>'."
)


@dataclass
class GenerationSettings:
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.95

    @classmethod
    def from_dict(cls, payload: Dict) -> "GenerationSettings":
        return cls(
            max_new_tokens=int(payload.get("max_new_tokens", 256)),
            temperature=float(payload.get("temperature", 0.2)),
            top_p=float(payload.get("top_p", 0.95)),
        )

    def to_kwargs(self) -> Dict:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }


@dataclass
class ModelConfig:
    key: str
    display_name: str
    repo_id: str
    local_path: Optional[str] = None
    revision: Optional[str] = None
    trust_remote_code: bool = False
    torch_dtype: Optional[str] = None
    device_map: Optional[str] = None
    generation: GenerationSettings = field(default_factory=GenerationSettings)

    @classmethod
    def from_dict(cls, key: str, payload: Dict, base_dir: Optional[Path] = None) -> "ModelConfig":
        local_path = payload.get("local_path")
        if local_path and base_dir:
            local_path = str((base_dir / local_path).expanduser().resolve())

        return cls(
            key=key,
            display_name=payload.get("display_name", key),
            repo_id=payload["repo_id"],
            local_path=local_path,
            revision=payload.get("revision"),
            trust_remote_code=bool(payload.get("trust_remote_code", False)),
            torch_dtype=payload.get("torch_dtype"),
            device_map=payload.get("device_map"),
            generation=GenerationSettings.from_dict(payload.get("generation", {})),
        )


def load_model_registry(path: str | Path) -> Dict[str, ModelConfig]:
    registry_path = Path(path)
    raw = yaml.safe_load(registry_path.read_text())
    models = raw.get("models", {})
    return {
        key: ModelConfig.from_dict(key, cfg, base_dir=registry_path.parent)
        for key, cfg in models.items()
    }
