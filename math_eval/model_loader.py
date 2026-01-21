from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .config import ModelConfig


def _resolve_dtype(dtype: Optional[str]) -> Optional[torch.dtype]:
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
    resolved = mapping.get(dtype.lower())
    if resolved == torch.float16 and not torch.cuda.is_available():
        # Avoid float16 on CPU; fall back to float32 for correctness.
        return torch.float32
    return resolved


def load_text_generation_pipeline(
    model_config: ModelConfig,
    *,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    dtype_override: Optional[str] = None,
    device_override: Optional[str] = None,
) -> Tuple[object, Dict]:
    """Load a text-generation pipeline and generation settings."""
    torch_dtype = _resolve_dtype(dtype_override or model_config.torch_dtype)
    source = model_config.local_path or model_config.repo_id
    local_files_only = bool(model_config.local_path)
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        revision=model_config.revision,
        trust_remote_code=model_config.trust_remote_code,
        local_files_only=local_files_only,
    )
    # Left padding is safer for decoder-only generation to avoid misplaced positions.
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        source,
        revision=model_config.revision,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch_dtype,
        device_map=device_override or model_config.device_map,
        local_files_only=local_files_only,
    )

    text_pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch_dtype,
        device_map=device_override or model_config.device_map,
    )

    # Some chat models do not set a pad token; align with EOS to avoid warnings.
    if text_pipe.tokenizer.pad_token_id is None:
        text_pipe.tokenizer.pad_token_id = text_pipe.tokenizer.eos_token_id

    generation_kwargs = model_config.generation.to_kwargs()
    if max_new_tokens is not None:
        generation_kwargs["max_new_tokens"] = max_new_tokens
    if temperature is not None:
        generation_kwargs["temperature"] = temperature
    if top_p is not None:
        generation_kwargs["top_p"] = top_p

    # Let pipeline strip prompts from its output for easier parsing.
    generation_kwargs.setdefault("return_full_text", False)
    return text_pipe, generation_kwargs
