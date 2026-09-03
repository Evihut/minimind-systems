from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelPreset:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int


MODEL_PRESETS = {
    "smoke": ModelPreset(64, 2, 4, 2),
    "26m": ModelPreset(512, 7, 8, 2),
    "64m": ModelPreset(768, 8, 8, 4),
}


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def load_tokenizer():
    return AutoTokenizer.from_pretrained(PROJECT_ROOT / "model", local_files_only=True)


def build_model(
    preset_name: str,
    tokenizer,
    device: torch.device,
    max_position_embeddings: int,
) -> tuple[MiniMindForCausalLM, ModelPreset]:
    try:
        preset = MODEL_PRESETS[preset_name]
    except KeyError as error:
        raise ValueError(f"unknown preset {preset_name!r}; choose from {sorted(MODEL_PRESETS)}") from error
    config = MiniMindConfig(
        hidden_size=preset.hidden_size,
        num_hidden_layers=preset.num_hidden_layers,
        vocab_size=len(tokenizer),
        num_attention_heads=preset.num_attention_heads,
        num_key_value_heads=preset.num_key_value_heads,
        head_dim=preset.hidden_size // preset.num_attention_heads,
        max_position_embeddings=max_position_embeddings,
        flash_attn=False,
        dropout=0.0,
    )
    return MiniMindForCausalLM(config).eval().to(device), preset


def environment_metadata(device: torch.device) -> dict:
    accelerator = None
    if device.type == "cuda":
        accelerator = torch.cuda.get_device_name(device)
    elif device.type == "mps":
        accelerator = "Apple Metal Performance Shaders"
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "platform": platform.platform(),
        "device": str(device),
        "accelerator": accelerator,
    }


def model_metadata(model: MiniMindForCausalLM, preset: ModelPreset) -> dict:
    parameters = sum(parameter.numel() for parameter in model.parameters())
    return {
        "preset": asdict(preset),
        "parameters": parameters,
        "parameters_millions": round(parameters / 1_000_000, 4),
        "estimated_fp32_size_mb": round(parameters * 4 / 1024**2, 3),
    }


def write_json_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
