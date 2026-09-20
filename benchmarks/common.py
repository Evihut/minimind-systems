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


ATTENTION_IMPLEMENTATIONS = ("manual", "sdpa")


def default_attention_implementation() -> str:
    """Historical default. Published CPU results were measured with the manual path.

    Kept as the default so existing artifacts stay reproducible; the GPU suite
    opts into ``sdpa`` explicitly and records which path produced each number.
    """
    return "manual"


def default_compile_backend(device: torch.device) -> str:
    """Inductor is the backend worth measuring on CUDA; aot_eager elsewhere."""
    return "inductor" if device.type == "cuda" else "aot_eager"


def build_model(
    preset_name: str,
    tokenizer,
    device: torch.device,
    max_position_embeddings: int,
    attention_implementation: str = "manual",
) -> tuple[MiniMindForCausalLM, ModelPreset]:
    try:
        preset = MODEL_PRESETS[preset_name]
    except KeyError as error:
        raise ValueError(f"unknown preset {preset_name!r}; choose from {sorted(MODEL_PRESETS)}") from error
    if attention_implementation not in ATTENTION_IMPLEMENTATIONS:
        raise ValueError(
            f"unknown attention implementation {attention_implementation!r}; "
            f"choose from {list(ATTENTION_IMPLEMENTATIONS)}"
        )
    config = MiniMindConfig(
        hidden_size=preset.hidden_size,
        num_hidden_layers=preset.num_hidden_layers,
        vocab_size=len(tokenizer),
        num_attention_heads=preset.num_attention_heads,
        num_key_value_heads=preset.num_key_value_heads,
        head_dim=preset.hidden_size // preset.num_attention_heads,
        max_position_embeddings=max_position_embeddings,
        flash_attn=attention_implementation == "sdpa",
        dropout=0.0,
    )
    return MiniMindForCausalLM(config).eval().to(device), preset


def attention_metadata(model: MiniMindForCausalLM, requested: str) -> dict:
    """Report the path that was requested and the one the model actually took.

    ``sdpa`` silently degrades to the manual path when the running PyTorch build
    has no ``scaled_dot_product_attention``, so the effective value is recorded
    rather than assumed.
    """
    effective = [bool(layer.self_attn.flash) for layer in model.model.layers]
    return {
        "requested": requested,
        "sdpa_enabled": all(effective),
        "effective": "sdpa" if all(effective) else "manual",
    }


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
