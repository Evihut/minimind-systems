"""Run a reproducible, local end-to-end MiniMind training smoke test.

The goal is not to produce a useful checkpoint.  It verifies that the bundled
tokenizer, Transformer, optimizer, backward pass, KV cache, and generation path
work together on a laptop, then records honest machine-specific measurements.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import setup_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = PROJECT_ROOT / "artifacts" / "smoke_report.json"


@dataclass(frozen=True)
class SmokeConfig:
    steps: int = 20
    batch_size: int = 8
    sequence_length: int = 48
    hidden_size: int = 64
    num_hidden_layers: int = 2
    learning_rate: float = 3e-3
    seed: int = 2026
    generation_tokens: int = 16


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_batch(tokenizer, config: SmokeConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    corpus = [
        "人工智能需要可靠的数据、清晰的目标和可复现的评估。",
        "MiniMind is a compact language model built with native PyTorch.",
        "机器学习工程强调实验记录、自动测试和可追溯的模型版本。",
        "A small model makes architecture experiments fast and affordable.",
        "预训练学习下一个 token，监督微调学习遵循用户指令。",
        "Grouped-query attention reduces the memory used by the KV cache.",
        "混合专家模型让每个 token 只激活少量专家网络。",
        "Reproducible benchmarks turn an interesting demo into engineering evidence.",
    ]
    texts = [corpus[index % len(corpus)] for index in range(config.batch_size)]
    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=config.sequence_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    labels = input_ids.clone()
    labels[encoded["attention_mask"].to(device) == 0] = -100
    return input_ids, labels


@torch.inference_mode()
def evaluate_loss(model: MiniMindForCausalLM, input_ids: torch.Tensor, labels: torch.Tensor) -> float:
    model.eval()
    return float(model(input_ids, labels=labels).loss.detach().cpu())


@torch.inference_mode()
def benchmark_generation(
    model: MiniMindForCausalLM,
    tokenizer,
    device: torch.device,
    max_new_tokens: int,
) -> dict[str, float | int | str]:
    prompt = tokenizer("MiniMind", return_tensors="pt")["input_ids"].to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    generated = model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        eos_token_id=None,
        use_cache=True,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    produced = generated.shape[1] - prompt.shape[1]
    return {
        "generated_tokens": produced,
        "generation_seconds": round(elapsed, 6),
        "decode_tokens_per_second": round(produced / max(elapsed, 1e-9), 3),
        "sample": tokenizer.decode(generated[0], skip_special_tokens=True),
    }


def run_smoke(config: SmokeConfig, device: torch.device) -> dict:
    setup_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(PROJECT_ROOT / "model", local_files_only=True)
    model_config = MiniMindConfig(
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        vocab_size=len(tokenizer),
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=config.hidden_size // 4,
        max_position_embeddings=config.sequence_length + config.generation_tokens + 16,
        flash_attn=False,
        dropout=0.0,
    )
    model = MiniMindForCausalLM(model_config).to(device)
    input_ids, labels = build_batch(tokenizer, config, device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    initial_loss = evaluate_loss(model, input_ids, labels)
    valid_tokens = int((labels[:, 1:] != -100).sum().item())
    model.train()
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - started
    final_loss = evaluate_loss(model, input_ids, labels)
    loss_reduction = (initial_loss - final_loss) / initial_loss

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    result = {
        "schema_version": 1,
        "status": "passed" if final_loss < initial_loss else "failed",
        "config": asdict(config),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "platform": platform.platform(),
            "device": str(device),
        },
        "model": {
            "parameters": parameter_count,
            "parameters_millions": round(parameter_count / 1_000_000, 4),
            "estimated_fp32_size_mb": round(parameter_count * 4 / 1024**2, 3),
        },
        "training": {
            "initial_loss": round(initial_loss, 6),
            "final_loss": round(final_loss, 6),
            "loss_reduction_percent": round(loss_reduction * 100, 3),
            "training_seconds": round(train_seconds, 6),
            "tokens_per_second": round(valid_tokens * config.steps / max(train_seconds, 1e-9), 3),
        },
        "inference": benchmark_generation(model, tokenizer, device, config.generation_tokens),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=SmokeConfig.steps)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda, or a concrete torch device")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--min-loss-reduction", type=float, default=5.0, help="minimum required percentage")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    config = SmokeConfig(steps=args.steps)
    result = run_smoke(config, resolve_device(args.device))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + os.linesep, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["training"]["loss_reduction_percent"] < args.min_loss_reduction:
        print(
            f"loss reduction did not reach {args.min_loss_reduction:.1f}%",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

