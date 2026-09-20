"""Reproducible MiniMind training benchmark with validation and DDP support."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import psutil
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from benchmarks.common import (
    ATTENTION_IMPLEMENTATIONS,
    MODEL_PRESETS,
    PROJECT_ROOT,
    attention_metadata,
    build_model,
    default_attention_implementation,
    default_compile_backend,
    environment_metadata,
    load_tokenizer,
    model_metadata,
    resolve_device,
    synchronize,
    write_json_report,
)


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


BUILTIN_TRAIN_TEXTS = [
    "MiniMind uses grouped query attention to reduce key value cache memory.",
    "MiniMind 使用分组查询注意力降低 KV Cache 内存。",
    "A static cache preallocates memory and avoids repeated tensor concatenation.",
    "静态缓存预分配内存，避免解码阶段反复拼接张量。",
    "RMSNorm, rotary embeddings, and SwiGLU form the Transformer block.",
    "可复现实验必须记录版本、硬件、配置和随机种子。",
    "Validation perplexity measures next-token prediction on held-out text.",
    "Training throughput is reported as non-padding tokens processed per second.",
] * 4

BUILTIN_VALIDATION_TEXTS = [
    "Grouped query attention shares key value heads to reduce cache memory.",
    "预分配 KV Cache 可以避免反复的内存申请。",
    "A reliable benchmark records latency, throughput, memory, and correctness.",
    "验证集困惑度用于检查模型对未见文本的预测。",
] * 2


def load_jsonl_texts(path: Path, field: str, limit: int) -> list[str]:
    texts = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if field in item and item[field]:
                texts.append(str(item[field]))
            if len(texts) >= limit:
                break
    if not texts:
        raise ValueError(f"no non-empty {field!r} values found in {path}")
    return texts


def make_dataset(tokenizer, texts: list[str], sequence_length: int) -> TensorDataset:
    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=sequence_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100
    return TensorDataset(encoded["input_ids"], labels)


def setup_distributed(requested_device: str) -> tuple[torch.device, int, int, int]:
    rank = int(os.environ.get("RANK", "-1"))
    if rank < 0:
        return resolve_device(requested_device), 0, 0, 1
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return device, rank, local_rank, world_size


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


@torch.inference_mode()
def evaluate(model, loader: DataLoader, device: torch.device, precision: str) -> tuple[float, int]:
    model.eval()
    loss_sum = torch.zeros(1, device=device)
    batch_count = torch.zeros(1, device=device)
    for input_ids, labels in loader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        with autocast_context(device, precision):
            loss = model(input_ids, labels=labels).loss
        loss_sum += loss.detach()
        batch_count += 1
    if dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(batch_count, op=dist.ReduceOp.SUM)
    model.train()
    return float((loss_sum / batch_count.clamp_min(1)).cpu()), int(batch_count.item())


def profile_training_step(
    model,
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    precision: str,
    output_dir: Path,
) -> dict:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_ids, labels = (tensor.to(device) for tensor in batch)
    model.zero_grad(set_to_none=True)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as profiler, torch.profiler.record_function("training_step"):
        with autocast_context(device, precision):
            loss = model(input_ids, labels=labels).loss
        loss.backward()
    trace_path = output_dir / "training_step.json"
    profiler.export_chrome_trace(str(trace_path))
    sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    table_path = output_dir / "training_step.txt"
    table_path.write_text(
        profiler.key_averages(group_by_input_shape=True).table(sort_by=sort_key, row_limit=25) + "\n",
        encoding="utf-8",
    )
    model.zero_grad(set_to_none=True)
    return {"chrome_trace": str(trace_path), "operator_table": str(table_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(MODEL_PRESETS), default="smoke")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="steps run before timing starts; excluded from throughput so "
                             "torch.compile cost is reported separately instead of amortized in")
    parser.add_argument("--batch-size", type=int, default=8, help="per-rank batch size (weak scaling under DDP)")
    parser.add_argument("--global-batch-size", type=int, default=None,
                        help="total batch across ranks; keeps the workload fixed as ranks grow (strong scaling)")
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--train-data", type=Path)
    parser.add_argument("--validation-data", type=Path)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-backend", default=None,
                        help="torch.compile backend; defaults to inductor on CUDA, aot_eager otherwise")
    parser.add_argument(
        "--attn",
        choices=ATTENTION_IMPLEMENTATIONS,
        default=default_attention_implementation(),
        help="attention path to benchmark (default preserves published CPU results)",
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--save-model", type=Path)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "artifacts" / "training_benchmark.json"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if min(args.steps, args.batch_size, args.sequence_length, args.max_samples) <= 0:
        raise SystemExit("steps, batch size, sequence length, and max samples must be positive")
    if args.warmup_steps < 0:
        raise SystemExit("warmup steps must be non-negative")
    if args.precision == "fp16" and args.device == "cpu":
        raise SystemExit("fp16 training is not supported on CPU; use fp32 or bf16")

    torch.manual_seed(2026)
    device, rank, _, world_size = setup_distributed(args.device)
    compile_backend = args.compile_backend or default_compile_backend(device)

    # Weak scaling (the default) grows the global batch with the rank count.
    # --global-batch-size pins total work instead, which is what a speedup claim
    # against a single rank actually requires.
    if args.global_batch_size is not None:
        if args.global_batch_size <= 0:
            raise SystemExit("global batch size must be positive")
        if args.global_batch_size % world_size != 0:
            raise SystemExit(
                f"global batch size {args.global_batch_size} is not divisible by world size {world_size}"
            )
        per_rank_batch_size = args.global_batch_size // world_size
        scaling_mode = "strong"
    else:
        per_rank_batch_size = args.batch_size
        scaling_mode = "weak"
    global_batch_size = per_rank_batch_size * world_size

    tokenizer = load_tokenizer()
    train_texts = (
        load_jsonl_texts(args.train_data, args.text_field, args.max_samples)
        if args.train_data
        else BUILTIN_TRAIN_TEXTS
    )
    validation_texts = (
        load_jsonl_texts(args.validation_data, args.text_field, args.max_samples)
        if args.validation_data
        else BUILTIN_VALIDATION_TEXTS
    )
    train_dataset = make_dataset(tokenizer, train_texts, args.sequence_length)
    validation_dataset = make_dataset(tokenizer, validation_texts, args.sequence_length)
    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True, seed=2026)
        if world_size > 1
        else None
    )
    validation_sampler = (
        DistributedSampler(validation_dataset, shuffle=False)
        if world_size > 1
        else None
    )
    generator = torch.Generator().manual_seed(2026)
    train_loader = DataLoader(
        train_dataset,
        batch_size=per_rank_batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=per_rank_batch_size,
        sampler=validation_sampler,
        shuffle=False,
    )
    model, preset = build_model(
        args.preset,
        tokenizer,
        device,
        max_position_embeddings=args.sequence_length,
        attention_implementation=args.attn,
    )
    attention = attention_metadata(model, args.attn)
    base_model = model
    compile_seconds = None
    if args.compile:
        started = time.perf_counter()
        model = torch.compile(model, backend=compile_backend, fullgraph=False)
        compile_seconds = time.perf_counter() - started
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
        )
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=args.precision == "fp16" and device.type == "cuda"
    )

    initial_validation_loss, validation_batches = evaluate(
        model, validation_loader, device, args.precision
    )
    model.train()
    losses = []
    step_seconds: list[float] = []
    valid_tokens = 0
    process = psutil.Process()
    iterator = iter(train_loader)
    data_epoch = 0

    def next_batch():
        nonlocal iterator, data_epoch
        try:
            return next(iterator)
        except StopIteration:
            data_epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(data_epoch)
            iterator = iter(train_loader)
            return next(iterator)

    def run_step(batch) -> tuple[float, int]:
        input_ids, labels = batch
        input_ids, labels = input_ids.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, args.precision):
            loss = model(input_ids, labels=labels).loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        return float(loss.detach().cpu()), int((labels[:, 1:] != -100).sum().item())

    # Warmup steps absorb torch.compile tracing and allocator growth so they do
    # not silently inflate the steady-state throughput below.
    first_step_seconds = None
    synchronize(device)
    warmup_started = time.perf_counter()
    for index in range(args.warmup_steps):
        step_start = time.perf_counter()
        run_step(next_batch())
        synchronize(device)
        if index == 0:
            first_step_seconds = time.perf_counter() - step_start
    synchronize(device)
    warmup_seconds = time.perf_counter() - warmup_started if args.warmup_steps else 0.0

    rss_before = process.memory_info().rss
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    for index in range(args.steps):
        step_start = time.perf_counter()
        step_loss, step_tokens = run_step(next_batch())
        # Required for meaningful per-step percentiles on CUDA; a no-op on CPU.
        synchronize(device)
        step_seconds.append(time.perf_counter() - step_start)
        if first_step_seconds is None and index == 0:
            first_step_seconds = step_seconds[0]
        losses.append(step_loss)
        valid_tokens += step_tokens
    synchronize(device)
    training_seconds = time.perf_counter() - started
    measured_tokens = valid_tokens
    if dist.is_initialized():
        elapsed_tensor = torch.tensor(training_seconds, device=device)
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
        training_seconds = float(elapsed_tensor.cpu())
        # Sum the tokens actually processed instead of assuming every rank
        # padded identically.
        token_tensor = torch.tensor(float(valid_tokens), device=device)
        dist.all_reduce(token_tensor, op=dist.ReduceOp.SUM)
        measured_tokens = int(token_tensor.cpu())

    final_validation_loss, _ = evaluate(model, validation_loader, device, args.precision)
    rss_after = process.memory_info().rss
    report = {
        "schema_version": 2,
        "environment": environment_metadata(device),
        "distributed": {
            "backend": dist.get_backend() if dist.is_initialized() else None,
            "world_size": world_size,
            "scaling_mode": scaling_mode,
        },
        "model": model_metadata(base_model, preset),
        "attention": attention,
        "data": {
            "train_samples": len(train_dataset),
            "validation_samples": len(validation_dataset),
            "sequence_length": args.sequence_length,
            "source": "jsonl" if args.train_data else "built-in engineering corpus",
            "corpus_is_toy": args.train_data is None,
        },
        "training": {
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "scaling_mode": scaling_mode,
            "per_rank_batch_size": per_rank_batch_size,
            "global_batch_size": global_batch_size,
            "precision": args.precision,
            "learning_rate": args.learning_rate,
            "loss_first": round(losses[0], 6),
            "loss_final": round(losses[-1], 6),
            "loss_curve": [round(value, 6) for value in losses],
            "seconds": round(training_seconds, 6),
            "tokens_per_second": round(measured_tokens / training_seconds, 3),
            "measured_tokens": measured_tokens,
            "step_seconds_p50_ms": round(percentile(step_seconds, 0.50) * 1000, 3),
            "step_seconds_p95_ms": round(percentile(step_seconds, 0.95) * 1000, 3),
            "step_seconds_mean_ms": round(1000 * sum(step_seconds) / len(step_seconds), 3),
            "per_step_synchronized": True,
            "process_rss_delta_mb": round(max(0, rss_after - rss_before) / 1024**2, 3),
        },
        "validation": {
            "batches": validation_batches,
            "initial_loss": round(initial_validation_loss, 6),
            "final_loss": round(final_validation_loss, 6),
            "initial_perplexity": round(math.exp(min(initial_validation_loss, 20)), 4),
            "final_perplexity": round(math.exp(min(final_validation_loss, 20)), 4),
            # The built-in corpus is 32 repeated sentences; perplexity on it
            # measures that the training loop runs, not that the model is good.
            "perplexity_is_indicative_only": args.train_data is None,
        },
        "compile": {
            "enabled": args.compile,
            "backend": compile_backend if args.compile else None,
            "wrapper_creation_seconds": round(compile_seconds, 6) if compile_seconds is not None else None,
            "first_step_seconds": round(first_step_seconds, 6) if first_step_seconds is not None else None,
            "warmup_seconds": round(warmup_seconds, 6),
        },
    }
    if device.type == "cuda":
        report["training"]["peak_gpu_memory_mb"] = round(
            torch.cuda.max_memory_allocated(device) / 1024**2, 3
        )
    if args.profile and rank == 0:
        report["profiler"] = profile_training_step(
            model,
            next(iter(train_loader)),
            device,
            args.precision,
            args.output.parent / "profiler",
        )
    if rank == 0:
        if args.save_model:
            args.save_model.mkdir(parents=True, exist_ok=True)
            base_model.save_pretrained(args.save_model)
            # Preserve the upstream tokenizer files byte-for-byte. Transformers
            # 5.x otherwise writes internal config fields that 4.x cannot load.
            for tokenizer_file in ("tokenizer.json", "tokenizer_config.json"):
                shutil.copy2(PROJECT_ROOT / "model" / tokenizer_file, args.save_model)
            report["saved_model"] = str(args.save_model.resolve())
        write_json_report(args.output, report)
        print(args.output.read_text(encoding="utf-8"))
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
