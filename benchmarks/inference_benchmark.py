"""Benchmark no-cache, dynamic-cache, and preallocated static-cache decoding."""

from __future__ import annotations

import argparse
import copy
import statistics
import time
from pathlib import Path
from typing import Literal

import psutil
import torch
from torch import nn

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
from model.cache import StaticKVCache

CacheMode = Literal["none", "dynamic", "static"]


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


@torch.inference_mode()
def greedy_decode(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    cache_mode: CacheMode,
    reusable_cache: StaticKVCache | None = None,
) -> tuple[torch.Tensor, list[float], StaticKVCache | None]:
    generated = input_ids
    past_key_values = None
    static_cache = None
    if cache_mode == "static":
        static_cache = reusable_cache or StaticKVCache(
            num_hidden_layers=model.config.num_hidden_layers,
            max_cache_len=input_ids.shape[1] + max_new_tokens,
            batch_size=input_ids.shape[0],
        )
        static_cache.reset()
        past_key_values = static_cache

    token_latencies = []
    for _ in range(max_new_tokens):
        needs_prefill = past_key_values is None or (
            isinstance(past_key_values, StaticKVCache) and past_key_values.position == 0
        )
        step_input = generated if cache_mode == "none" or needs_prefill else generated[:, -1:]
        synchronize(input_ids.device)
        started = time.perf_counter()
        output = model(
            step_input,
            past_key_values=past_key_values,
            use_cache=cache_mode != "none",
        )
        next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        synchronize(input_ids.device)
        token_latencies.append(time.perf_counter() - started)
        generated = torch.cat((generated, next_token), dim=-1)
        if cache_mode != "none":
            past_key_values = output.past_key_values
    return generated, token_latencies, static_cache


def benchmark_variant(
    name: str,
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    cache_mode: CacheMode,
    warmup: int,
    repeats: int,
    reuse_static_cache: bool = False,
) -> tuple[dict, torch.Tensor]:
    reusable_cache = None
    if reuse_static_cache:
        reusable_cache = StaticKVCache(
            num_hidden_layers=model.config.num_hidden_layers,
            max_cache_len=input_ids.shape[1] + max_new_tokens,
            batch_size=input_ids.shape[0],
        )
    for _ in range(warmup):
        greedy_decode(model, input_ids, max_new_tokens, cache_mode, reusable_cache)

    request_latencies = []
    ttft_latencies = []
    inter_token_latencies = []
    last_output = input_ids
    last_cache = None
    process = psutil.Process()
    rss_before = process.memory_info().rss
    if input_ids.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(input_ids.device)
    for _ in range(repeats):
        last_output, token_latencies, last_cache = greedy_decode(
            model, input_ids, max_new_tokens, cache_mode, reusable_cache
        )
        request_latencies.append(sum(token_latencies))
        ttft_latencies.append(token_latencies[0])
        inter_token_latencies.extend(token_latencies[1:])
    rss_after = process.memory_info().rss

    batch_size = input_ids.shape[0]
    total_tokens = repeats * max_new_tokens * batch_size
    total_seconds = sum(request_latencies)
    result = {
        "variant": name,
        "cache_mode": cache_mode,
        "batch_size": batch_size,
        "warmup_requests": warmup,
        "measured_requests": repeats,
        "generated_tokens": total_tokens,
        "throughput_tokens_per_second": round(total_tokens / total_seconds, 3),
        "ttft_p50_ms": round(percentile(ttft_latencies, 0.50) * 1000, 3),
        "ttft_p95_ms": round(percentile(ttft_latencies, 0.95) * 1000, 3),
        "request_latency_p50_ms": round(percentile(request_latencies, 0.50) * 1000, 3),
        "request_latency_p95_ms": round(percentile(request_latencies, 0.95) * 1000, 3),
        "inter_token_latency_p50_ms": round(percentile(inter_token_latencies, 0.50) * 1000, 3),
        "inter_token_latency_p95_ms": round(percentile(inter_token_latencies, 0.95) * 1000, 3),
        "request_latency_mean_ms": round(statistics.mean(request_latencies) * 1000, 3),
        "process_rss_delta_mb": round(max(0, rss_after - rss_before) / 1024**2, 3),
    }
    if input_ids.device.type == "cuda":
        result["peak_gpu_memory_mb"] = round(
            torch.cuda.max_memory_allocated(input_ids.device) / 1024**2, 3
        )
    if last_cache is not None:
        result["kv_cache_allocated_mb"] = round(
            last_cache.stats().allocated_bytes / 1024**2, 6
        )
        result["cache_buffer_reused"] = reuse_static_cache
    return result, last_output


def profile_variant(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    cache_mode: CacheMode,
    trace_path: Path,
) -> dict:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if input_ids.device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as profiler, torch.profiler.record_function(f"decode_{cache_mode}"):
        greedy_decode(model, input_ids, max_new_tokens, cache_mode)
    profiler.export_chrome_trace(str(trace_path))
    sort_key = "self_cuda_time_total" if input_ids.device.type == "cuda" else "self_cpu_time_total"
    table = profiler.key_averages(group_by_input_shape=True).table(sort_by=sort_key, row_limit=20)
    table_path = trace_path.with_suffix(".txt")
    table_path.write_text(table + "\n", encoding="utf-8")
    return {"chrome_trace": str(trace_path), "operator_table": str(table_path)}


def speedup(baseline: dict, optimized: dict) -> dict:
    throughput_ratio = (
        optimized["throughput_tokens_per_second"]
        / baseline["throughput_tokens_per_second"]
    )
    latency_reduction = 1 - (
        optimized["request_latency_p95_ms"] / baseline["request_latency_p95_ms"]
    )
    return {
        "throughput_speedup_x": round(throughput_ratio, 3),
        "throughput_improvement_percent": round((throughput_ratio - 1) * 100, 2),
        "p95_latency_reduction_percent": round(latency_reduction * 100, 2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(MODEL_PRESETS), default="smoke")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prompt", default="MiniMind 是一个小型语言模型。")
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--attn",
        choices=ATTENTION_IMPLEMENTATIONS,
        default=default_attention_implementation(),
        help="attention path to benchmark (default preserves published CPU results)",
    )
    parser.add_argument(
        "--variants",
        default="no_cache,dynamic_cache,static_cache",
        help="comma-separated subset of no_cache,dynamic_cache,static_cache",
    )
    parser.add_argument("--include-compile", action="store_true")
    parser.add_argument("--compile-backend", default=None,
                        help="torch.compile backend; defaults to inductor on CUDA, aot_eager otherwise")
    parser.add_argument("--include-int8", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "artifacts" / "inference_benchmark.json"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if min(args.prompt_tokens, args.max_new_tokens, args.repeats) <= 0 or args.warmup < 0:
        raise SystemExit("token counts and repeats must be positive; warmup must be non-negative")
    torch.manual_seed(2026)
    device = resolve_device(args.device)
    compile_backend = args.compile_backend or default_compile_backend(device)
    tokenizer = load_tokenizer()
    model, preset = build_model(
        args.preset,
        tokenizer,
        device,
        max_position_embeddings=args.prompt_tokens + args.max_new_tokens + 8,
        attention_implementation=args.attn,
    )
    # One prompt repeated across the batch keeps every row the same length, so a
    # batch-size sweep varies only the batch dimension.
    encoded = tokenizer(
        [args.prompt] * args.batch_size,
        max_length=args.prompt_tokens,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)

    all_variants = (
        ("no_cache", "none", False),
        ("dynamic_cache", "dynamic", False),
        ("static_cache", "static", True),
    )
    known = {name for name, _, _ in all_variants}
    selected = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = sorted(set(selected) - known)
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; choose from {sorted(known)}")
    if not selected:
        raise SystemExit("--variants must select at least one variant")

    variants = []
    outputs = {}
    for name, cache_mode, reuse in all_variants:
        if name not in selected:
            continue
        result, output = benchmark_variant(
            name,
            model,
            input_ids,
            args.max_new_tokens,
            cache_mode,
            args.warmup,
            args.repeats,
            reuse_static_cache=reuse,
        )
        variants.append(result)
        outputs[name] = output

    # no_cache is the ground truth when it was measured; otherwise fall back to
    # the first selected variant so a partial matrix still self-checks.
    reference = "no_cache" if "no_cache" in outputs else next(iter(outputs))

    compile_metadata = None
    if args.include_compile:
        compile_started = time.perf_counter()
        compiled_model = torch.compile(model, backend=compile_backend, fullgraph=False)
        _, _ = benchmark_variant(
            "compiled_warmup",
            compiled_model,
            input_ids,
            args.max_new_tokens,
            "dynamic",
            warmup=1,
            repeats=1,
        )
        synchronize(device)
        compile_seconds = time.perf_counter() - compile_started
        result, output = benchmark_variant(
            f"compiled_{compile_backend}",
            compiled_model,
            input_ids,
            args.max_new_tokens,
            "dynamic",
            args.warmup,
            args.repeats,
        )
        variants.append(result)
        outputs[result["variant"]] = output
        compile_metadata = {
            "backend": compile_backend,
            "compile_and_first_request_seconds": round(compile_seconds, 3),
        }

    quantization_metadata = None
    if args.include_int8:
        if device.type != "cpu":
            raise SystemExit("dynamic INT8 benchmark currently requires --device cpu")
        try:
            quantized_model = torch.ao.quantization.quantize_dynamic(
                copy.deepcopy(model), {nn.Linear}, dtype=torch.qint8
            )
            result, output = benchmark_variant(
                "dynamic_cache_int8",
                quantized_model,
                input_ids,
                args.max_new_tokens,
                "dynamic",
                args.warmup,
                args.repeats,
            )
            variants.append(result)
            outputs[result["variant"]] = output
            quantization_metadata = {
                "status": "completed",
                "dtype": "qint8",
                "modules": ["torch.nn.Linear"],
                "output_matches_fp32_greedy": bool(torch.equal(output, outputs[reference])),
            }
        except (RuntimeError, NotImplementedError) as error:
            quantization_metadata = {
                "status": "skipped",
                "reason": str(error),
                "recommendation": "Install torchao and use a build with a supported CPU/GPU quantization backend.",
            }

    correctness = {
        name: bool(torch.equal(output, outputs[reference]))
        for name, output in outputs.items()
    }
    by_name = {variant["variant"]: variant for variant in variants}
    possible_comparisons = (
        ("dynamic_vs_no_cache", "no_cache", "dynamic_cache"),
        ("static_vs_dynamic", "dynamic_cache", "static_cache"),
        ("static_vs_no_cache", "no_cache", "static_cache"),
    )
    report = {
        "schema_version": 2,
        "environment": environment_metadata(device),
        "model": model_metadata(model, preset),
        "attention": attention_metadata(model, args.attn),
        "workload": {
            "prompt_tokens": args.prompt_tokens,
            "max_new_tokens": args.max_new_tokens,
            "warmup_requests": args.warmup,
            "measured_requests": args.repeats,
            "batch_size": args.batch_size,
            "batch_composition": "homogeneous: one prompt repeated across the batch",
            "decoding": "greedy",
        },
        "variants": variants,
        "correctness_reference": reference,
        "correctness_matches_reference": correctness,
        "comparisons": {
            label: speedup(by_name[base], by_name[other])
            for label, base, other in possible_comparisons
            if base in by_name and other in by_name
        },
        "compile": compile_metadata,
        "quantization": quantization_metadata,
    }
    if reference == "no_cache":
        # Retained so reports written before the batch-size axis stay readable.
        report["correctness_matches_no_cache"] = correctness
    if args.profile:
        trace_dir = args.output.parent / "profiler"
        report["profiler"] = {
            mode: profile_variant(
                model,
                input_ids,
                min(args.max_new_tokens, 16),
                mode,
                trace_dir / f"decode_{mode}.json",
            )
            for mode in ("none", "dynamic", "static")
        }
    write_json_report(args.output, report)
    print(args.output.read_text(encoding="utf-8"))
    required_correctness = {
        name: matches
        for name, matches in correctness.items()
        if not name.endswith("_int8")
    }
    return 0 if all(required_correctness.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
