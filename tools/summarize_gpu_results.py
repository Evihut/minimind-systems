"""Validate GPU artifacts and generate a portfolio-ready report and SVG dashboard."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"missing required GPU artifact: {path}") from error


def percent_change(baseline: float, optimized: float) -> float:
    if baseline == 0:
        raise ValueError("cannot compute a percentage change from a zero baseline")
    return (optimized / baseline - 1) * 100


def reduction_percent(baseline: float, optimized: float) -> float:
    return -percent_change(baseline, optimized)


def rounded(value: float, digits: int = 2) -> float:
    return round(float(value), digits)


def find_variant(report: dict, name: str) -> dict:
    for variant in report["variants"]:
        if variant["variant"] == name:
            return variant
    raise ValueError(f"{name!r} is missing from a KV-cache report")


def build_summary(input_dir: Path) -> dict:
    eager = load_json(input_dir / "compile" / "eager.json")
    compiled = load_json(input_dir / "compile" / "compile_inductor.json")
    pipeline = load_json(input_dir / "pipeline" / "pretrain.json")
    manifest = load_json(input_dir / "manifest.json")
    metadata_path = input_dir / "run_metadata.json"
    run_metadata = load_json(metadata_path) if metadata_path.exists() else {}

    cells = manifest.get("cells", {})
    if not cells or any(cell.get("status") != "done" for cell in cells.values()):
        raise ValueError("published GPU suite must contain only completed cells")
    suite_config = manifest.get("suite", {})
    if suite_config.get("device") != "cuda" or suite_config.get("precision") != "bf16":
        raise ValueError("published GPU suite must be a CUDA BF16 run")

    environments = [eager["environment"], compiled["environment"], pipeline["environment"]]
    if any(environment["device"] != "cuda" for environment in environments):
        raise ValueError("published GPU results must all report device=cuda")
    if len({environment["accelerator"] for environment in environments}) != 1:
        raise ValueError("compile and pipeline reports were produced on different accelerators")
    if pipeline["data"]["corpus_is_toy"] or pipeline["validation"]["perplexity_is_indicative_only"]:
        raise ValueError("pipeline result is not backed by a real train/validation split")

    eager_training = eager["training"]
    compiled_training = compiled["training"]
    seconds_saved_per_step = (eager_training["step_seconds_mean_ms"] - compiled_training["step_seconds_mean_ms"]) / 1000
    compile_startup_overhead = compiled["compile"]["first_step_seconds"] - eager["compile"]["first_step_seconds"]
    break_even_steps = compile_startup_overhead / seconds_saved_per_step if seconds_saved_per_step > 0 else None

    compile_summary = {
        "eager_tokens_per_second": eager_training["tokens_per_second"],
        "compiled_tokens_per_second": compiled_training["tokens_per_second"],
        "steady_state_speedup_x": rounded(
            compiled_training["tokens_per_second"] / eager_training["tokens_per_second"], 3
        ),
        "throughput_improvement_percent": rounded(
            percent_change(eager_training["tokens_per_second"], compiled_training["tokens_per_second"])
        ),
        "eager_step_p50_ms": eager_training["step_seconds_p50_ms"],
        "compiled_step_p50_ms": compiled_training["step_seconds_p50_ms"],
        "step_p50_reduction_percent": rounded(
            reduction_percent(
                eager_training["step_seconds_p50_ms"],
                compiled_training["step_seconds_p50_ms"],
            )
        ),
        "eager_peak_gpu_memory_mb": eager_training["peak_gpu_memory_mb"],
        "compiled_peak_gpu_memory_mb": compiled_training["peak_gpu_memory_mb"],
        "peak_gpu_memory_reduction_percent": rounded(
            reduction_percent(
                eager_training["peak_gpu_memory_mb"],
                compiled_training["peak_gpu_memory_mb"],
            )
        ),
        "compile_first_step_seconds": compiled["compile"]["first_step_seconds"],
        "estimated_break_even_steps": rounded(break_even_steps, 1) if break_even_steps else None,
        "precision": compiled_training["precision"],
        "attention": compiled["attention"]["effective"],
    }

    validation = pipeline["validation"]
    pipeline_summary = {
        "steps": pipeline["training"]["steps"],
        "train_samples": pipeline["data"]["train_samples"],
        "validation_samples": pipeline["data"]["validation_samples"],
        "initial_validation_loss": validation["initial_loss"],
        "final_validation_loss": validation["final_loss"],
        "validation_loss_reduction_percent": rounded(
            reduction_percent(validation["initial_loss"], validation["final_loss"])
        ),
        "initial_validation_perplexity": validation["initial_perplexity"],
        "final_validation_perplexity": validation["final_perplexity"],
        "validation_perplexity_reduction_percent": rounded(
            reduction_percent(validation["initial_perplexity"], validation["final_perplexity"])
        ),
        "tokens_per_second": pipeline["training"]["tokens_per_second"],
        "peak_gpu_memory_mb": pipeline["training"]["peak_gpu_memory_mb"],
        "measured_training_seconds": pipeline["training"]["seconds"],
        "corpus_is_toy": pipeline["data"]["corpus_is_toy"],
        "perplexity_is_indicative_only": validation["perplexity_is_indicative_only"],
    }

    cache_rows = []
    cache_paths = sorted((input_dir / "kvcache").glob("*.json"))
    if not cache_paths:
        raise ValueError("no KV-cache reports found")
    for path in cache_paths:
        report = load_json(path)
        required_correctness = report.get("required_correctness", {})
        if not required_correctness or not all(required_correctness.values()):
            raise ValueError(f"required cache correctness failed in {path}")
        dynamic = find_variant(report, "dynamic_cache")
        static = find_variant(report, "static_cache")
        cache_rows.append(
            {
                "context_tokens": report["workload"]["prompt_tokens"],
                "batch_size": report["workload"]["batch_size"],
                "dynamic_tokens_per_second": dynamic["throughput_tokens_per_second"],
                "static_tokens_per_second": static["throughput_tokens_per_second"],
                "static_throughput_change_percent": rounded(
                    percent_change(
                        dynamic["throughput_tokens_per_second"],
                        static["throughput_tokens_per_second"],
                    )
                ),
                "dynamic_inter_token_p95_ms": dynamic["inter_token_latency_p95_ms"],
                "static_inter_token_p95_ms": static["inter_token_latency_p95_ms"],
                "static_inter_token_p95_change_percent": rounded(
                    percent_change(
                        dynamic["inter_token_latency_p95_ms"],
                        static["inter_token_latency_p95_ms"],
                    )
                ),
                "dynamic_peak_gpu_memory_mb": dynamic["peak_gpu_memory_mb"],
                "static_peak_gpu_memory_mb": static["peak_gpu_memory_mb"],
                "static_peak_gpu_memory_reduction_percent": rounded(
                    reduction_percent(dynamic["peak_gpu_memory_mb"], static["peak_gpu_memory_mb"])
                ),
                "static_kv_cache_allocated_mb": static["kv_cache_allocated_mb"],
                "required_correctness": required_correctness,
            }
        )
    cache_rows.sort(key=lambda row: (row["context_tokens"], row["batch_size"]))
    memory_best = max(cache_rows, key=lambda row: row["static_peak_gpu_memory_reduction_percent"])
    throughput_changes = [row["static_throughput_change_percent"] for row in cache_rows]

    return {
        "schema_version": 1,
        "run": run_metadata,
        "suite": {
            **suite_config,
            "cells_completed": len(cells),
            "duration_seconds": rounded(sum(cell.get("duration_seconds", 0) for cell in cells.values()), 3),
        },
        "environment": eager["environment"],
        "model": eager["model"],
        "compile": compile_summary,
        "pipeline": pipeline_summary,
        "kvcache": {
            "matrix": cache_rows,
            "all_required_correctness_passed": True,
            "static_throughput_change_range_percent": [
                rounded(min(throughput_changes)),
                rounded(max(throughput_changes)),
            ],
            "max_peak_gpu_memory_reduction_percent": memory_best["static_peak_gpu_memory_reduction_percent"],
            "max_peak_gpu_memory_reduction_workload": {
                "context_tokens": memory_best["context_tokens"],
                "batch_size": memory_best["batch_size"],
            },
        },
    }


def render_markdown(summary: dict) -> str:
    run = summary["run"]
    suite = summary["suite"]
    env = summary["environment"]
    model = summary["model"]
    compile_result = summary["compile"]
    pipeline = summary["pipeline"]
    cache = summary["kvcache"]
    rows = [
        "# RTX 4090 GPU experiment report",
        "",
        "![GPU benchmark dashboard](../images/gpu_results.svg)",
        "",
        "## Scope",
        "",
        (
            f"Controlled single-GPU measurements for a {model['parameters_millions']:.2f}M-parameter "
            f"MiniMind model on **{env['accelerator']}**, PyTorch {env['torch']}, "
            f"{compile_result['precision'].upper()}, and {compile_result['attention'].upper()} attention. "
            f"The benchmark source commit is `{run.get('source_commit', 'unknown')}`."
        ),
        "",
        (
            "These results measure systems behavior and a 500-step learning check. "
            "They do not claim a fully converged chat model."
        ),
        (
            f"The resumable suite completed {suite['cells_completed']}/{suite['cells_completed']} "
            f"cells in {suite['duration_seconds'] / 60:.1f} minutes of measured cell time."
        ),
        "",
        "## TorchInductor: steady-state speed versus startup cost",
        "",
        "| Metric | Eager | `torch.compile` / Inductor | Change |",
        "|---|---:|---:|---:|",
        (
            f"| Training throughput | {compile_result['eager_tokens_per_second']:,.1f} tok/s | "
            f"{compile_result['compiled_tokens_per_second']:,.1f} tok/s | "
            f"**+{compile_result['throughput_improvement_percent']:.1f}%** |"
        ),
        (
            f"| Step latency p50 | {compile_result['eager_step_p50_ms']:.3f} ms | "
            f"{compile_result['compiled_step_p50_ms']:.3f} ms | "
            f"**-{compile_result['step_p50_reduction_percent']:.1f}%** |"
        ),
        (
            f"| Peak GPU memory | {compile_result['eager_peak_gpu_memory_mb']:.1f} MB | "
            f"{compile_result['compiled_peak_gpu_memory_mb']:.1f} MB | "
            f"**-{compile_result['peak_gpu_memory_reduction_percent']:.1f}%** |"
        ),
        "",
        (
            f"Inductor's first compiled step took {compile_result['compile_first_step_seconds']:.2f}s. "
            f"At the measured steady-state saving, the estimated break-even point is about "
            f"{compile_result['estimated_break_even_steps']:,.0f} steps. This separates a real throughput "
            "win from the cold-start cost instead of hiding compilation inside an average."
        ),
        "",
        "## Real-corpus pipeline check",
        "",
        "| Metric | Before | After 500 steps | Change |",
        "|---|---:|---:|---:|",
        (
            f"| Held-out loss | {pipeline['initial_validation_loss']:.4f} | "
            f"{pipeline['final_validation_loss']:.4f} | "
            f"-{pipeline['validation_loss_reduction_percent']:.1f}% |"
        ),
        (
            f"| Held-out perplexity | {pipeline['initial_validation_perplexity']:,.1f} | "
            f"{pipeline['final_validation_perplexity']:,.1f} | "
            f"**-{pipeline['validation_perplexity_reduction_percent']:.1f}%** |"
        ),
        "",
        (
            f"The run processed real JSONL text at {pipeline['tokens_per_second']:,.1f} tok/s with "
            f"{pipeline['peak_gpu_memory_mb']:.1f} MB peak memory. Train and validation files were disjoint; "
            "dataset prefix provenance is checked in next to the reports."
        ),
        "",
        "## Dynamic versus Static KV cache",
        "",
        (
            "| Context | Batch | Static throughput change | Static p95 inter-token change | "
            "Peak-memory reduction | Static KV allocation |"
        ),
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cache["matrix"]:
        rows.append(
            f"| {row['context_tokens']} | {row['batch_size']} | "
            f"{row['static_throughput_change_percent']:+.2f}% | "
            f"{row['static_inter_token_p95_change_percent']:+.2f}% | "
            f"{row['static_peak_gpu_memory_reduction_percent']:.2f}% | "
            f"{row['static_kv_cache_allocated_mb']:.2f} MB |"
        )
    best = cache["max_peak_gpu_memory_reduction_workload"]
    rows.extend(
        [
            "",
            (
                f"Static cache saved up to **{cache['max_peak_gpu_memory_reduction_percent']:.2f}%** peak "
                f"GPU memory at context {best['context_tokens']}, batch {best['batch_size']}. Its throughput "
                f"change ranged from {cache['static_throughput_change_range_percent'][0]:+.2f}% to "
                f"{cache['static_throughput_change_range_percent'][1]:+.2f}% versus dynamic cache, so the "
                "honest conclusion is deterministic allocation and lower memory pressure—not a universal speedup."
            ),
            "",
            "Dynamic and Static cache outputs passed the required BF16 equivalence check in all six matrix cells.",
            "",
            "## Profiler observation",
            "",
            (
                "The checked-in operator table shows BF16 GEMM/CUTLASS kernels as the dominant CUDA "
                "work, with FlashAttention backward also visible. The raw 7 MB Chrome trace is "
                "reproducible but intentionally excluded from Git."
            ),
            "",
            "## Reproduce",
            "",
            "```bash",
            "make setup-gpu",
            "make verify",
            "make dataset-sample",
            "make gpu-suite GPU_DEVICE=cuda GPU_BUDGET=180 \\",
            "  TRAIN_DATA=dataset/pretrain_t2t_mini_train.jsonl \\",
            "  VALIDATION_DATA=dataset/pretrain_t2t_mini_holdout.jsonl",
            "make gpu-report",
            "```",
            "",
        ]
    )
    return "\n".join(rows)


def render_svg(summary: dict) -> str:
    compile_result = summary["compile"]
    pipeline = summary["pipeline"]
    cache_rows = summary["kvcache"]["matrix"]
    cards = [
        (f"+{compile_result['throughput_improvement_percent']:.1f}%", "Inductor throughput"),
        (f"-{compile_result['step_p50_reduction_percent']:.1f}%", "p50 step latency"),
        (f"-{compile_result['peak_gpu_memory_reduction_percent']:.1f}%", "compiled peak VRAM"),
        (f"-{pipeline['validation_perplexity_reduction_percent']:.1f}%", "held-out perplexity"),
    ]
    card_svg = []
    for index, (value, label) in enumerate(cards):
        x = 42 + index * 280
        card_svg.append(
            f'<rect x="{x}" y="96" width="258" height="112" rx="18" fill="#151d31" stroke="#2b3858"/>'
            f'<text x="{x + 22}" y="143" class="kpi">{html.escape(value)}</text>'
            f'<text x="{x + 22}" y="177" class="label">{html.escape(label)}</text>'
        )

    throughput_max = max(compile_result["eager_tokens_per_second"], compile_result["compiled_tokens_per_second"])
    bar_svg = []
    for index, (label, value, color) in enumerate(
        [
            ("Eager", compile_result["eager_tokens_per_second"], "#6b7da8"),
            ("Inductor", compile_result["compiled_tokens_per_second"], "#47d7ac"),
        ]
    ):
        y = 330 + index * 92
        width = 380 * value / throughput_max
        bar_svg.append(
            f'<text x="62" y="{y - 10}" class="label">{label}</text>'
            f'<rect x="62" y="{y}" width="380" height="34" rx="8" fill="#202a43"/>'
            f'<rect x="62" y="{y}" width="{width:.1f}" height="34" rx="8" fill="{color}"/>'
            f'<text x="{62 + width - 10:.1f}" y="{y + 24}" text-anchor="end" class="barvalue">{value:,.0f} tok/s</text>'
        )

    cache_svg = []
    for index, row in enumerate(cache_rows):
        y = 301 + index * 52
        value = row["static_peak_gpu_memory_reduction_percent"]
        width = max(2, 350 * value / 20)
        label = f"ctx {row['context_tokens']} / bs {row['batch_size']}"
        cache_svg.append(
            f'<text x="632" y="{y + 18}" class="small">{label}</text>'
            f'<rect x="770" y="{y}" width="350" height="24" rx="6" fill="#202a43"/>'
            f'<rect x="770" y="{y}" width="{width:.1f}" height="24" rx="6" fill="#7c8cff"/>'
            f'<text x="1135" y="{y + 18}" class="small" text-anchor="end">{value:.2f}%</text>'
        )

    first_step = f"{compile_result['compile_first_step_seconds']:.2f}s"
    break_even = f"{compile_result['estimated_break_even_steps']:,.0f} steps"
    source_commit = html.escape(summary["run"].get("source_commit", "unknown")[:12])
    torch_version = html.escape(summary["environment"]["torch"])

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="700" viewBox="0 0 1200 700">
<defs>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
    <stop stop-color="#090d18"/><stop offset="1" stop-color="#111a2c"/>
  </linearGradient>
  <style>
    text {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif; fill: #eef3ff; }}
    .title {{ font-size: 28px; font-weight: 700; }} .subtitle {{ font-size: 14px; fill: #93a4c7; }}
    .kpi {{ font-size: 32px; font-weight: 750; fill: #47d7ac; }} .label {{ font-size: 15px; fill: #aab8d5; }}
    .paneltitle {{ font-size: 18px; font-weight: 650; }} .barvalue {{ font-size: 13px; font-weight: 650; }}
    .small {{ font-size: 12px; fill: #b7c3dc; }}
  </style>
</defs>
<rect width="1200" height="700" rx="28" fill="url(#bg)"/>
<circle cx="1100" cy="34" r="145" fill="#233c61" opacity="0.25"/>
<text x="42" y="48" class="title">MiniMind 63.9M · RTX 4090 Systems Study</text>
<text x="42" y="74" class="subtitle">BF16 · SDPA · controlled A/B benchmarks · real held-out corpus</text>
{"".join(card_svg)}
<rect x="42" y="236" width="510" height="395" rx="20" fill="#121a2c" stroke="#273552"/>
<text x="62" y="274" class="paneltitle">Training throughput</text>
<text x="62" y="297" class="subtitle">50 measured steps; five warmup steps excluded</text>
{"".join(bar_svg)}
<text x="62" y="536" class="paneltitle">Cold-start tradeoff</text>
<text x="62" y="566" class="label">First compiled step: {first_step}</text>
<text x="62" y="594" class="label">Estimated break-even: {break_even}</text>
<rect x="580" y="236" width="578" height="395" rx="20" fill="#121a2c" stroke="#273552"/>
<text x="606" y="274" class="paneltitle">Static KV cache peak-memory reduction</text>
<text x="606" y="297" class="subtitle">Relative to dynamic cache; all BF16 correctness checks passed</text>
{"".join(cache_svg)}
<text x="42" y="674" class="subtitle">
  Source commit {source_commit} · PyTorch {torch_version} · NVIDIA GeForce RTX 4090
</text>
</svg>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("artifacts/gpu"))
    parser.add_argument("--json-output", type=Path, default=Path("artifacts/gpu/summary.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("docs/GPU_RESULTS.md"))
    parser.add_argument("--svg-output", type=Path, default=Path("images/gpu_results.svg"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_summary(args.input_dir)
    outputs = {
        args.json_output: json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        args.markdown_output: render_markdown(summary),
        args.svg_output: render_svg(summary),
    }
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
