"""Compare two load-test reports and emit resume-safe deltas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def percent_change(baseline: float, optimized: float) -> float:
    return (optimized / baseline - 1) * 100 if baseline else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("optimized", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/load_test_comparison.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))["summary"]
    optimized = json.loads(args.optimized.read_text(encoding="utf-8"))["summary"]
    report = {
        "schema_version": 1,
        "baseline": str(args.baseline),
        "optimized": str(args.optimized),
        "requests_per_second_improvement_percent": round(
            percent_change(baseline["requests_per_second"], optimized["requests_per_second"]), 2
        ),
        "generated_tokens_per_second_improvement_percent": round(
            percent_change(
                baseline["generated_tokens_per_second"],
                optimized["generated_tokens_per_second"],
            ),
            2,
        ),
        "p95_latency_reduction_percent": round(
            -percent_change(baseline["latency_p95_ms"], optimized["latency_p95_ms"]), 2
        ),
        "p95_ttft_change_percent": round(
            percent_change(baseline["ttft_p95_ms"], optimized["ttft_p95_ms"]), 2
        ),
        "baseline_summary": baseline,
        "optimized_summary": optimized,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

