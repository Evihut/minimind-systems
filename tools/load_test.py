"""Concurrent OpenAI-compatible API load generator with percentile reporting."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx


@dataclass
class RequestResult:
    status_code: int
    latency_seconds: float
    completion_tokens: int
    ttft_ms: float | None
    error: str | None = None


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


async def run_request(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    url: str,
    payload: dict,
) -> RequestResult:
    async with semaphore:
        started = time.perf_counter()
        try:
            response = await client.post(url, json=payload)
            latency = time.perf_counter() - started
            data = response.json()
            return RequestResult(
                status_code=response.status_code,
                latency_seconds=latency,
                completion_tokens=int(data.get("usage", {}).get("completion_tokens", 0)),
                ttft_ms=data.get("performance", {}).get("ttft_ms"),
                error=None if response.is_success else str(data),
            )
        except Exception as error:
            return RequestResult(
                status_code=0,
                latency_seconds=time.perf_counter() - started,
                completion_tokens=0,
                ttft_ms=None,
                error=f"{type(error).__name__}: {error}",
            )


async def benchmark(args: argparse.Namespace) -> dict:
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": False,
    }
    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        semaphore = asyncio.Semaphore(args.concurrency)
        for _ in range(args.warmup):
            await run_request(client, semaphore, args.url, payload)
        started = time.perf_counter()
        results = await asyncio.gather(
            *(
                run_request(client, semaphore, args.url, payload)
                for _ in range(args.requests)
            )
        )
        wall_seconds = time.perf_counter() - started

    successful = [result for result in results if 200 <= result.status_code < 300]
    latencies_ms = [result.latency_seconds * 1000 for result in successful]
    ttft_values = [result.ttft_ms for result in successful if result.ttft_ms is not None]
    total_tokens = sum(result.completion_tokens for result in successful)
    return {
        "schema_version": 1,
        "config": {
            "url": args.url,
            "model": args.model,
            "requests": args.requests,
            "concurrency": args.concurrency,
            "warmup": args.warmup,
            "max_tokens": args.max_tokens,
        },
        "summary": {
            "successful_requests": len(successful),
            "failed_requests": len(results) - len(successful),
            "wall_seconds": round(wall_seconds, 4),
            "requests_per_second": round(len(successful) / max(wall_seconds, 1e-9), 3),
            "generated_tokens_per_second": round(total_tokens / max(wall_seconds, 1e-9), 3),
            "latency_mean_ms": round(statistics.mean(latencies_ms), 3) if latencies_ms else 0,
            "latency_p50_ms": round(percentile(latencies_ms, 0.50), 3),
            "latency_p95_ms": round(percentile(latencies_ms, 0.95), 3),
            "latency_p99_ms": round(percentile(latencies_ms, 0.99), 3),
            "ttft_p50_ms": round(percentile(ttft_values, 0.50), 3),
            "ttft_p95_ms": round(percentile(ttft_values, 0.95), 3),
        },
        "results": [asdict(result) for result in results],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8998/v1/chat/completions")
    parser.add_argument("--model", default="minimind-3")
    parser.add_argument("--prompt", default="请简要介绍 MiniMind。")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path, default=Path("artifacts/load_test.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if min(args.requests, args.concurrency, args.max_tokens) <= 0 or args.warmup < 0:
        raise SystemExit("requests, concurrency, and max tokens must be positive")
    report = asyncio.run(benchmark(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output.read_text(encoding="utf-8"))
    return 0 if report["summary"]["failed_requests"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

