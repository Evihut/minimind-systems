# MiniMind Systems — Personal ML Systems Practice

[![quality](https://github.com/Evihut/minimind-systems/actions/workflows/ci.yml/badge.svg)](https://github.com/Evihut/minimind-systems/actions/workflows/ci.yml)

A personal learning and engineering project built on [jingyaogong/minimind](https://github.com/jingyaogong/minimind), exploring KV-cache management, reproducible benchmarking and batched LLM serving with PyTorch.

这是一个基于 MiniMind 的个人练习与工程扩展作品，不是从零原创的 LLM 框架。上游提供模型和训练基础；本项目新增缓存、评测、服务与自动验证代码。当前结果来自 CPU 微型模型实验，尚未完成 64M GPU 训练，也未验证生产级部署。

## What this project adds

- Preallocated, reusable **Static KV Cache**, with logits/output equivalence and stable-buffer tests.
- Training and inference benchmarks with JSON reports, validation perplexity, latency, throughput, memory and optional profiler traces.
- FastAPI chat-completions endpoint with request batching, queue limits, LRU model loading and Prometheus metrics.
- CPU tests, GitHub Actions workflow, Docker and Prometheus deployment recipes.

The base Dense/MoE Transformer, GQA, dynamic KV cache and training scripts come from MiniMind. Read [UPSTREAM.md](UPSTREAM.md) and [NOTICE](NOTICE) for the contribution boundary.

## Measured results — scope matters

Exploratory measurements on an Apple Silicon CPU with a **532,864-parameter** model:

| Experiment | Comparison | Observation |
|---|---|---|
| Cache decoding | static vs no cache | +74.10% tokens/s |
| Cache allocation change alone | static vs existing dynamic cache | +0.11% tokens/s; not evidence of a robust speedup |
| Batched serving | batch 8 / 2ms vs batch 1; 100 requests, concurrency 8 | +155.20% requests/s, -54.39% p95 latency |

These are short local runs, not statistical guarantees, GPU results or model-quality claims. Cache benchmark latency sums timed model/decode steps; API latency is measured by the client. Server TTFT is an estimate of request-handler arrival to token computation, **not client-observed streaming TTFT**.

Reports: [inference](artifacts/inference_benchmark.json), [training](artifacts/training_benchmark.json), [serving comparison](artifacts/load_test_comparison_2ms.json). See [experiment notes](docs/EXPERIMENTS.md) and the [detailed Chinese overview](README_PORTFOLIO.md).

## Quick start

Use a Python 3.10+ virtual environment from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m portfolio.smoke_pipeline --device cpu --steps 5 --min-loss-reduction 0.1
```

Run benchmarks without overwriting the checked-in evidence:

```bash
python -m benchmarks.inference_benchmark --device cpu --profile --output /tmp/minimind-inference.json
python -m benchmarks.training_benchmark --device cpu --save-model artifacts/models/smoke --output /tmp/minimind-training.json
```

The second command creates a tiny demo model locally; it is not a useful chat model. Weights and raw profiler traces are intentionally not committed.

Serve it:

```bash
MODEL_ROOT=artifacts/models DEVICE=cpu python -m serving.app --host 127.0.0.1 --port 8998
# In another terminal:
python tools/load_test.py --url http://127.0.0.1:8998/v1/chat/completions --model smoke --requests 100 --concurrency 8 --max-tokens 32 --output /tmp/minimind-load.json
```

API discovery: `/docs`, `/health`, `/ready`, `/v1/models`, `/metrics`. Keep the demo local; it has no production authentication or tenancy controls.

## Limitations and next steps

- SSE currently buffers completed output; true token streaming and continuous batching are future work.
- 25.76M/63.91M presets, CUDA, mixed precision and DDP are experimental entry points, not completed GPU studies.
- Local `aot_eager` validation did not demonstrate acceleration; INT8 was skipped because the local quantization backend was unavailable.
- Docker Compose provides a GPU deployment recipe and expects local weights; only its configuration has been checked.
- Planned: real GPU training/evaluation, mixed-length load tests, cancellation/backpressure coverage and repeated performance trials.

## Documentation and attribution

- [Architecture](docs/ARCHITECTURE.md)
- [Experiments](docs/EXPERIMENTS.md)
- [Upstream source and changes](UPSTREAM.md)
- [Archived upstream Chinese introduction](README_UPSTREAM.md) / [English introduction](README_en.md)

Licensed under [Apache-2.0](LICENSE). Upstream source, images and documentation remain credited to MiniMind and its contributors; they are not presented as independent personal achievements.
