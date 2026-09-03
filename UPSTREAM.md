# Upstream provenance

## Source

- Repository: <https://github.com/jingyaogong/minimind>
- Branch: `master`
- Commit: `7a6fddd63a30c06b2fdd5fac4089922b29bc841b`
- Commit date: `2026-08-31T10:08:05Z`
- Imported on: `2026-09-02`
- License: Apache License 2.0 (`LICENSE`)

The source archive was downloaded from GitHub because `git clone` was unavailable in the local network environment. The commit SHA was independently resolved through the GitHub commits API.

## Portfolio-owned implementation

- `model/cache.py`: preallocated, reusable Static KV Cache with capacity and shape checks.
- `model/model_minimind.py`: static-cache integration, backward-compatible sequence-length handling, fail-fast architecture checks, and safer generation arguments.
- `scripts/serve_openai_api.py`, `scripts/api_schema.py`: extracted API response parsing helpers and tests.
- `requirements.txt`, `.gitignore`: API dependencies and publication exclusions for local data, weights and credentials.
- `benchmarks/`: controlled training/inference runners, model presets, DDP/mixed-precision/compile hooks, profiler export, and machine-readable reports.
- `serving/`: bounded queue, generation-aware dynamic batching, concurrency-safe LRU model cache, batched backend, OpenAI-compatible API, and Prometheus metrics.
- `tools/load_test.py`, `tools/compare_load_tests.py`: concurrent client load test and baseline/optimized comparison.
- `portfolio/`: reproducible local train-and-generate smoke pipeline.
- `tests/`: model, cache, MoE, tokenizer, scheduler, sampler, batching, cache, metrics, and backend tests.
- `Dockerfile`, `docker-compose.yml`, `deploy/`: container and Prometheus deployment.
- `.github/workflows/ci.yml`, `Makefile`, `pyproject.toml`: automated quality and reproducibility entry points.
- `README_PORTFOLIO.md`, `docs/`, `agent.md`: project evidence, experiment protocol, resume guidance, and handoff state.

The upstream repository already includes the base Dense/MoE architecture, GQA, dynamic KV cache, Pretrain/SFT/LoRA/DPO/RL scripts, evaluation utilities, and a basic API. Those capabilities must be described as the experimental foundation, not as original portfolio implementation.

## Publication notes

`README.md` is the personal practice-project landing page. The imported Chinese introduction is retained in `README_UPSTREAM.md`; `README_en.md` and `images/` remain upstream reference material. Upstream performance figures and screenshots are not results of the portfolio experiments. `LICENSE` is retained, and `NOTICE` records attribution. Local handoff notes, generated weights and raw machine-specific profiler traces are excluded from Git; summary evidence is included.
