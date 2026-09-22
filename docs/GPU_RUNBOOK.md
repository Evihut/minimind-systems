# GPU runbook

A short, budget-capped GPU session that produces the CUDA evidence the CPU
benchmarks cannot. The first completed RTX 4090 run spent 4m19s inside the
nine-cell suite; setup, verification, dataset download, profiling, and artifact
collection made the end-to-end rental roughly half an hour. Treat that as one
observed run, not a guarantee: cold caches, image setup, and provider bandwidth
can vary. Everything below is rehearsed locally first so paid time is not spent
debugging scripts.

## 0. Rehearse locally (free)

```bash
make gpu-suite-rehearse
```

This runs the entire suite on CPU with the `smoke` preset in well under a
minute. It exercises the same code paths the GPU run will use: subprocess
launch, JSON reports, the manifest, resume, and the skip logic. Fix anything
that fails here, not on a metered box. CPU rehearsal uses `aot_eager` because
macOS Inductor is not a reliable prerequisite; a CUDA plan defaults to the
real `inductor` backend.

Preview the compute-only plan without running it. The pipeline cell remains
skipped until the train and holdout files are supplied in step 2:

```bash
make gpu-suite-plan GPU_DEVICE=cuda
```

## 1. Prepare the dataset before renting

The `pipeline` group needs a real corpus. Without `--train-data` it is
**skipped on purpose** — the built-in corpus is 32 repeated sentences, and its
perplexity measures only that the training loop runs. Every report carries
`data.corpus_is_toy` and `validation.perplexity_is_indicative_only` so a number
from the toy corpus can never be quoted as a model-quality result by accident.

### Where the data lives

MiniMind's corpus is on ModelScope at
[`gongjy/minimind_dataset`](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files)
(HuggingFace mirror: `jingyaogong/minimind_dataset`). Verified file sizes:

| File | Size | Use |
|---|---|---|
| `pretrain_t2t_mini.jsonl` | 1.18 GB | pretrain, the variant the upstream trainer defaults to |
| `sft_t2t_mini.jsonl` | 1.66 GB | SFT |
| `pretrain_t2t.jsonl` | 7.89 GB | full pretrain, too large for a short rental |
| `sft_t2t.jsonl` | 13.44 GB | full SFT |
| `dpo.jsonl` | 51 MB | DPO |

Records are one JSON object per line with a single `text` field, which is the
`--text-field` default.

### Fetch only what the benchmark reads

A short benchmark reads a few thousand lines, so downloading 1.18 GB is wasted
rental time. ModelScope serves range requests, and this pulls just the prefix:

```bash
make dataset-sample                        # 20000 train + 2000 holdout, ~26 MB
make dataset-sample SAMPLES=4096 HOLDOUT=512
```

Records average ~775 bytes, so 22000 records costs roughly 26 MB instead of
1.18 GB. The tool truncates at the last complete line (a range request routinely
cuts mid-character in CJK text), splits off the holdout, and writes a
`*_provenance.json` recording the source URL, revision, byte count and SHA-256
of the downloaded prefix — so a result can name exactly which bytes produced it.

It writes into `dataset/`:

```
dataset/pretrain_t2t_mini_train.jsonl
dataset/pretrain_t2t_mini_holdout.jsonl
dataset/pretrain_t2t_mini_provenance.json
```

If ModelScope is unreachable, download the file from the web UI and pass it
with `--train-data` directly; the tool prints that fallback on failure.

## 2. On the rented box

After cloning the branch, install the project and verification dependencies.
This also installs `pytest` and `ruff`; they are required by `make verify` and
must not be assumed to exist in a stock GPU image.

```bash
make setup-gpu
make verify
make gpu-suite-rehearse
make dataset-sample
```

Preview the paid plan with both sides of the train/validation split. The
pipeline cell refuses to run if either file is missing or both paths identify
the same file.

```bash
make gpu-suite-plan GPU_DEVICE=cuda \
  TRAIN_DATA=dataset/pretrain_t2t_mini_train.jsonl \
  VALIDATION_DATA=dataset/pretrain_t2t_mini_holdout.jsonl
```

Then run with a three-hour launch budget:

```bash
make gpu-suite GPU_DEVICE=cuda GPU_BUDGET=180 \
  TRAIN_DATA=dataset/pretrain_t2t_mini_train.jsonl \
  VALIDATION_DATA=dataset/pretrain_t2t_mini_holdout.jsonl
```

The suite stops launching new cells once the budget is reached, so an overrun
costs nothing beyond the cell already running. Everything lands under
`artifacts/gpu/`:

```
artifacts/gpu/
  manifest.json          per-cell status, command, duration
  compile/*.json         eager vs inductor
  kvcache/*.json         context x batch matrix
  pipeline/*.json        short pretrain with held-out eval
  logs/*.log             full stdout per cell
```

After the run, validate the evidence and regenerate the human-readable report:

```bash
make gpu-report
```

The committed reference run is summarized in [`GPU_RESULTS.md`](GPU_RESULTS.md).

**If the connection drops or you Ctrl-C**, just re-run the same command. Cells
already recorded as `done` are reused; a cell is only reused when the same
command produced a report that still parses, so changing a parameter re-runs it
instead of passing off a stale number.

## 3. What each group answers

| Group | Question | Key fields |
|---|---|---|
| `compile` | Does Inductor beat eager on a 64M model, and what training startup cost does it add? | `training.tokens_per_second`, `training.step_seconds_p50_ms` / `p95`, `compile.first_step_seconds`, `compile.warmup_seconds`, `training.peak_gpu_memory_mb` |
| `kvcache` | How do dynamic and static caches compare as context and batch grow? | `throughput_tokens_per_second`, `ttft_p50_ms`, `inter_token_latency_p95_ms`, `kv_cache_allocated_mb`, `peak_gpu_memory_mb` |
| `pipeline` | Does the training path actually learn on real text? | `validation.initial_loss` vs `final_loss`, `corpus_is_toy: false` |

Initial validation runs before the model is compiled. Warmup steps are excluded
from the timed window, so Inductor's training compile cost shows up in
`compile.first_step_seconds` instead of quietly depressing the steady-state
throughput average. The `kvcache` matrix runs `no_cache` only in the smallest
cell; Dynamic and Static Cache equivalence remains a required check in BF16 at
every point.

Inference parameters are loaded in the requested deployment dtype rather than
kept in FP32 behind per-token autocast. Reports record both `precision` and
`parameter_dtype`. The SDPA path supports both prefill and one-token cached
decode; chunked cached decode still falls back to the manual path.

## 4. DDP (optional, lowest priority)

Only if the single-card numbers are stable and budget remains:

```bash
python -m benchmarks.gpu_suite --device cuda --groups ddp \
  --ddp-ranks 1 2 --global-batch-size 16 --time-budget-minutes 60
```

This is **strong scaling**: `--global-batch-size` is held fixed and divided
across ranks, so a 1-rank and a 2-rank run do the same total work. The older
`--batch-size` flag is per-rank and grows the global batch with the rank count;
every report states which mode produced it in `training.scaling_mode`.

When writing up DDP results, state the interconnect. Consumer cards on these
hosts talk over PCIe without NVLink, so the scaling number describes that
topology and should not be presented as a general multi-GPU result.

## 5. Reporting discipline

- Quote only what a report file contains. Throughput measured in one
  configuration is not a claim about training convergence or model quality.
- Always state the attention path: every report records `attention.requested`
  and `attention.effective`, because `sdpa` silently degrades to the manual
  path on builds without `scaled_dot_product_attention`.
- CPU results published earlier were measured on the manual attention path,
  which is still the default. GPU cells opt into `sdpa` explicitly, so the two
  sets are not directly comparable — say so rather than putting them in one
  table.
