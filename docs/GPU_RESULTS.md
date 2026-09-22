# RTX 4090 GPU experiment report

![GPU benchmark dashboard](../images/gpu_results.svg)

## Scope

Controlled single-GPU measurements for a 63.91M-parameter MiniMind model on **NVIDIA GeForce RTX 4090**, PyTorch 2.5.1+cu124, BF16, and SDPA attention. The benchmark source commit is `84f04c56c31b55df7576fb5d4c8bcc5e9c07df63`.

These results measure systems behavior and a 500-step learning check. They do not claim a fully converged chat model.
The resumable suite completed 9/9 cells in 4.3 minutes of measured cell time.

## TorchInductor: steady-state speed versus startup cost

| Metric | Eager | `torch.compile` / Inductor | Change |
|---|---:|---:|---:|
| Training throughput | 2,534.2 tok/s | 4,160.3 tok/s | **+64.2%** |
| Step latency p50 | 69.906 ms | 42.062 ms | **-39.8%** |
| Peak GPU memory | 2770.0 MB | 1973.7 MB | **-28.8%** |

Inductor's first compiled step took 33.42s. At the measured steady-state saving, the estimated break-even point is about 1,229 steps. This separates a real throughput win from the cold-start cost instead of hiding compilation inside an average.

## Real-corpus pipeline check

| Metric | Before | After 500 steps | Change |
|---|---:|---:|---:|
| Held-out loss | 8.9248 | 6.0640 | -32.0% |
| Held-out perplexity | 7,515.8 | 430.1 | **-94.3%** |

The run processed real JSONL text at 22,830.2 tok/s with 2770.0 MB peak memory. Train and validation files were disjoint; dataset prefix provenance is checked in next to the reports.

## Dynamic versus Static KV cache

| Context | Batch | Static throughput change | Static p95 inter-token change | Peak-memory reduction | Static KV allocation |
|---:|---:|---:|---:|---:|---:|
| 128 | 1 | -1.35% | +0.77% | 0.56% | 2.25 MB |
| 128 | 8 | -1.43% | +3.35% | 0.82% | 18.00 MB |
| 512 | 1 | -4.36% | +28.78% | 2.67% | 6.75 MB |
| 512 | 8 | -1.18% | -7.04% | 10.75% | 54.00 MB |
| 1024 | 1 | +1.79% | -20.41% | 5.51% | 12.75 MB |
| 1024 | 8 | -1.50% | +1.52% | 16.26% | 102.00 MB |

Static cache saved up to **16.26%** peak GPU memory at context 1024, batch 8. Its throughput change ranged from -4.36% to +1.79% versus dynamic cache, so the honest conclusion is deterministic allocation and lower memory pressure—not a universal speedup.

Dynamic and Static cache outputs passed the required BF16 equivalence check in all six matrix cells.

## Profiler observation

The checked-in operator table shows BF16 GEMM/CUTLASS kernels as the dominant CUDA work, with FlashAttention backward also visible. The raw 7 MB Chrome trace is reproducible but intentionally excluded from Git.

## Reproduce

```bash
make setup-gpu
make verify
make dataset-sample
make gpu-suite GPU_DEVICE=cuda GPU_BUDGET=180 \
  TRAIN_DATA=dataset/pretrain_t2t_mini_train.jsonl \
  VALIDATION_DATA=dataset/pretrain_t2t_mini_holdout.jsonl
make gpu-report
```
