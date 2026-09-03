# 实验记录与复现协议

## 已完成实验

| ID | 变量 | 固定条件 | 结论 | 原始报告 |
|---|---|---|---|---|
| INF-CACHE-CPU-001 | no/dynamic/static cache | 0.533M、prompt 32、new 64、greedy、batch 1 | Static 对 no-cache +74.10% tok/s；对 dynamic +0.11% | [`inference_benchmark.json`](../artifacts/inference_benchmark.json) |
| SERVE-BATCH-CPU-001 | batch 1/8、wait 0/2/10ms | 100 请求、并发 8、new 32 | 2ms 对 batch=1：+155.20% req/s，p95 -54.39%，TTFT -73.35% | [`load_test_comparison_2ms.json`](../artifacts/load_test_comparison_2ms.json) |
| SERVE-SPARSE-CPU-001 | wait 2/10ms | 50 请求、并发 2、new 8 | 2ms 对 10ms：p95 -51.10%，TTFT -66.46% | [`load_test_sparse_comparison.json`](../artifacts/load_test_sparse_comparison.json) |
| TRAIN-SMOKE-CPU-001 | 40-step optimization | 0.533M、seq 64、global batch 8、FP32 | train loss 8.7857→1.5036；validation PPL 6552.0→2630.7 | [`training_benchmark.json`](../artifacts/training_benchmark.json) |
| COMPILE-CPU-001 | eager/aot_eager | 同一 smoke decode | 正确性通过；本机无加速，不用于简历收益数字 | [`inference_feature_check.json`](../artifacts/inference_feature_check.json) |

Profiler operator table 位于 [`artifacts/profiler/`](../artifacts/profiler/)。原始 Chrome Trace 含机器路径且较大，仅在本地保存；可用 benchmark 的 `--profile` 参数重新生成。

## 测量规则

- 报告环境、模型参数量、precision、batch、prompt/output length、warmup 和重复次数。
- 加速前后使用同一输入、seed 和生成策略，并校验 token output 一致。
- accelerator 测量前后同步；CUDA 显存先 reset peak stats，再读取 max allocated。
- latency 报告分位数而非只报平均值；服务压测同时报告成功/失败请求。
- 结论只引用版本化 JSON。任何手动截图、单次峰值或不同 workload 间的数字不做横向对比。

## 下一组 GPU 实验矩阵

| 实验 | 对照 | 主指标 | 约束 |
|---|---|---|---|
| TRAIN-64M-GPU | eager vs compile | train tok/s、step p95、peak VRAM | 同 checkpoint/data/token budget |
| DDP-SCALE-GPU | 1 vs 2/4 GPU | speedup、scaling efficiency | 固定 per-GPU batch 与 global tokens |
| CACHE-LONG-GPU | dynamic vs static | tok/s、p95、VRAM | context 128/512/1024，batch 1/8 |
| QUANT-GPU | BF16/INT8/INT4 | tok/s、VRAM、PPL/accuracy | 同 eval set 与 generation params |
| DENSE-MOE-GPU | Dense vs MoE | PPL、tok/s、activated params | 同训练 FLOPs/token budget |

完成标准：至少 3 个 seed 或 3 次性能重复，保存均值/标准差、原始 JSON、profiler trace、代码 commit、checkpoint hash 与硬件信息。
