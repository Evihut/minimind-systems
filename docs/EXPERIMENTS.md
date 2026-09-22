# 实验记录与复现协议

## 已完成实验

| ID | 变量 | 固定条件 | 结论 | 原始报告 |
|---|---|---|---|---|
| INF-CACHE-CPU-001 | no/dynamic/static cache | 0.533M、prompt 32、new 64、greedy、batch 1 | Static 对 no-cache +74.10% tok/s；对 dynamic +0.11% | [`inference_benchmark.json`](../artifacts/inference_benchmark.json) |
| SERVE-BATCH-CPU-001 | batch 1/8、wait 0/2/10ms | 100 请求、并发 8、new 32 | 2ms 对 batch=1：+155.20% req/s，p95 -54.39%，TTFT -73.35% | [`load_test_comparison_2ms.json`](../artifacts/load_test_comparison_2ms.json) |
| SERVE-SPARSE-CPU-001 | wait 2/10ms | 50 请求、并发 2、new 8 | 2ms 对 10ms：p95 -51.10%，TTFT -66.46% | [`load_test_sparse_comparison.json`](../artifacts/load_test_sparse_comparison.json) |
| TRAIN-SMOKE-CPU-001 | 40-step optimization | 0.533M、seq 64、global batch 8、FP32 | train loss 8.7857→1.5036；validation PPL 6552.0→2630.7 | [`training_benchmark.json`](../artifacts/training_benchmark.json) |
| COMPILE-CPU-001 | eager/aot_eager | 同一 smoke decode | 正确性通过；本机无加速，不用于简历收益数字 | [`inference_feature_check.json`](../artifacts/inference_feature_check.json) |
| COMPILE-4090-001 | eager/Inductor | 63.91M、BF16、SDPA、同一 workload | 稳态 tok/s +64.16%，p50 step -39.83%，peak VRAM -28.75%；约 1,229 steps 回本 | [`GPU report`](GPU_RESULTS.md) |
| CACHE-4090-001 | dynamic/static cache | context 128/512/1024、batch 1/8、BF16 | Static peak VRAM 最多 -16.26%；吞吐 -4.36%～+1.79%；6/6 correctness 通过 | [`GPU report`](GPU_RESULTS.md) |
| PIPELINE-4090-001 | 500-step real-corpus learning check | 63.91M、seq 512、batch 8、独立 validation | held-out loss 8.9248→6.0640；不作收敛或聊天能力声明 | [`pretrain.json`](../artifacts/gpu/pipeline/pretrain.json) |

CPU profiler operator table 位于 [`artifacts/profiler/`](../artifacts/profiler/)，GPU 表位于 [`artifacts/gpu/profile/profiler/`](../artifacts/gpu/profile/profiler/)。原始 Chrome Trace 含机器路径且较大，仅在本地保存；可用 benchmark 的 `--profile` 参数重新生成。

## 测量规则

- 报告环境、模型参数量、precision、batch、prompt/output length、warmup 和重复次数。
- 加速前后使用同一输入、seed 和生成策略，并校验 token output 一致。
- accelerator 测量前后同步；CUDA 显存先 reset peak stats，再读取 max allocated。
- latency 报告分位数而非只报平均值；服务压测同时报告成功/失败请求。
- 结论只引用版本化 JSON。任何手动截图、单次峰值或不同 workload 间的数字不做横向对比。

## 后续 GPU 实验矩阵

| 实验 | 对照 | 主指标 | 约束 |
|---|---|---|---|
| DDP-SCALE-GPU | 1 vs 2/4 GPU | speedup、scaling efficiency | 固定 per-GPU batch 与 global tokens |
| QUANT-GPU | BF16/INT8/INT4 | tok/s、VRAM、PPL/accuracy | 同 eval set 与 generation params |
| DENSE-MOE-GPU | Dense vs MoE | PPL、tok/s、activated params | 同训练 FLOPs/token budget |
| SERVE-MIXED-GPU | 动态 batch + mixed lengths | req/s、client TTFT、p95/p99、waste ratio | 固定到达率与请求分布 |
| TRAIN-LONG-GPU | 更长真实语料训练 | held-out PPL、下游 eval、cost | 固定数据 revision 与 token budget |

4090 单次矩阵是可复现的工程证据，不是统计结论。后续性能结论应至少做 3 次重复并保存均值/标准差；模型能力实验应固定 seed、数据 revision、token budget、checkpoint hash 与评测集。
