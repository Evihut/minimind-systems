# 系统架构与设计决策

## 训练路径

```mermaid
flowchart LR
  A[JSONL / built-in corpus] --> B[Tokenizer]
  B --> C[DistributedSampler]
  C --> D[Dense or MoE Transformer]
  D --> E[Autocast + backward + optimizer]
  E --> F[Checkpoint]
  D --> G[Validation loss / perplexity]
  D --> H[torch.profiler trace]
  E --> I[tokens/s + peak memory]
```

`benchmarks/training_benchmark.py` 提供 smoke、25.76M 和 63.91M 三档配置。每个 rank 使用独立 `DistributedSampler`；CUDA 下采用 NCCL、CPU 下采用 Gloo。报告记录 world size、global batch、precision、loss curve、验证 PPL、吞吐和峰值内存，避免只展示一张无法复现的 loss 截图。

## Static KV Cache

动态缓存每步执行 `cat([past, current])`，缓存长度越长，累计复制和 allocator 开销越明显。`StaticKVCache` 为每层一次性分配 `[batch, kv_heads, max_length, head_dim]` 的 K/V buffer，prefill/decode 仅原位写入对应 slice。

```text
dynamic: K_1 → cat(K_1, K_2) → cat(K_1..2, K_3) → ...
static:  [________________ max_length ________________]
          ↑ write prefix         ↑ write next token
```

安全约束包括：仅限 inference、batch/head/head_dim 一致、写入不超过 capacity、所有层共享一致的 sequence position。`reset()` 只归零逻辑长度并复用已分配内存。

## 在线推理路径

```mermaid
flowchart LR
  A[OpenAI chat request] --> B[Validation]
  B --> C{Queue full?}
  C -- yes --> D[HTTP 429]
  C -- no --> E[Generation-settings bucket]
  E --> F[2 ms / max 8 batch]
  F --> G[LRU model cache]
  G --> H[Tokenizer + padded batch]
  H --> I[Static KV decode]
  I --> J[JSON or buffered SSE]
  B --> K[Prometheus metrics]
  F --> K
  G --> K
  I --> K
```

### 动态批处理

请求只与完全相同的 `model / max_tokens / temperature / top_p` 合并，避免不同停止条件或采样分布互相污染。worker 在首请求到达后等待最多 2ms，或达到 batch 8 立即执行。队列上限提供背压；关闭服务时取消 worker 并清理待处理任务。

### 模型缓存

`AsyncModelCache` 使用锁保护 LRU 状态，同一模型的并发 cold start 共享一个 in-flight load future，避免重复占用显存。超过容量时淘汰最久未使用实例并允许 CUDA allocator 回收缓存。

### 可观测性

`/metrics` 暴露 request/error/in-progress、request latency、端到端 TTFT、generated tokens、queue depth/wait、batch size/latency 和 model-cache hit/miss/coalesced 指标。TTFT 从请求进入 handler 开始，包含排队、模型查找、分词与 backend 首次 forward；API 另行返回 backend-only TTFT。负载工具从客户端计算 req/s、token/s 和 p50/p95/p99，并汇总服务端 TTFT。

## 自动验证

1. 模型：Dense/MoE 形状与梯度、GQA/RoPE/MoE fail-fast、权重绑定。
2. Cache：full forward、dynamic cache、static cache 的 logits 与 greedy output 一致；buffer pointer 稳定。
3. 训练：scheduler、断点采样器、真实 tokenizer 的端到端反向传播。
4. 服务：并发请求合批、参数隔离、LRU/并发加载合并、Prometheus 指标、真实 batch backend 的 TTFT。
5. 交付：Ruff、65 项 pytest、CI smoke benchmark、Docker Compose 配置检查。

## GPU 验证状态与剩余边界

- 已完成：单卡 RTX 4090、63.91M、BF16/SDPA 的 eager/Inductor 对照、Static/Dynamic KV Cache 矩阵和真实语料 500-step pipeline check；原始证据见 [`GPU_RESULTS.md`](GPU_RESULTS.md)。
- 仍缺少：更长训练后的 C-Eval/C-MMLU/OpenBookQA 等能力指标，以及 checkpoint 质量评估。
- 仍缺少：FP16 对照、DDP scaling efficiency、CUDA Graph 兼容性与动态 shape graph-break 分析。
- INT8/INT4 的效果—速度—显存 Pareto curve。
- 真实在线流量下的 client-observed streaming TTFT、取消、混合长度 batching 和持续高并发稳定性。
