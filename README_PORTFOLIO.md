# MiniMind Training & Inference Optimization Platform

> Personal practice / engineering extensions based on MiniMind. See [the repository homepage](README.md) for publication scope and limitations. All measured numbers below are exploratory CPU microbenchmarks, not 64M GPU results or statistically established speedups. Raw traces and generated weights are reproducible local artifacts, not part of this Git upload.

一个基于原生 PyTorch 的小型 LLM 训练、性能分析与在线推理平台。项目以 [MiniMind](https://github.com/jingyaogong/minimind) 为实验底座，个人贡献集中在 **Static KV Cache、可复现实验框架、动态批处理、模型缓存、可观测 API 服务和自动验证**，而不是把运行上游代码包装成个人成果。

## 核心成果

- 实现预分配、可复用的 `StaticKVCache`，避免逐 token 解码时反复 `torch.cat` 和内存重分配；通过完整前向、动态缓存和静态缓存的 logits/生成结果一致性测试。
- 建立训练与推理 benchmark，统一记录 loss、validation perplexity、tokens/s、TTFT、p50/p95 latency、RSS/峰值显存，并可导出 `torch.profiler` Chrome Trace。
- 构建 OpenAI-compatible FastAPI 服务，加入有界请求队列、按生成参数分组的动态批处理、并发安全 LRU 模型缓存、SSE 响应和 Prometheus 指标。
- 提供 25.76M / 63.91M 参数实验预设、BF16/FP16、`torch.compile`、DDP `torchrun` 和动态 INT8 探测入口；未执行的 GPU 实验不会写成已完成成果。
- 建立 21 项 CPU 自动测试、Python 3.10/3.12 CI、训练/推理 smoke benchmark、Docker Compose 和负载测试工具。

## 本机实测

环境：2026-09-02，Apple Silicon macOS，CPU，Python 3.13.5，PyTorch 2.12.1。模型为随机初始化的 0.533M 工程微型配置，结果用于验证优化机制，不代表 64M 模型能力。

### KV Cache 解码

工作负载：32-token prompt，greedy 生成 64 tokens，warmup 2 次、测量 5 次，batch size 1。

| 方案 | tokens/s | p95 请求延迟 | p95 TTFT | 结果一致 |
|---|---:|---:|---:|:---:|
| 无缓存 | 1,932.3 | 33.539 ms | 0.463 ms | ✓ |
| 动态 KV Cache | 3,360.4 | 19.563 ms | 0.473 ms | ✓ |
| Static KV Cache | **3,364.1** | **19.344 ms** | **0.450 ms** | ✓ |

Static Cache 相比无缓存吞吐提升 **74.10%**、p95 延迟下降 **42.32%**；相比动态 Cache 吞吐提升 **0.11%**、p95 延迟下降 1.12%，并通过底层 buffer 地址不变测试证明解码期间未重分配。原始数据见 [`artifacts/inference_benchmark.json`](artifacts/inference_benchmark.json)。

### 并发服务与动态批处理

工作负载：100 个请求、并发 8、每请求生成 32 tokens；同一模型、机器和客户端，仅改变最大 batch 与等待窗口。

| 配置 | req/s | generated tokens/s | p95 延迟 | p95 TTFT |
|---|---:|---:|---:|---:|
| batch=1，0ms | 79.69 | 2,550.0 | 108.077 ms | 95.875 ms |
| batch=8，2ms | 203.36 | 6,507.5 | 49.295 ms | 25.552 ms |
| batch=8，10ms | **261.32** | **8,362.4** | **30.212 ms** | **4.851 ms** |

2ms 配置相较无批处理，吞吐提升 **155.20%**、p95 延迟下降 **54.39%**、p95 TTFT 下降 **73.35%**。饱和流量下 10ms 能形成更完整的 batch，因此三项指标更优；但在并发 2、8-token 的稀疏流量下，2ms 相对 10ms 将 p95 延迟降低 **51.10%**、p95 TTFT 降低 **66.46%**。默认采用 2ms，以牺牲部分峰值吞吐换取稀疏流量响应性。对照报告见 [`artifacts/load_test_comparison_2ms.json`](artifacts/load_test_comparison_2ms.json) 和 [`artifacts/load_test_sparse_comparison.json`](artifacts/load_test_sparse_comparison.json)。

### 训练链路

40-step CPU 控制实验将 train loss 从 8.7857 降至 1.5036，独立 validation perplexity 从 6,552.0 降至 2,630.7，训练吞吐 13,642.8 tokens/s。该实验只证明数据、反向传播、评估、保存和 profiler 链路正确；完整 26M/64M GPU 训练仍列为下一阶段。原始数据见 [`artifacts/training_benchmark.json`](artifacts/training_benchmark.json)。

## 快速验证

```bash
pip install -r requirements-dev.txt
make verify
make benchmark-inference
make benchmark-training
```

启动本地服务并压测：

```bash
MODEL_ROOT=artifacts/models DEVICE=cpu python -m serving.app --port 8998
python tools/load_test.py --url http://127.0.0.1:8998/v1/chat/completions --model smoke \
  --requests 100 --concurrency 8 --max-tokens 32
```

CUDA / DDP 实验入口：

```bash
python -m benchmarks.training_benchmark --preset 64m --device cuda \
  --precision bf16 --steps 1000 --profile

torchrun --standalone --nproc-per-node=2 -m benchmarks.training_benchmark \
  --preset 64m --device cuda --precision bf16 --steps 1000
```

容器部署：

```bash
docker compose up --build
# API: http://localhost:8998  Prometheus: http://localhost:9090
```

## 系统结构

```text
JSONL / built-in corpus
        │
        ▼
Tokenizer → Pretrain / SFT / LoRA / DPO → Checkpoint + validation PPL
        │                                      │
        └──────── training benchmark/profiler ─┘
                                               ▼
OpenAI API → bounded queue → compatibility buckets → dynamic batches
                                               │
                                  LRU model cache → Static KV Cache
                                               │
                     TTFT / latency / throughput / Prometheus / load test
```

更多实现与实验边界见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)、[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)；可直接使用的简历与面试材料见 [`docs/RESUME.md`](docs/RESUME.md)。

## 设计参考

- [PyTorch Profiler / torch.compile profiling](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html)
- [`torch.compile` API](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
- [DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [Prometheus instrumentation best practices](https://prometheus.io/docs/practices/instrumentation/)

## 贡献边界

上游 MiniMind 已提供 Dense/MoE、Pretrain、SFT、LoRA、DPO、PPO/GRPO/CISPO、Tool Use、评测和基础 API。本仓库新增内容及修改文件在 [`UPSTREAM.md`](UPSTREAM.md) 中逐项标注。上游版本固定为 commit `7a6fddd63a30c06b2fdd5fac4089922b29bc841b`，Apache-2.0 License。
