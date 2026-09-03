# 简历与面试材料

## 推荐项目标题

**MiniMind Training & Inference Optimization Platform**  
PyTorch / CUDA-ready / FastAPI / Prometheus / Docker / DDP

“平台”和“优化”必须由实现与对照实验支撑。不要写“复现 MiniMind”，也不要把上游已有的 GQA、MoE、SFT、LoRA、DPO 当成个人实现。

## 当前即可使用的简历三条

- 基于 PyTorch 为 MiniMind 实现预分配、可复用的 **Static KV Cache**，消除自回归解码中的逐 token `torch.cat` 重分配；以 logits、greedy output 和 buffer pointer 三层测试验证正确性，CPU 微基准相较无缓存提升 **74.1% tokens/s**、降低 **42.3% p95 latency**。
- 开发 OpenAI-compatible FastAPI 推理服务，引入有界队列、按采样参数分组的动态批处理、并发安全 LRU 模型缓存及 Prometheus 可观测性；100 请求/并发 8 的控制压测中，相较 batch=1 将吞吐提升 **155.2%**、p95 延迟降低 **54.4%**。
- 搭建覆盖 0.53M/25.76M/63.91M 配置的训练与性能评测框架，统一采集 loss、validation perplexity、TTFT、tokens/s、内存及 `torch.profiler` traces，支持 mixed precision、`torch.compile` 与 DDP；以 **21 项 pytest + Python 3.10/3.12 CI** 验证 Dense/MoE、缓存、批处理、API 与断点续训路径。

## 英文版

- Implemented a preallocated, reusable **Static KV Cache** for MiniMind in PyTorch, eliminating per-token `torch.cat` reallocations during autoregressive decoding; verified logits/output equivalence and stable buffer addresses, improving CPU decoding throughput by **74.1%** and reducing p95 latency by **42.3%** versus no-cache decoding.
- Built an OpenAI-compatible FastAPI inference service with a bounded queue, generation-aware dynamic batching, a concurrency-safe LRU model cache, and Prometheus instrumentation; improved throughput by **155.2%** and reduced p95 latency by **54.4%** versus batch size 1 in a controlled 100-request, concurrency-8 load test.
- Developed reproducible training/profiling harnesses for 0.53M/25.76M/63.91M model presets, capturing validation perplexity, TTFT, tokens/s, memory, and `torch.profiler` traces with mixed-precision, `torch.compile`, and DDP entry points; covered model and serving paths with **21 tests** and a Python 3.10/3.12 CI matrix.

## 指标口径（面试时主动说明）

- 所有已填数字来自 Apple Silicon CPU 上的 0.533M 微型模型，用于机制级 A/B 对照。
- “74.1%”是 Static KV Cache 对无缓存；Static 对上游动态 cache 的吞吐增益为 0.11%、p95 延迟下降 1.12%。这样表达能同时说明 cache 的主要价值和预分配优化的边际价值。
- TTFT 口径包含队列等待、模型缓存、分词和 backend 首次 forward。饱和并发下 10ms 窗口形成更完整的 batch；稀疏并发下 2ms 相较 10ms 将 p95 TTFT 从 13.825ms 降至 4.637ms，因此选作默认值。
- 当前没有宣称完成 64M 训练、CUDA 加速、INT4 或线上生产部署。

## 完成 GPU 实验后替换的版本

只有在 `artifacts/` 中保存原始 JSON、profiler trace、硬件和 checkpoint hash 后，才将第三条替换为：

> Trained a 63.91M-parameter Transformer from scratch with BF16 and DDP on [GPU], reaching validation perplexity [PPL]; used `torch.profiler` to identify [bottleneck] and `torch.compile` to improve training throughput by [X%] while reducing peak memory by [Y%].

建议 GPU 实验至少包含：同一数据与 token budget、3 次重复、固定 seed、warmup 与同步、P50/P95、峰值显存，以及 eager/compile 与 Dense/MoE 的受控对照。

## 60 秒面试介绍

我没有把项目停留在运行 MiniMind 官方训练脚本，而是把它改造成了一个可测量的 ML Systems 平台。模型侧，我实现了预分配 Static KV Cache，并用数值一致性和 buffer 地址测试证明它不会在逐 token 解码中重分配。系统侧，我写了带有界队列、动态批处理和 LRU 模型缓存的 OpenAI-compatible 服务，同时暴露 Prometheus 指标。最后我用统一 benchmark 测 TTFT、p95、tokens/s、perplexity 和内存。压测里 2ms 动态批处理把吞吐提高了 155.2%，并把 p95 延迟降低 54.4%；我还用稀疏流量实验解释了 batching window 的延迟—吞吐权衡。训练框架已经支持 26M、64M、混合精度、compile 和 DDP，当前公开数字只使用本机 CPU 微模型，避免夸大结果。

## 高频追问

**为什么 Static KV Cache 只比动态 Cache 快 0.11%？**  
两者都避免了重复计算历史 token，主要加速来自“使用 KV Cache”本身。Static Cache 进一步消除 `torch.cat` 和 allocator 开销；微模型计算量很小、测量噪声相对高，因此边际增益有限。在长上下文、较大 batch 和 CUDA Graph/compile 场景下更值得继续验证。

**动态批处理如何保证请求兼容？**  
服务按 model、max tokens、temperature 和 top-p 建立独立 bucket，只合并相同生成设置的请求；队列有容量上限，满载返回 429，并记录 queue depth、wait time 与 batch size。

**为什么 2ms 而不是 10ms？**  
在并发 8 的饱和流量下，10ms 更容易形成完整 batch，峰值吞吐达到 261.3 req/s；在并发 2 的稀疏流量下，它会等满窗口，p95 TTFT 为 13.825ms，而 2ms 配置为 4.637ms。默认选择 2ms，并把 10ms 保留为吞吐优先配置。

**`torch.compile` 和 INT8 结果呢？**  
本机 `aot_eager` 功能验证通过但没有性能收益；动态 INT8 因当前 PyTorch arm64 构建缺少 quantized linear engine 被标记为 skipped。项目保留原始失败原因，不把“有入口”写成“已加速”。
