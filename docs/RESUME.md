# 简历与面试材料

## 推荐项目标题

**MiniMind Systems — Reproducible LLM Training & Serving**
PyTorch / CUDA / TorchInductor / FastAPI / Prometheus / Docker

不要写“从零实现 LLM 框架”或把上游的 GQA、MoE、SFT、LoRA、DPO 算作个人成果。项目真正有区分度的是：实现系统组件、设计受控实验、保留原始证据，并能解释收益何时成立、何时不成立。

## 中文简历三条

- 为 63.9M 参数 MiniMind 实现预分配、可复用的 **Static KV Cache**，以 logits/greedy output/BF16 数值一致性和 buffer pointer 测试验证正确性；在 RTX 4090 的 context 128–1024、batch 1/8 矩阵中将峰值显存最多降低 **16.3%**，同时如实记录吞吐变化为 -4.4%～+1.8%。
- 构建可恢复、带时间预算和参数失效检查的 CUDA 实验系统，统一采集 tok/s、p50/p95 latency、峰值显存、held-out perplexity 和 profiler 数据；受控 BF16 实验中 TorchInductor 将稳态训练吞吐提升 **64.2%**、p50 step time 降低 **39.8%**、峰值显存降低 **28.8%**，并量化约 1,229 steps 的编译回本点。
- 开发 OpenAI-compatible FastAPI 推理服务，加入有界队列、generation-aware 动态批处理、并发安全 LRU 模型缓存和 Prometheus 指标；100 请求/并发 8 的控制压测中，相较 batch=1 将吞吐提升 **155.2%**、p95 延迟降低 **54.4%**，并以 65 项测试与 Python 3.10/3.12 CI 覆盖模型、缓存、训练与服务路径。

## English resume bullets

- Implemented a preallocated, reusable **Static KV Cache** for a 63.9M-parameter MiniMind model, verifying logits/greedy-output equivalence, BF16 numerical correctness, and stable buffer addresses; reduced peak GPU memory by up to **16.3%** across an RTX 4090 context × batch matrix while reporting the measured throughput tradeoff (-4.4% to +1.8%).
- Built a resumable, budget-capped CUDA experiment harness with parameter-based invalidation and structured JSON evidence for throughput, latency, memory, held-out perplexity, and profiling; improved steady-state BF16 training throughput by **64.2%**, reduced p50 step time by **39.8%**, and cut peak memory by **28.8%** with TorchInductor, including a measured ~1,229-step compile break-even analysis.
- Developed an OpenAI-compatible FastAPI inference service with bounded backpressure, generation-aware dynamic batching, a concurrency-safe LRU model cache, and Prometheus telemetry; improved throughput by **155.2%** and reduced p95 latency by **54.4%** versus batch size 1 in a controlled load test, with 65 tests and Python 3.10/3.12 CI.

## 指标口径

- RTX 4090 数据来自单卡、PyTorch 2.5.1+cu124、BF16、SDPA、63.91M 参数的同机 A/B 实验；原始 JSON 和运行 metadata 已提交。
- “+64.2%”是排除 warmup 后的 Inductor 稳态训练吞吐收益。首次 compiled step 为 33.42s，按每步节省时间估算约 1,229 steps 回本；短任务不一定值得 compile。
- Static Cache 的主要证据是固定分配和显存可控。在六个 GPU workload 中吞吐并未稳定胜过 dynamic cache，因此不要把它写成“推理加速 16%”。
- 500-step 真实语料实验将 held-out loss 从 8.9248 降至 6.0640，只验证数据、训练和验证链路确实学习；不要写成“训练完成 64M LLM”或暗示对话质量。
- CPU 的“74.1%”是 Static Cache 对无缓存；Static 对上游 dynamic cache 只有 +0.11%。它适合在面试追问时解释 cache 的主要收益与预分配的边际收益，不建议占用主简历 bullet。
- 服务端 TTFT 从 handler 到 token computation，不是客户端观测到的真流式首 token；当前 SSE 会在生成完成后缓冲输出。

## 60 秒面试介绍

我把 MiniMind 从一个训练示例扩展成了可测量、可复现的 ML Systems 项目。模型侧，我实现了预分配 Static KV Cache，并用数值一致性和稳定地址测试验证正确性；GPU 矩阵显示它在 context 1024、batch 8 时节省 16.3% 峰值显存，但速度不总是更快。实验侧，我写了支持 Ctrl-C 续跑、时间预算、参数失效和 JSON manifest 的 runner，在 RTX 4090 上测出 TorchInductor 的稳态吞吐提升 64.2%，也把 33 秒冷启动换算成约 1,229 步的回本点。系统侧，我实现了有界队列、动态批处理、LRU 模型缓存和 Prometheus 指标的 OpenAI-compatible 服务。这个项目的重点不是宣称训练出了强模型，而是用正确性门槛和可复现实验解释系统优化的真实边界。

## 高频追问

**为什么 Static KV Cache 不一定更快？**
Dynamic 和 Static 都消除了历史 token 的重复计算。Static 进一步避免增长式拼接并给出确定的容量，但 64M 模型在这张卡上的 allocator 开销不总是主瓶颈；矩阵中它稳定降低长上下文大 batch 的峰值显存，却没有稳定的吞吐收益。这个负结果本身说明实验没有挑数字。

**为什么 `torch.compile` 看起来很快，却要 1,229 步才回本？**
稳态每步节省约几十毫秒，但首次图捕获与 kernel compilation 花了 33.42 秒。训练足够长时收益会累积；短任务、动态 shape 或频繁重编译时，eager 可能更合适。

**500 步训练说明了什么？**
说明真实 JSONL 文本、tokenizer、反向传播和独立 held-out validation 的整条链路有效，并能在付费 GPU 上复现。它不说明模型已经收敛，更不说明具备聊天能力；模型能力需要更长训练和下游评测。

**如何保证付费 GPU 实验不会半途丢失？**
Suite 为每个 cell 写状态、命令和输出清单；只有命令指纹与可解析报告都匹配时才复用结果。SIGINT 会将正在运行的 cell 标记为 interrupted，保留已经完成的 cell，下次从断点继续；到达预算后不再启动新 cell。

**动态批处理如何避免错误合批？**
服务按 model、max tokens、temperature 和 top-p 建立独立 bucket，只合并兼容请求；有界队列在满载时返回 429，模型 cold start 则通过共享 in-flight future 防止并发重复加载。

## 简历布局建议

将本项目放在 Projects 第一项，控制在三条 bullet。第二项优先放已经被上游合并、能由第三方验证的开源贡献。ESG 项目只有在具备可复现数据管道、明确评测和部署成果时才保留；否则宁可用一个强项目加开源贡献，也不要用一个弱项目稀释信号。
