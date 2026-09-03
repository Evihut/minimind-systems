from __future__ import annotations

import asyncio

import torch
from transformers import AutoTokenizer

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from serving.backend import GenerationInput, GenerationSettings, TransformersBackend
from serving.core import AsyncDynamicBatcher, AsyncModelCache, DynamicBatchManager
from serving.metrics import ServiceMetrics


def test_dynamic_batcher_combines_concurrent_requests_and_preserves_order():
    async def scenario():
        observed_batches = []

        async def processor(values):
            observed_batches.append(list(values))
            return [value * 10 for value in values]

        batcher = AsyncDynamicBatcher(processor, max_batch_size=4, max_wait_ms=20)
        outputs = await asyncio.gather(*(batcher.submit(value) for value in range(4)))
        await batcher.close()
        return outputs, observed_batches

    outputs, observed_batches = asyncio.run(scenario())
    assert outputs == [0, 10, 20, 30]
    assert observed_batches == [[0, 1, 2, 3]]


def test_dynamic_batch_manager_separates_incompatible_keys():
    async def scenario():
        calls = []

        def factory(key):
            async def processor(values):
                calls.append((key, list(values)))
                return [f"{key}:{value}" for value in values]

            return processor

        manager = DynamicBatchManager(factory, max_batch_size=8, max_wait_ms=10)
        outputs = await asyncio.gather(
            manager.submit("greedy", 1),
            manager.submit("greedy", 2),
            manager.submit("sample", 3),
        )
        await manager.close()
        return outputs, calls

    outputs, calls = asyncio.run(scenario())
    assert outputs == ["greedy:1", "greedy:2", "sample:3"]
    assert ("greedy", [1, 2]) in calls
    assert ("sample", [3]) in calls


def test_model_cache_coalesces_loads_and_evicts_lru_entry():
    async def scenario():
        loads = []

        async def loader(key):
            loads.append(key)
            await asyncio.sleep(0.01)
            return {"model": key}

        cache = AsyncModelCache(loader, max_entries=2)
        first, duplicate = await asyncio.gather(cache.get("a"), cache.get("a"))
        await cache.get("b")
        await cache.get("c")
        return first, duplicate, loads, cache.keys

    first, duplicate, loads, keys = asyncio.run(scenario())
    assert first is duplicate
    assert loads.count("a") == 1
    assert keys == ("b", "c")


def test_prometheus_metrics_expose_queue_cache_latency_and_errors():
    metrics = ServiceMetrics()
    metrics.set_queue_depth(3)
    metrics.observe_queue_wait(0.002)
    metrics.observe_batch(4, 0.02)
    metrics.observe_model_cache("hit")
    metrics.observe_error("timeout")

    rendered = metrics.render().decode()
    assert "minimind_queue_depth 3.0" in rendered
    assert 'minimind_model_cache_queries_total{result="hit"} 1.0' in rendered
    assert 'minimind_errors_total{type="timeout"} 1.0' in rendered


def test_backend_batches_prompts_and_records_real_ttft():
    async def scenario():
        tokenizer = AutoTokenizer.from_pretrained("model", local_files_only=True)
        config = MiniMindConfig(
            hidden_size=32,
            num_hidden_layers=1,
            vocab_size=len(tokenizer),
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
            flash_attn=False,
        )
        model = MiniMindForCausalLM(config)
        backend = TransformersBackend(model, tokenizer, torch.device("cpu"))
        return await backend.generate_batch(
            [
                GenerationInput(messages=[{"role": "user", "content": "你好"}]),
                GenerationInput(messages=[{"role": "user", "content": "Hello"}]),
            ],
            GenerationSettings(model="tiny", max_tokens=3),
        )

    outputs = asyncio.run(scenario())
    assert len(outputs) == 2
    assert all(output.completion_tokens == 3 for output in outputs)
    assert all(output.ttft_seconds > 0 for output in outputs)
    assert all(output.total_seconds >= output.ttft_seconds for output in outputs)
