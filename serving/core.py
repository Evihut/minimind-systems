from __future__ import annotations

import asyncio
import inspect
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Hashable
from contextlib import suppress
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
KeyT = TypeVar("KeyT", bound=Hashable)
ModelT = TypeVar("ModelT")


class BatcherMetrics(Protocol):
    def set_queue_depth(self, depth: int) -> None: ...
    def observe_queue_wait(self, seconds: float) -> None: ...
    def observe_batch(self, size: int, seconds: float) -> None: ...
    def observe_error(self, error_type: str) -> None: ...


class CacheMetrics(Protocol):
    def observe_model_cache(self, result: str) -> None: ...


@dataclass
class _QueuedItem(Generic[InputT, OutputT]):
    payload: InputT
    future: asyncio.Future[OutputT]
    enqueued_at: float


class AsyncDynamicBatcher(Generic[InputT, OutputT]):
    """Collect compatible requests until max batch size or wait budget is hit."""

    def __init__(
        self,
        processor: Callable[[list[InputT]], Awaitable[list[OutputT]]],
        *,
        max_batch_size: int = 8,
        max_wait_ms: float = 10.0,
        max_queue_size: int = 256,
        metrics: BatcherMetrics | None = None,
    ):
        if max_batch_size <= 0 or max_queue_size <= 0:
            raise ValueError("batch and queue sizes must be positive")
        if max_wait_ms < 0:
            raise ValueError("max_wait_ms must be non-negative")
        self.processor = processor
        self.max_batch_size = max_batch_size
        self.max_wait_seconds = max_wait_ms / 1000
        self.queue: asyncio.Queue[_QueuedItem[InputT, OutputT]] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self.metrics = metrics
        self._worker: asyncio.Task | None = None
        self._closed = False

    @property
    def queue_depth(self) -> int:
        return self.queue.qsize()

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run(), name="minimind-dynamic-batcher")

    async def close(self) -> None:
        self._closed = True
        if self._worker is not None:
            self._worker.cancel()
            with suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None
        while not self.queue.empty():
            item = self.queue.get_nowait()
            if not item.future.done():
                item.future.set_exception(RuntimeError("batcher closed"))
            self.queue.task_done()
        self._update_queue_metric()

    async def submit(self, payload: InputT) -> OutputT:
        if self._closed:
            raise RuntimeError("batcher is closed")
        await self.start()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[OutputT] = loop.create_future()
        item = _QueuedItem(payload=payload, future=future, enqueued_at=loop.time())
        try:
            self.queue.put_nowait(item)
        except asyncio.QueueFull as error:
            if self.metrics:
                self.metrics.observe_error("queue_full")
            raise RuntimeError("inference queue is full") from error
        self._update_queue_metric()
        return await future

    def _update_queue_metric(self) -> None:
        if self.metrics:
            self.metrics.set_queue_depth(self.queue.qsize())

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            first = await self.queue.get()
            batch = [first]
            deadline = loop.time() + self.max_wait_seconds
            while len(batch) < self.max_batch_size:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(), remaining))
                except TimeoutError:
                    break
            self._update_queue_metric()
            now = loop.time()
            if self.metrics:
                for item in batch:
                    self.metrics.observe_queue_wait(now - item.enqueued_at)
            started = time.perf_counter()
            try:
                outputs = await self.processor([item.payload for item in batch])
                if len(outputs) != len(batch):
                    raise RuntimeError(
                        f"batch processor returned {len(outputs)} outputs for {len(batch)} inputs"
                    )
                for item, output in zip(batch, outputs, strict=True):
                    if not item.future.done():
                        item.future.set_result(output)
            except Exception as error:
                if self.metrics:
                    self.metrics.observe_error(type(error).__name__)
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(error)
            finally:
                if self.metrics:
                    self.metrics.observe_batch(len(batch), time.perf_counter() - started)
                for _ in batch:
                    self.queue.task_done()


class DynamicBatchManager(Generic[KeyT, InputT, OutputT]):
    """Maintain one batcher per generation-parameter/model compatibility key."""

    def __init__(
        self,
        processor_factory: Callable[
            [KeyT], Callable[[list[InputT]], Awaitable[list[OutputT]]]
        ],
        *,
        max_batch_size: int = 8,
        max_wait_ms: float = 10.0,
        max_queue_size: int = 256,
        metrics: BatcherMetrics | None = None,
    ):
        self.processor_factory = processor_factory
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.max_queue_size = max_queue_size
        self.metrics = metrics
        self._batchers: dict[KeyT, AsyncDynamicBatcher[InputT, OutputT]] = {}
        self._lock = asyncio.Lock()

    async def submit(self, key: KeyT, payload: InputT) -> OutputT:
        async with self._lock:
            batcher = self._batchers.get(key)
            if batcher is None:
                batcher = AsyncDynamicBatcher(
                    self.processor_factory(key),
                    max_batch_size=self.max_batch_size,
                    max_wait_ms=self.max_wait_ms,
                    max_queue_size=self.max_queue_size,
                    metrics=self.metrics,
                )
                self._batchers[key] = batcher
        return await batcher.submit(payload)

    async def close(self) -> None:
        await asyncio.gather(*(batcher.close() for batcher in self._batchers.values()))
        self._batchers.clear()


class AsyncModelCache(Generic[KeyT, ModelT]):
    """Concurrency-safe LRU model cache with in-flight load de-duplication."""

    def __init__(
        self,
        loader: Callable[[KeyT], ModelT | Awaitable[ModelT]],
        *,
        max_entries: int = 2,
        metrics: CacheMetrics | None = None,
        on_evict: Callable[[ModelT], None] | None = None,
    ):
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.loader = loader
        self.max_entries = max_entries
        self.metrics = metrics
        self.on_evict = on_evict
        self._values: OrderedDict[KeyT, ModelT] = OrderedDict()
        self._inflight: dict[KeyT, asyncio.Task[ModelT]] = {}
        self._lock = asyncio.Lock()

    @property
    def keys(self) -> tuple[KeyT, ...]:
        return tuple(self._values.keys())

    async def _load(self, key: KeyT) -> ModelT:
        value = self.loader(key)
        return await value if inspect.isawaitable(value) else value

    async def get(self, key: KeyT) -> ModelT:
        async with self._lock:
            if key in self._values:
                value = self._values.pop(key)
                self._values[key] = value
                if self.metrics:
                    self.metrics.observe_model_cache("hit")
                return value
            task = self._inflight.get(key)
            if task is None:
                task = asyncio.create_task(self._load(key), name=f"load-model-{key}")
                self._inflight[key] = task
                if self.metrics:
                    self.metrics.observe_model_cache("miss")
            elif self.metrics:
                self.metrics.observe_model_cache("coalesced")

        try:
            value = await task
        except Exception:
            async with self._lock:
                self._inflight.pop(key, None)
            raise
        async with self._lock:
            self._inflight.pop(key, None)
            if key not in self._values:
                self._values[key] = value
                if len(self._values) > self.max_entries:
                    _, evicted = self._values.popitem(last=False)
                    if self.on_evict:
                        self.on_evict(evicted)
            return self._values[key]

    async def clear(self) -> None:
        async with self._lock:
            values = list(self._values.values())
            self._values.clear()
        if self.on_evict:
            for value in values:
                self.on_evict(value)
