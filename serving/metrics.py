from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

LATENCY_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)


class ServiceMetrics:
    """Prometheus metrics following online-serving and queue instrumentation guidance."""

    def __init__(self, registry: CollectorRegistry | None = None):
        self.registry = registry or CollectorRegistry()
        self.requests = Counter(
            "minimind_requests_total",
            "Completed inference requests",
            ("status",),
            registry=self.registry,
        )
        self.errors = Counter(
            "minimind_errors_total",
            "Inference errors",
            ("type",),
            registry=self.registry,
        )
        self.in_progress = Gauge(
            "minimind_requests_in_progress",
            "Requests currently being processed",
            registry=self.registry,
        )
        self.request_latency = Histogram(
            "minimind_request_latency_seconds",
            "End-to-end request latency",
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.ttft = Histogram(
            "minimind_ttft_seconds",
            "Time to first generated token",
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.generated_tokens = Counter(
            "minimind_generated_tokens_total",
            "Generated output tokens",
            registry=self.registry,
        )
        self.queue_depth = Gauge(
            "minimind_queue_depth",
            "Requests waiting for dynamic batching",
            registry=self.registry,
        )
        self.queue_wait = Histogram(
            "minimind_queue_wait_seconds",
            "Time spent waiting for a compatible batch",
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.batch_size = Histogram(
            "minimind_batch_size",
            "Executed dynamic batch sizes",
            buckets=(1, 2, 4, 8, 16, 32),
            registry=self.registry,
        )
        self.batch_latency = Histogram(
            "minimind_batch_latency_seconds",
            "Dynamic batch processor latency",
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.model_cache = Counter(
            "minimind_model_cache_queries_total",
            "Model-cache queries",
            ("result",),
            registry=self.registry,
        )

    def set_queue_depth(self, depth: int) -> None:
        self.queue_depth.set(depth)

    def observe_queue_wait(self, seconds: float) -> None:
        self.queue_wait.observe(seconds)

    def observe_batch(self, size: int, seconds: float) -> None:
        self.batch_size.observe(size)
        self.batch_latency.observe(seconds)

    def observe_error(self, error_type: str) -> None:
        self.errors.labels(type=error_type).inc()

    def observe_model_cache(self, result: str) -> None:
        self.model_cache.labels(result=result).inc()

    def render(self) -> bytes:
        return generate_latest(self.registry)

