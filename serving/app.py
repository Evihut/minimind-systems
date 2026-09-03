"""OpenAI-compatible MiniMind API with batching, model caching, and metrics."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from serving.backend import (
    GenerationInput,
    GenerationOutput,
    GenerationSettings,
    TransformersBackend,
)
from serving.core import AsyncModelCache, DynamicBatchManager
from serving.metrics import ServiceMetrics


@dataclass(frozen=True)
class ServiceConfig:
    model_root: Path = Path("/models")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_batch_size: int = 8
    max_wait_ms: float = 2.0
    max_queue_size: int = 256
    model_cache_size: int = 2


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict]
    max_tokens: int = Field(default=128, ge=1, le=8192)
    temperature: float = Field(default=0.0, ge=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    stream: bool = False


def _safe_model_path(model_root: Path, model_id: str) -> Path:
    if not model_id or model_id.startswith(("/", ".")):
        raise ValueError("invalid model id")
    root = model_root.resolve()
    candidate = (root / model_id).resolve()
    if root not in candidate.parents:
        raise ValueError("model id escapes MODEL_ROOT")
    if not candidate.is_dir():
        raise FileNotFoundError(f"model {model_id!r} is not available")
    return candidate


def create_app(config: ServiceConfig | None = None) -> FastAPI:
    config = config or ServiceConfig(
        model_root=Path(os.environ.get("MODEL_ROOT", "/models")),
        device=os.environ.get(
            "DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
        ),
        max_batch_size=int(os.environ.get("MAX_BATCH_SIZE", "8")),
        max_wait_ms=float(os.environ.get("MAX_BATCH_WAIT_MS", "2")),
        max_queue_size=int(os.environ.get("MAX_QUEUE_SIZE", "256")),
        model_cache_size=int(os.environ.get("MODEL_CACHE_SIZE", "2")),
    )
    metrics = ServiceMetrics()
    device = torch.device(config.device)

    async def load_backend(model_id: str) -> TransformersBackend:
        try:
            path = _safe_model_path(config.model_root, model_id)
        except (ValueError, FileNotFoundError) as error:
            raise RuntimeError(str(error)) from error
        return await asyncio.to_thread(
            TransformersBackend.from_pretrained, path, device
        )

    def evict_backend(backend: TransformersBackend) -> None:
        del backend
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model_cache = AsyncModelCache(
        load_backend,
        max_entries=config.model_cache_size,
        metrics=metrics,
        on_evict=evict_backend,
    )

    def processor_factory(settings: GenerationSettings):
        async def process(inputs: list[GenerationInput]) -> list[GenerationOutput]:
            backend = await model_cache.get(settings.model)
            return await backend.generate_batch(inputs, settings)

        return process

    batch_manager = DynamicBatchManager(
        processor_factory,
        max_batch_size=config.max_batch_size,
        max_wait_ms=config.max_wait_ms,
        max_queue_size=config.max_queue_size,
        metrics=metrics,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await batch_manager.close()
        await model_cache.clear()

    app = FastAPI(
        title="MiniMind Training & Inference Optimization Platform",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.config = config
    app.state.metrics = metrics
    app.state.model_cache = model_cache
    app.state.batch_manager = batch_manager

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        available = config.model_root.is_dir()
        return JSONResponse(
            {"ready": available, "device": str(device)},
            status_code=200 if available else 503,
        )

    @app.get("/v1/models")
    async def models():
        model_ids = []
        if config.model_root.is_dir():
            model_ids = sorted(
                child.name
                for child in config.model_root.iterdir()
                if child.is_dir() and (child / "config.json").exists()
            )
        return {"object": "list", "data": [{"id": item, "object": "model"} for item in model_ids]}

    @app.get("/metrics")
    async def prometheus_metrics():
        return Response(metrics.render(), media_type="text/plain; version=0.0.4")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        started = time.perf_counter()
        metrics.in_progress.inc()
        try:
            settings = GenerationSettings(
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )
            output = await batch_manager.submit(
                settings, GenerationInput(messages=request.messages)
            )
            elapsed = time.perf_counter() - started
            # The backend timer starts after queueing, model lookup, and
            # tokenization. Add that pre-generation time back so TTFT is
            # measured from request arrival rather than kernel launch.
            end_to_end_ttft = max(
                0.0, elapsed - output.total_seconds + output.ttft_seconds
            )
            metrics.requests.labels(status="success").inc()
            metrics.request_latency.observe(elapsed)
            metrics.ttft.observe(end_to_end_ttft)
            metrics.generated_tokens.inc(output.completion_tokens)
        except RuntimeError as error:
            metrics.requests.labels(status="error").inc()
            metrics.observe_error(type(error).__name__)
            status_code = 429 if "queue is full" in str(error) else 503
            raise HTTPException(status_code=status_code, detail=str(error)) from error
        finally:
            metrics.in_progress.dec()

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        if request.stream:
            async def stream_response():
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"content": output.text}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                chunk["choices"][0] = {"index": 0, "delta": {}, "finish_reason": "stop"}
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream_response(), media_type="text/event-stream")

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": output.text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": output.prompt_tokens,
                "completion_tokens": output.completion_tokens,
                "total_tokens": output.prompt_tokens + output.completion_tokens,
            },
            "performance": {
                "ttft_ms": round(end_to_end_ttft * 1000, 3),
                "backend_ttft_ms": round(output.ttft_seconds * 1000, 3),
                "generation_ms": round(output.total_seconds * 1000, 3),
                "end_to_end_ms": round(elapsed * 1000, 3),
            },
        }

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8998)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run("serving.app:app", host=args.host, port=args.port, workers=args.workers)


if __name__ == "__main__":
    main()
