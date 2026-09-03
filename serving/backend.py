from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.cache import StaticKVCache
from model.model_minimind import MiniMindForCausalLM


@dataclass(frozen=True)
class GenerationSettings:
    model: str
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0

    def __post_init__(self):
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")


@dataclass(frozen=True)
class GenerationInput:
    messages: list[dict]


@dataclass(frozen=True)
class GenerationOutput:
    text: str
    prompt_tokens: int
    completion_tokens: int
    ttft_seconds: float
    total_seconds: float


class TransformersBackend:
    """Batched Transformers backend used by the optimized API service."""

    def __init__(self, model, tokenizer, device: torch.device):
        self.model = model.eval().to(device)
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = device
        self._lock = asyncio.Lock()

    @classmethod
    def from_pretrained(cls, model_path: Path, device: torch.device):
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        config_data = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        if config_data.get("model_type") == MiniMindForCausalLM.config_class.model_type:
            model = MiniMindForCausalLM.from_pretrained(model_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        return cls(model, tokenizer, device)

    def _generate_sync(
        self, inputs: list[GenerationInput], settings: GenerationSettings
    ) -> list[GenerationOutput]:
        prompts = [
            self.tokenizer.apply_chat_template(
                item.messages, tokenize=False, add_generation_prompt=True
            )
            for item in inputs
        ]
        encoded = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        past_key_values = None
        if isinstance(self.model, MiniMindForCausalLM):
            past_key_values = StaticKVCache(
                num_hidden_layers=self.model.config.num_hidden_layers,
                max_cache_len=input_ids.shape[1] + settings.max_tokens,
                batch_size=input_ids.shape[0],
            )
        generated_tokens = []
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=self.device)
        ttft_seconds = 0.0
        with torch.inference_mode():
            step_input = input_ids
            for step in range(settings.max_tokens):
                output = self.model(
                    step_input,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = output.logits[:, -1]
                if settings.temperature > 0:
                    logits = logits / settings.temperature
                    if settings.top_p < 1:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        remove = cumulative > settings.top_p
                        remove[:, 1:] = remove[:, :-1].clone()
                        remove[:, 0] = False
                        original_order_remove = torch.zeros_like(remove).scatter(
                            1, sorted_indices, remove
                        )
                        logits = logits.masked_fill(original_order_remove, -torch.inf)
                    next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                if self.tokenizer.eos_token_id is not None:
                    next_token = torch.where(
                        finished[:, None],
                        torch.full_like(next_token, self.tokenizer.eos_token_id),
                        next_token,
                    )
                generated_tokens.append(next_token)
                if step == 0:
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    ttft_seconds = time.perf_counter() - started
                past_key_values = output.past_key_values
                step_input = next_token
                attention_mask = torch.cat(
                    (attention_mask, torch.ones_like(next_token, dtype=attention_mask.dtype)),
                    dim=1,
                )
                if self.tokenizer.eos_token_id is not None:
                    finished |= next_token.squeeze(1).eq(self.tokenizer.eos_token_id)
                    if bool(finished.all()):
                        break
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        total_seconds = time.perf_counter() - started
        generated = torch.cat(generated_tokens, dim=1)
        outputs = []
        for index, completion in enumerate(generated):
            if self.tokenizer.eos_token_id is not None:
                eos_positions = completion.eq(self.tokenizer.eos_token_id).nonzero()
                if eos_positions.numel():
                    completion = completion[: int(eos_positions[0].item()) + 1]
            text = self.tokenizer.decode(completion, skip_special_tokens=True)
            prompt_tokens = int(encoded["attention_mask"][index].sum().item())
            completion_tokens = int(completion.numel())
            outputs.append(
                GenerationOutput(
                    text=text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    ttft_seconds=ttft_seconds,
                    total_seconds=total_seconds,
                )
            )
        return outputs

    async def generate_batch(
        self, inputs: list[GenerationInput], settings: GenerationSettings
    ) -> list[GenerationOutput]:
        # Serialize access to one model instance. Dynamic batching creates GPU
        # parallelism inside the request instead of racing multiple CUDA streams.
        async with self._lock:
            return await asyncio.to_thread(self._generate_sync, inputs, settings)
