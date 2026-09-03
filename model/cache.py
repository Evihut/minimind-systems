"""Inference cache implementations for MiniMind."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CacheStats:
    current_tokens: int
    capacity_tokens: int
    allocated_bytes: int


class StaticKVCache:
    """Preallocated KV cache that avoids a ``torch.cat`` on every decode step.

    Buffers are allocated lazily from the first key/value tensors, so the cache
    automatically follows the model's device and dtype. The cache is intended
    for inference; it deliberately rejects autograd-enabled updates.
    """

    def __init__(self, num_hidden_layers: int, max_cache_len: int, batch_size: int):
        if num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if max_cache_len <= 0:
            raise ValueError("max_cache_len must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.num_hidden_layers = num_hidden_layers
        self.max_cache_len = max_cache_len
        self.batch_size = batch_size
        self.position = 0
        self.key_cache: list[torch.Tensor | None] = [None] * num_hidden_layers
        self.value_cache: list[torch.Tensor | None] = [None] * num_hidden_layers

    def _allocate_like(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[0] != self.batch_size:
            raise ValueError(
                f"cache batch size is {self.batch_size}, received {tensor.shape[0]}"
            )
        shape = (self.batch_size, self.max_cache_len, *tensor.shape[2:])
        return torch.empty(shape, dtype=tensor.dtype, device=tensor.device)

    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.is_grad_enabled():
            raise RuntimeError("StaticKVCache is inference-only; use torch.inference_mode()")
        if not 0 <= layer_idx < self.num_hidden_layers:
            raise IndexError(f"invalid layer index {layer_idx}")
        end = self.position + key_states.shape[1]
        if end > self.max_cache_len:
            raise ValueError(
                f"KV cache capacity exceeded: requested {end}, capacity {self.max_cache_len}"
            )
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = self._allocate_like(key_states)
            self.value_cache[layer_idx] = self._allocate_like(value_states)

        key_buffer = self.key_cache[layer_idx]
        value_buffer = self.value_cache[layer_idx]
        assert key_buffer is not None and value_buffer is not None
        key_buffer[:, self.position : end].copy_(key_states)
        value_buffer[:, self.position : end].copy_(value_states)
        return key_buffer[:, :end], value_buffer[:, :end]

    def advance(self, token_count: int) -> None:
        if token_count <= 0:
            raise ValueError("token_count must be positive")
        new_position = self.position + token_count
        if new_position > self.max_cache_len:
            raise ValueError(
                f"KV cache capacity exceeded: requested {new_position}, capacity {self.max_cache_len}"
            )
        self.position = new_position

    def reset(self) -> None:
        """Reuse the allocated buffers for a new request with the same batch size."""
        self.position = 0

    def stats(self) -> CacheStats:
        buffers = [
            tensor
            for tensor in (*self.key_cache, *self.value_cache)
            if tensor is not None
        ]
        allocated_bytes = sum(tensor.numel() * tensor.element_size() for tensor in buffers)
        return CacheStats(
            current_tokens=self.position,
            capacity_tokens=self.max_cache_len,
            allocated_bytes=allocated_bytes,
        )

