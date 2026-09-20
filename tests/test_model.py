from __future__ import annotations

import pytest
import torch

from model.cache import StaticKVCache
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def tiny_config(**overrides) -> MiniMindConfig:
    values = {
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "vocab_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 64,
        "flash_attn": False,
        "dropout": 0.0,
    }
    values.update(overrides)
    return MiniMindConfig(**values)


def test_dense_forward_backward_and_tied_embeddings():
    torch.manual_seed(7)
    model = MiniMindForCausalLM(tiny_config())
    input_ids = torch.randint(0, 128, (2, 12))

    output = model(input_ids, labels=input_ids)
    output.loss.backward()

    assert output.logits.shape == (2, 12, 128)
    assert torch.isfinite(output.loss)
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    assert model.model.layers[0].self_attn.q_proj.weight.grad is not None


def test_cached_decode_matches_full_forward():
    torch.manual_seed(11)
    model = MiniMindForCausalLM(tiny_config()).eval()
    input_ids = torch.randint(0, 128, (1, 9))

    with torch.inference_mode():
        full_logits = model(input_ids).logits[:, -1]
        prefix = model(input_ids[:, :-1], use_cache=True)
        cached_logits = model(
            input_ids[:, -1:],
            past_key_values=prefix.past_key_values,
            use_cache=True,
        ).logits[:, -1]

    torch.testing.assert_close(cached_logits, full_logits, rtol=1e-4, atol=1e-5)


def test_static_cache_matches_dynamic_cache_without_reallocation():
    torch.manual_seed(12)
    model = MiniMindForCausalLM(tiny_config()).eval()
    input_ids = torch.randint(0, 128, (1, 8))
    static_cache = StaticKVCache(
        num_hidden_layers=model.config.num_hidden_layers,
        max_cache_len=16,
        batch_size=1,
    )

    with torch.inference_mode():
        dynamic_prefix = model(input_ids[:, :-1], use_cache=True)
        dynamic_step = model(
            input_ids[:, -1:],
            past_key_values=dynamic_prefix.past_key_values,
            use_cache=True,
        )
        model(input_ids[:, :-1], past_key_values=static_cache, use_cache=True)
        pointer_before = static_cache.key_cache[0].data_ptr()
        static_step = model(
            input_ids[:, -1:], past_key_values=static_cache, use_cache=True
        )
        pointer_after = static_cache.key_cache[0].data_ptr()

    torch.testing.assert_close(
        static_step.logits[:, -1], dynamic_step.logits[:, -1], rtol=1e-4, atol=1e-5
    )
    assert pointer_before == pointer_after
    assert static_cache.position == input_ids.shape[1]
    assert static_cache.stats().allocated_bytes > 0


def test_static_cache_first_prefill_uses_the_same_attention_path():
    torch.manual_seed(14)
    model = MiniMindForCausalLM(tiny_config(flash_attn=True)).eval()
    input_ids = torch.randint(0, 128, (1, 8))
    static_cache = StaticKVCache(
        num_hidden_layers=model.config.num_hidden_layers,
        max_cache_len=12,
        batch_size=1,
    )

    with torch.inference_mode():
        expected = model(input_ids, use_cache=True)
        actual = model(input_ids, past_key_values=static_cache, use_cache=True)

    torch.testing.assert_close(actual.logits, expected.logits, rtol=1e-5, atol=1e-6)
    assert static_cache.position == input_ids.shape[1]


def test_sdpa_cached_decode_matches_full_forward():
    torch.manual_seed(15)
    model = MiniMindForCausalLM(tiny_config(flash_attn=True)).eval()
    input_ids = torch.randint(0, 128, (2, 9))

    with torch.inference_mode():
        full_logits = model(input_ids).logits[:, -1]
        prefix = model(input_ids[:, :-1], use_cache=True)
        cached_logits = model(
            input_ids[:, -1:],
            past_key_values=prefix.past_key_values,
            use_cache=True,
        ).logits[:, -1]
        static_cache = StaticKVCache(
            num_hidden_layers=model.config.num_hidden_layers,
            max_cache_len=input_ids.shape[1],
            batch_size=input_ids.shape[0],
        )
        model(input_ids[:, :-1], past_key_values=static_cache, use_cache=True)
        static_logits = model(
            input_ids[:, -1:],
            past_key_values=static_cache,
            use_cache=True,
        ).logits[:, -1]

    torch.testing.assert_close(cached_logits, full_logits, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(static_logits, cached_logits, rtol=1e-4, atol=1e-5)


def test_moe_routes_tokens_and_backpropagates_auxiliary_loss():
    torch.manual_seed(13)
    model = MiniMindForCausalLM(
        tiny_config(use_moe=True, num_experts=4, num_experts_per_tok=2)
    )
    input_ids = torch.randint(0, 128, (2, 10))

    output = model(input_ids, labels=input_ids)
    (output.loss + output.aux_loss).backward()

    assert output.aux_loss.item() > 0
    assert model.model.layers[0].mlp.gate.weight.grad is not None


def test_max_seq_len_alias_is_honored():
    config = tiny_config(max_position_embeddings=96)
    assert config.max_seq_len == 96
    assert config.max_position_embeddings == 96

    values = tiny_config().to_dict()
    values.pop("max_position_embeddings")
    values["max_seq_len"] = 80
    aliased = MiniMindConfig(**values)
    assert aliased.max_position_embeddings == 80
    assert tiny_config(num_key_value_heads=None).num_key_value_heads == 4


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"num_attention_heads": 3, "num_key_value_heads": 2}, "divisible"),
        ({"head_dim": 7}, "even"),
        ({"use_moe": True, "num_experts": 2, "num_experts_per_tok": 3}, "between"),
    ],
)
def test_invalid_architecture_fails_fast(overrides, match):
    with pytest.raises(ValueError, match=match):
        tiny_config(**overrides)


def test_generation_validates_sampling_and_handles_large_top_k():
    model = MiniMindForCausalLM(tiny_config()).eval()
    input_ids = torch.tensor([[1, 5, 9]])

    generated = model.generate(
        input_ids,
        max_new_tokens=2,
        do_sample=False,
        top_k=10_000,
        top_p=1.0,
        eos_token_id=None,
    )
    assert generated.shape == (1, 5)

    static_generated = model.generate(
        input_ids,
        max_new_tokens=2,
        do_sample=False,
        top_k=0,
        top_p=1.0,
        eos_token_id=None,
        cache_implementation="static",
    )
    torch.testing.assert_close(static_generated, generated)

    with pytest.raises(ValueError, match="temperature"):
        model.generate(input_ids, max_new_tokens=1, do_sample=True, temperature=0)
