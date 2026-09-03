from __future__ import annotations

from pathlib import Path

from transformers import AutoTokenizer


def test_bundled_tokenizer_round_trip_and_chat_template():
    tokenizer_path = Path(__file__).resolve().parents[1] / "model"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    text = "MiniMind 支持中英文。"

    token_ids = tokenizer(text, add_special_tokens=False).input_ids
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "你好"}],
        tokenize=False,
        add_generation_prompt=True,
    )

    assert token_ids
    assert "MiniMind" in decoded
    assert "你好" in prompt
    assert tokenizer.bos_token in prompt

