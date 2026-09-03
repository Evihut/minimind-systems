"""Dependency-light request schema and response parsing for the API server."""

from __future__ import annotations

import json
import re
import time

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    model: str
    messages: list
    temperature: float = 0.7
    top_p: float = 0.92
    max_tokens: int = 8192
    stream: bool = True
    tools: list = Field(default_factory=list)
    open_thinking: bool = False
    chat_template_kwargs: dict | None = None

    def get_open_thinking(self) -> bool:
        """Support the two common names used to enable reasoning output."""
        if self.open_thinking:
            return True
        if self.chat_template_kwargs:
            return self.chat_template_kwargs.get(
                "open_thinking", False
            ) or self.chat_template_kwargs.get("enable_thinking", False)
        return False


def parse_response(text: str) -> tuple[str, str | None, list[dict] | None]:
    """Split generated text into OpenAI-compatible content fields."""
    reasoning_content = None
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        reasoning_content = think_match.group(1).strip()
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    elif "</think>" in text:
        parts = text.split("</think>", 1)
        reasoning_content = parts[0].strip()
        text = parts[1].strip() if len(parts) > 1 else ""

    tool_calls = []
    for index, match in enumerate(re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)):
        try:
            call = json.loads(match.strip())
            tool_calls.append(
                {
                    "id": f"call_{int(time.time())}_{index}",
                    "type": "function",
                    "function": {
                        "name": call.get("name", ""),
                        "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False),
                    },
                }
            )
        except (json.JSONDecodeError, AttributeError):
            continue
    if tool_calls:
        text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)
    return text.strip(), reasoning_content, tool_calls or None

