from __future__ import annotations

import json

from scripts.api_schema import ChatRequest, parse_response


def test_parse_response_separates_reasoning_content_and_tool_calls():
    response = (
        "<think>Need the current weather.</think>\n"
        "I will check it."
        '<tool_call>{"name":"get_weather","arguments":{"city":"Shanghai"}}</tool_call>'
    )

    content, reasoning, tool_calls = parse_response(response)

    assert content == "I will check it."
    assert reasoning == "Need the current weather."
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"city": "Shanghai"}


def test_parse_response_tolerates_incomplete_reasoning_prefix():
    content, reasoning, tool_calls = parse_response("private reasoning</think>public answer")

    assert content == "public answer"
    assert reasoning == "private reasoning"
    assert tool_calls is None


def test_chat_request_accepts_both_thinking_switch_names():
    direct = ChatRequest(model="minimind", messages=[], open_thinking=True)
    compatible = ChatRequest(
        model="minimind",
        messages=[],
        chat_template_kwargs={"enable_thinking": True},
    )

    assert direct.get_open_thinking() is True
    assert compatible.get_open_thinking() is True
