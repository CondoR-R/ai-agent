"""Вызов модели с JSON Schema для ЛР2."""

import time
from typing import Any, cast

import httpx
import openai
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from ai_agent.labs.lr1.endpoint import Endpoint

from .prompts import SCHEMA

CALL_ERRORS = (openai.OpenAIError, httpx.HTTPError)


def make_client(endpoint: Endpoint) -> OpenAI:
    """Клиент без автоматических повторов."""
    return OpenAI(
        base_url=endpoint.base_url,
        api_key=endpoint.api_key,
        timeout=endpoint.timeout_s,
        max_retries=0,
    )


def chat(
    client: OpenAI,
    endpoint: Endpoint,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int = 128,
) -> dict:
    """Один вызов chat.completions с response_format по схеме ЛР2."""
    kwargs = endpoint.chat_kwargs(max_tokens)
    kwargs["response_format"] = SCHEMA
    started = time.perf_counter()
    try:
        completion = client.chat.completions.create(
            messages=cast(list[ChatCompletionMessageParam], messages),
            **kwargs,
        )
    except CALL_ERRORS as exc:
        return {
            "status": "call_error",
            "error": f"{type(exc).__name__}: {exc}",
            "raw_response": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "finish_reason": None,
            "generation_s": time.perf_counter() - started,
        }

    choice = completion.choices[0]
    usage = completion.usage
    return {
        "status": "ok",
        "error": None,
        "raw_response": choice.message.content,
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "finish_reason": choice.finish_reason,
        "generation_s": time.perf_counter() - started,
    }
