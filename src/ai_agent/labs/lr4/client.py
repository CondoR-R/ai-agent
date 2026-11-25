"""OpenAI-совместимый клиент с вызовом инструментов для ЛР4."""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import yaml
from openai import OpenAI

from ai_agent import files


@dataclass
class ChatResult:
    content: str
    tool_calls: list[dict]
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class ModelClient:
    def __init__(self, endpoint_path: Path = files.ENDPOINT_PATH):
        endpoint = yaml.safe_load(endpoint_path.read_text())
        api_key_env = endpoint.get("api_key_env")
        api_key = (
            os.environ.get(api_key_env, "local") if api_key_env else "local"
        )
        self.client = OpenAI(
            base_url=endpoint["base_url"],
            api_key=api_key,
            timeout=float(endpoint["timeout_s"]),
        )
        self.base_url = endpoint["base_url"]
        self.model = endpoint["model"]
        self.generation = endpoint["generation"]

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
        max_tokens: int = 1024,
    ) -> ChatResult:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.generation["temperature"],
            "top_p": self.generation["top_p"],
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"
        started = time.perf_counter()
        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message
        calls = []
        for call in message.tool_calls or []:
            try:
                args = json.loads(call.function.arguments)
            except (TypeError, json.JSONDecodeError):
                args = {}
            calls.append(
                {
                    "id": call.id,
                    "name": call.function.name,
                    "args": args,
                    "raw_args": call.function.arguments,
                }
            )
        usage = response.usage
        return ChatResult(
            content=message.content or "",
            tool_calls=calls,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_s=time.perf_counter() - started,
        )
