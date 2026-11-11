"""Описание сервера модели: endpoint.yaml (ЛР1 и далее)"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from ai_agent import files

# поля запроса, которыми управляет клиент, а не endpoint.yaml
RESERVED_GENERATION_KEYS = (
    "model",
    "messages",
    "max_tokens",
    "stream",
    "stream_options",
)


@dataclass(frozen=True)
class Endpoint:
    """Адрес сервера, имя модели и параметры генерации по умолчанию."""

    base_url: str
    model: str
    generation: dict
    extra_body: dict = field(default_factory=dict)
    timeout_s: float = 600.0
    api_key_env: str | None = None

    @property
    def api_key(self) -> str:
        """Ключ берётся из переменной окружения, а не из репозитория."""
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise ValueError(
                    f"Переменная окружения {self.api_key_env} не задана"
                )
            return key
        return "none"

    @property
    def origin(self) -> str:
        """Адрес сервера без суффикса /v1 (для /metrics)."""
        url = self.base_url.rstrip("/")
        return url[: -len("/v1")] if url.endswith("/v1") else url

    def chat_kwargs(self, max_tokens: int, *, ignore_eos: bool = False) -> dict:
        """Аргументы chat.completions.create для потокового запроса.

        Все параметры генерации передаются явно: умолчания сервера
        (например, generation_config.json у vLLM) на результат влиять
        не должны.

        params:
            max_tokens - предел длины ответа
            ignore_eos - не обрывать генерацию на токене конца текста
                         (ответ ровно max_tokens токенов)
        """
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            **self.generation,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        body = dict(self.extra_body)
        if ignore_eos:
            body["ignore_eos"] = True
        if body:
            kwargs["extra_body"] = body
        return kwargs

    def to_record(self) -> dict:
        """Описание для metadata.json (без ключа доступа)."""
        return {
            "base_url": self.base_url,
            "model": self.model,
            "generation": dict(self.generation),
            "extra_body": dict(self.extra_body),
            "timeout_s": self.timeout_s,
        }


def load_endpoint(path: Path | None = None) -> Endpoint:
    """Читает и проверяет endpoint.yaml

    params:
        path - путь к файлу (по умолчанию endpoint.yaml в корне проекта)

    raise:
        ValueError, если нет обязательных полей или не задана температура
    """
    path = path or files.ENDPOINT_PATH
    raw = files.read_yaml(path) or {}

    missing = [k for k in ("base_url", "model", "generation") if k not in raw]
    if missing:
        raise ValueError(f"В {path.name} нет полей: {missing}")

    generation = dict(raw["generation"] or {})
    if "temperature" not in generation:
        raise ValueError(
            f"В {path.name} не задана generation.temperature: "
            "температуру нужно передавать явно"
        )
    reserved = [k for k in generation if k in RESERVED_GENERATION_KEYS]
    if reserved:
        raise ValueError(
            f"В generation нельзя задавать {reserved}: их ставит клиент"
        )

    return Endpoint(
        base_url=raw["base_url"],
        model=raw["model"],
        generation=generation,
        extra_body=dict(raw.get("extra_body") or {}),
        timeout_s=float(raw.get("timeout_s", 600.0)),
        api_key_env=raw.get("api_key_env"),
    )
