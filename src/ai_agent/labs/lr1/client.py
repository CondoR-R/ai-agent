"""Клиент-измеритель к OpenAI-совместимому серверу (ЛР1, п. 3.2)"""

import time

import httpx
import openai
from openai import AsyncOpenAI, OpenAI

from .endpoint import Endpoint
from .timing import StreamRecorder

# Сбои вызова — это неуспешный результат записи, а не остановка серии
# (Правила, п. 6). Ошибки в нашем коде сюда не попадают и серию остановят.
CALL_ERRORS = (openai.OpenAIError, httpx.HTTPError)


def make_client(endpoint: Endpoint) -> OpenAI:
    """Синхронный клиент.

    max_retries=0: молчаливый повтор запроса спрятал бы сбой и
    приплюсовал время ожидания к TTFT.
    """
    return OpenAI(
        base_url=endpoint.base_url,
        api_key=endpoint.api_key,
        timeout=endpoint.timeout_s,
        max_retries=0,
    )


def make_async_client(endpoint: Endpoint) -> AsyncOpenAI:
    """Асинхронный клиент для опытов под нагрузкой."""
    return AsyncOpenAI(
        base_url=endpoint.base_url,
        api_key=endpoint.api_key,
        timeout=endpoint.timeout_s,
        max_retries=0,
    )


def timed_chat(
    client: OpenAI,
    endpoint: Endpoint,
    messages: list[dict],
    max_tokens: int,
    *,
    request_id: str,
    condition: dict | None = None,
    ignore_eos: bool = False,
    warmup: bool = False,
    keep_text: bool = False,
    keep_chunk_times: bool = False,
) -> dict:
    """Один потоковый запрос с замером времени.

    Время: t0 — перед отправкой, t1 — первый фрагмент текста, tN —
    последний. Запись готова для files.append_jsonl(results.jsonl).

    params:
        client - клиент из make_client
        endpoint - параметры сервера и генерации
        messages - сообщения чата
        max_tokens - предел длины ответа
        request_id - идентификатор записи
        condition - условие опыта (попадает в запись)
        ignore_eos - ответ ровно max_tokens токенов
        warmup - пометить прогревочный запрос
        keep_text - сохранить текст ответа
        keep_chunk_times - сохранить время каждого фрагмента

    return: запись результата; при сбое вызова status="error"
    """
    kwargs = endpoint.chat_kwargs(max_tokens, ignore_eos=ignore_eos)
    t0 = time.perf_counter()
    recorder = StreamRecorder(t0)
    error = None
    try:
        stream = client.chat.completions.create(messages=messages, **kwargs)
        for chunk in stream:
            recorder.feed(chunk, time.perf_counter())
    except CALL_ERRORS as exc:
        error = f"{type(exc).__name__}: {exc}"
    t_end = time.perf_counter()
    return recorder.record(
        request_id=request_id,
        condition=condition,
        warmup=warmup,
        t_end=t_end,
        error=error,
        keep_text=keep_text,
        keep_chunk_times=keep_chunk_times,
    )


async def async_timed_chat(
    client: AsyncOpenAI,
    endpoint: Endpoint,
    messages: list[dict],
    max_tokens: int,
    *,
    request_id: str,
    condition: dict | None = None,
    ignore_eos: bool = False,
    warmup: bool = False,
    keep_text: bool = False,
    keep_chunk_times: bool = False,
) -> dict:
    """Асинхронный вариант timed_chat: те же отметки и та же запись."""
    kwargs = endpoint.chat_kwargs(max_tokens, ignore_eos=ignore_eos)
    t0 = time.perf_counter()
    recorder = StreamRecorder(t0)
    error = None
    try:
        stream = await client.chat.completions.create(
            messages=messages, **kwargs
        )
        async for chunk in stream:
            recorder.feed(chunk, time.perf_counter())
    except CALL_ERRORS as exc:
        error = f"{type(exc).__name__}: {exc}"
    t_end = time.perf_counter()
    return recorder.record(
        request_id=request_id,
        condition=condition,
        warmup=warmup,
        t_end=t_end,
        error=error,
        keep_text=keep_text,
        keep_chunk_times=keep_chunk_times,
    )
