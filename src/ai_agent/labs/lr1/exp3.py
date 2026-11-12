"""Эксперимент 3 ЛР1: нагрузка, пропускная способность и goodput (п. 4.3)

Уровень — число одновременных клиентов. Клиенты берут запросы из общей
очереди, пока она не опустеет; каждый следующий запрос отправляется
сразу после ответа на предыдущий. До и после уровня снимаются метрики
очереди и заполнения KV-кеша с адреса /metrics.
"""

import asyncio
from pathlib import Path

from ai_agent import console, files, metadata

from . import data, exp3_analysis, prompts, server
from .client import async_timed_chat, make_async_client
from .endpoint import Endpoint, load_endpoint

# номер прогревочного запроса не пересекается с номерами измеряемых
WARMUP_REQUEST_ID = 900_000


def _snapshot(endpoint: Endpoint) -> dict:
    """Снимок счётчиков сервера; пустой, если /metrics недоступен."""
    try:
        return server.fetch_metrics(endpoint, server.DEFAULT_METRICS)
    except OSError:
        return {}


def _delta(before: dict, after: dict) -> dict:
    return {k: after[k] - before[k] for k in after if k in before}


def _build_specs(
    builder: prompts.PromptBuilder,
    clients: int,
    first_request: int,
    ecfg: dict,
) -> list[dict]:
    """Запросы уровня с разным текстом и уникальными метками.

    Запросы строятся до запуска клиентов, поэтому подготовка сообщений
    не попадает в измеряемый интервал.
    """
    n_requests = exp3_analysis.requests_for_level(
        clients, ecfg["min_requests"], ecfg["requests_per_client"]
    )
    specs = []
    for offset in range(n_requests):
        number = first_request + offset
        position = builder.spread_offset(number, ecfg["prompt_tokens"])
        messages, expected = builder.build(
            number, ecfg["prompt_tokens"], offset=position
        )
        specs.append(
            {"number": number, "messages": messages, "expected": expected}
        )
    return specs


async def _worker(
    client,
    endpoint: Endpoint,
    queue: asyncio.Queue,
    condition: dict,
    results_path: Path,
    output_tokens: int,
) -> None:
    """Берёт запросы из очереди, пока она не опустеет."""
    while True:
        try:
            spec = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        record = await async_timed_chat(
            client,
            endpoint,
            spec["messages"],
            output_tokens,
            request_id=f"k{condition['clients']}-r{spec['number']}",
            condition={
                **condition,
                "request": spec["number"],
                "prompt_tokens_expected": spec["expected"],
            },
            ignore_eos=True,
        )
        files.append_jsonl(results_path, record)
        if record["status"] == "ok":
            console.info(
                f"клиентов {condition['clients']}, запрос {spec['number']}: "
                f"TTFT={record['ttft_s'] * 1000:.0f} мс, "
                f"TPOT={record['tpot_s'] * 1000:.0f} мс"
            )
        else:
            console.info(
                f"клиентов {condition['clients']}, запрос {spec['number']}: "
                f"ОШИБКА {record['error']}"
            )


async def _run_level(
    client,
    endpoint: Endpoint,
    specs: list[dict],
    clients: int,
    level_index: int,
    results_path: Path,
    output_tokens: int,
) -> dict:
    """Один уровень нагрузки: c клиентов на общей очереди запросов."""
    before = _snapshot(endpoint)
    queue: asyncio.Queue = asyncio.Queue()
    for spec in specs:
        queue.put_nowait(spec)

    condition = {"phase": "run", "clients": clients, "level": level_index}
    await asyncio.gather(
        *(
            _worker(
                client,
                endpoint,
                queue,
                condition,
                results_path,
                output_tokens,
            )
            for _ in range(clients)
        )
    )

    after = _snapshot(endpoint)
    level_record = {
        "id": f"level-{clients}",
        "condition": {"phase": "level", "clients": clients},
        "status": "ok",
        "error": None,
        "warmup": False,
        "prompt_tokens": None,
        "completion_tokens": None,
        "ttft_s": None,
        "tpot_s": None,
        "e2e_s": None,
        "finish_reason": None,
        "n_chunks": 0,
        "server_before": before,
        "server_after": after,
        "server_delta": _delta(before, after),
    }
    files.append_jsonl(results_path, level_record)
    return level_record


async def _run_exp3(cfg: dict, run_dir: Path) -> None:
    """Прогрев и все уровни нагрузки; клиент закрывается в finally."""
    ecfg = cfg["exp3"]
    results_path = run_dir / files.RESULTS_FILENAME

    endpoint = load_endpoint(files.PROJECT_ROOT / cfg["endpoint_path"])
    info = server.collect_server_info(endpoint, cfg["server"])
    metadata.add(run_dir, {"endpoint": endpoint.to_record(), **info})
    console.info(
        f"сервер: {info['server']['name']} {info['server']['version']}, "
        f"модель {info['model']['repo']}, "
        f"контекст {info['server']['max_model_len']}"
    )

    text = data.load_verified_text()
    tokenizer = prompts.load_tokenizer(
        endpoint.model, info["model"]["revision"]
    )
    builder = prompts.PromptBuilder(text, tokenizer)

    client = make_async_client(endpoint)
    try:
        warmup_messages, _ = builder.build(
            WARMUP_REQUEST_ID, ecfg["prompt_tokens"]
        )
        warmup = await async_timed_chat(
            client,
            endpoint,
            warmup_messages,
            ecfg["warmup_output_tokens"],
            request_id="warmup",
            condition={"phase": "warmup"},
            ignore_eos=True,
            warmup=True,
        )
        files.append_jsonl(results_path, warmup)
        if warmup["status"] != "ok":
            console.info(f"прогрев: ОШИБКА {warmup['error']}")

        request_number = 1
        for level_index, clients in enumerate(ecfg["clients"], start=1):
            specs = _build_specs(builder, clients, request_number, ecfg)
            console.info(
                f"уровень {level_index}: клиентов={clients}, "
                f"запросов={len(specs)}"
            )
            await _run_level(
                client,
                endpoint,
                specs,
                clients,
                level_index,
                results_path,
                ecfg["output_tokens"],
            )
            request_number += len(specs)
    finally:
        await client.close()

    records = files.read_jsonl(results_path)
    metrics = exp3_analysis.summarize_level_run(
        records, cfg, info["kv_cache"].get("kv_cache_tokens")
    )
    files.write_json(run_dir / files.METRICS_FILENAME, metrics)

    console.info(
        f"запросов: {metrics['n_ok']} из {metrics['n_planned']} успешно"
    )
    if metrics["n_error"]:
        console.info(f"ВНИМАНИЕ: ошибок {metrics['n_error']}")
    if metrics["max_clients_within_limits"] is not None:
        console.info(
            "пределы TTFT/TPOT выполняются до "
            f"{metrics['max_clients_within_limits']} клиентов"
        )
    else:
        console.info("пределы TTFT/TPOT не выполнены ни на одном уровне")
    if metrics["kv_requests_capacity"] is not None:
        console.info(
            f"в KV-кеш помещается запросов: {metrics['kv_requests_capacity']}"
        )


def exp3_lab(cfg: dict, run_dir: Path) -> None:
    """Запускает асинхронную серию опыта 3 под управлением каркаса."""
    asyncio.run(_run_exp3(cfg, run_dir))
