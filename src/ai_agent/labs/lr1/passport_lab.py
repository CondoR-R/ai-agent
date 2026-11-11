"""Паспорт сервера и проверки реализации (ЛР1, п. 3.1 и 3.4)

Запускается через каркас как запуск exp 0, серия passport: паспорт
и результаты проверок лежат в каталоге запуска и в MLflow, а дерево git
остаётся чистым.
"""

from pathlib import Path

from ai_agent import console, files, metadata

from . import data, prompts, server
from .client import make_client, timed_chat
from .endpoint import load_endpoint

DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4}
PASSPORT_FILENAME = "passport.json"
# допустимое расхождение длины запроса: локальный токенизатор и сервер
PROMPT_TOLERANCE = 2


def _require_kv_log(kv_cache: dict, log_path: str | None) -> None:
    missing = [
        key
        for key in ("kv_cache_tokens", "available_kv_gib", "weights_dtype")
        if kv_cache.get(key) is None
    ]
    if missing:
        raise ValueError(
            f"В журнале сервера ({log_path}) не найдено: {missing}. "
            "Проверьте путь к журналу и версию vLLM"
        )


def passport_lab(cfg: dict, run_dir: Path) -> None:
    """Собирает паспорт и выполняет проверки п. 3.4 против сервера.

    Проверки: ёмкость KV-кеша против журнала; ответ ровно заданной
    длины с ignore_eos; длина запроса по usage против локального
    токенизатора; два одинаковых запроса при температуре 0 (результат
    записывается, но ошибкой не считается).

    raise:
        RuntimeError, если проверка не пройдена (файлы при этом сохранены)
    """
    endpoint = load_endpoint(files.PROJECT_ROOT / cfg["endpoint_path"])
    checks_cfg = cfg["checks"]
    results_path = run_dir / files.RESULTS_FILENAME

    # 1. паспорт сервера
    info = server.collect_server_info(endpoint, cfg["server"])
    metadata.add(run_dir, {"endpoint": endpoint.to_record(), **info})
    gpu = info["device"]["gpu"]
    console.info(
        f"сервер: {info['server']['name']} {info['server']['version']}, "
        f"модель {info['model']['repo']}, "
        f"ревизия {str(info['model']['revision'])[:7]}, "
        f"контекст {info['server']['max_model_len']}"
    )
    console.info(
        f"устройство: {gpu['name'] if gpu else 'GPU не найден'}, "
        f"CPU потоков: {info['device']['cpu_count']}"
    )

    # 2. ёмкость KV-кеша против журнала
    kv_cache = info["kv_cache"]
    _require_kv_log(kv_cache, info["server"]["log_path"])
    kv_cfg = server.kv_config_from_hf(endpoint.model, info["model"]["revision"])
    dtype = kv_cache["weights_dtype"]
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"Неизвестный dtype весов в журнале: {dtype!r}")
    bytes_per_token = server.kv_bytes_per_token(
        kv_cfg["n_layers"],
        kv_cfg["n_kv_heads"],
        kv_cfg["head_dim"],
        DTYPE_BYTES[dtype],
    )
    kv_check = server.check_kv_capacity(
        kv_cache["kv_cache_tokens"],
        bytes_per_token,
        kv_cache["available_kv_gib"],
        checks_cfg["kv_rel_tol"],
    )
    max_len = info["server"]["max_model_len"]
    console.info(
        f"KV-кеш: {bytes_per_token} байт/токен, "
        f"{kv_cache['kv_cache_tokens']} токенов в кеше, "
        f"{kv_check['computed_gib']:.2f} из "
        f"{kv_check['available_gib']:.2f} GiB (ok={kv_check['ok']})"
    )

    # 3. проверки на сервере; все запросы пишутся в results.jsonl
    text = data.load_verified_text()
    tokenizer = prompts.load_tokenizer(
        endpoint.model, info["model"]["revision"]
    )
    builder = prompts.PromptBuilder(text, tokenizer)
    client = make_client(endpoint)

    def send(request_id: str, messages: list[dict], max_tokens: int, **kw):
        record = timed_chat(
            client,
            endpoint,
            messages,
            max_tokens,
            request_id=request_id,
            condition={"check": request_id.split("-")[0]},
            **kw,
        )
        files.append_jsonl(results_path, record)

    warm_messages, _ = builder.build(0, 256)
    send("warmup", warm_messages, 8, warmup=True)

    exact_messages, _ = builder.build(1, 256)
    send(
        "exact_tokens",
        exact_messages,
        checks_cfg["fixed_output_tokens"],
        ignore_eos=True,
    )

    determinism_messages, _ = builder.build(2, 256)
    for attempt in (1, 2):
        send(
            f"determinism-{attempt}",
            determinism_messages,
            64,
            keep_text=True,
        )

    length_messages, expected_len = builder.build(
        3, checks_cfg["prompt_tokens"]
    )
    send("prompt_length", length_messages, 16, ignore_eos=True)

    # 4. итоги считаются из results.jsonl, а не из памяти
    records = {r["id"]: r for r in files.read_jsonl(results_path)}
    failed_requests = [rid for rid, r in records.items() if r["status"] != "ok"]

    exact = records["exact_tokens"]
    exact_ok = (
        exact["status"] == "ok"
        and exact["completion_tokens"] == checks_cfg["fixed_output_tokens"]
    )

    first, second = records["determinism-1"], records["determinism-2"]
    same_text = (
        first["text"] == second["text"]
        if first["status"] == "ok" and second["status"] == "ok"
        else None
    )

    length = records["prompt_length"]
    length_diff = (
        length["prompt_tokens"] - expected_len
        if length["prompt_tokens"] is not None
        else None
    )
    length_ok = length_diff is not None and abs(length_diff) <= PROMPT_TOLERANCE

    checks = {
        "kv_capacity": kv_check["ok"],
        "exact_tokens": exact_ok,
        "prompt_length": length_ok,
        "requests_ok": not failed_requests,
    }
    console.info(
        f"ровно {checks_cfg['fixed_output_tokens']} токенов: "
        f"получено {exact['completion_tokens']} (ok={exact_ok})"
    )
    console.info(
        f"длина запроса: цель {checks_cfg['prompt_tokens']}, "
        f"локально {expected_len}, usage {length['prompt_tokens']} "
        f"(ok={length_ok})"
    )
    console.info(
        "температура 0, два одинаковых запроса: "
        + {True: "тексты совпали", False: "тексты РАЗЛИЧАЮТСЯ", None: "n/a"}[
            same_text
        ]
    )

    passport = {
        "server": info["server"],
        "model": info["model"],
        "device": info["device"],
        "generation": {
            **endpoint.generation,
            "extra_body": endpoint.extra_body,
        },
        "kv_cache": {
            **kv_cache,
            **kv_cfg,
            "bytes_per_token": bytes_per_token,
            "context_gib": max_len * bytes_per_token / server.GIB,
            "check": kv_check,
        },
        "checks": {
            **checks,
            "determinism_same_text": same_text,
            "prompt_length_diff": length_diff,
            "failed_requests": failed_requests,
        },
    }
    files.write_json(run_dir / PASSPORT_FILENAME, passport)

    files.write_json(
        run_dir / files.METRICS_FILENAME,
        {
            "kv_bytes_per_token": bytes_per_token,
            "kv_cache_tokens": kv_cache["kv_cache_tokens"],
            "kv_computed_gib": kv_check["computed_gib"],
            "kv_available_gib": kv_check["available_gib"],
            "kv_ratio": kv_check["ratio"],
            "kv_context_gib": max_len * bytes_per_token / server.GIB,
            "max_concurrency_x": kv_cache["max_concurrency_x"],
            "exact_tokens_completion": exact["completion_tokens"],
            "prompt_length_diff": length_diff,
            "determinism_same_text": (
                None if same_text is None else int(same_text)
            ),
            **{f"check_{name}_ok": int(ok) for name, ok in checks.items()},
        },
    )

    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"Проверки не пройдены: {failed}")
