"""Итоги эксперимента 3 из results.jsonl (чистые функции).

Уровень нагрузки — число одновременных клиентов. Каждый клиент берёт
следующий запрос из общей очереди, пока она не опустеет. Пропускная
способность считается по всем успешным запросам уровня, goodput — по
тем, что уложились в пределы TTFT и TPOT (Правила, п. 6, 7).
"""

from statistics import median

import numpy as np


def requests_for_level(
    clients: int, min_requests: int, requests_per_client: int
) -> int:
    """Сколько запросов планируется на уровень: max(16, 4c)."""
    return max(min_requests, requests_per_client * clients)


def percentile(values: list[float], q: float) -> float | None:
    """q-й процентиль (линейная интерполяция), None для пустого списка."""
    if not values:
        return None
    return float(np.percentile(values, q))


def _median(values: list[float]) -> float | None:
    return median(values) if values else None


def _snapshot_value(record: dict, key: str, moment: str) -> float | None:
    """Значение метрики сервера из служебной записи уровня."""
    snapshot = record.get(f"server_{moment}") or {}
    return snapshot.get(key)


def summarize_level_run(
    records: list[dict],
    cfg: dict,
    kv_cache_tokens: int | None = None,
) -> dict:
    """Пропускная способность, задержки и goodput по уровням нагрузки.

    params:
        records - записи results.jsonl (запросы и служебные записи уровней)
        cfg - конфигурация запуска: exp3
        kv_cache_tokens - ёмкость KV-кеша из metadata.json, токенов

    return: metrics.json эксперимента 3
    """
    ecfg = cfg["exp3"]
    limits = ecfg["limits"]
    prompt_tokens = ecfg["prompt_tokens"]
    output_tokens = ecfg["output_tokens"]

    run_records = [
        r
        for r in records
        if not r["warmup"] and r["condition"].get("phase") == "run"
    ]
    level_records = {
        r["condition"].get("clients"): r
        for r in records
        if r["condition"].get("phase") == "level"
    }

    levels: dict[str, dict] = {}
    for clients in ecfg["clients"]:
        group = [
            r for r in run_records if r["condition"].get("clients") == clients
        ]
        ok = [r for r in group if r["status"] == "ok"]
        planned = requests_for_level(
            clients, ecfg["min_requests"], ecfg["requests_per_client"]
        )
        entry: dict = {
            "clients": clients,
            "n_planned": planned,
            "n_recorded": len(group),
            "n_ok": len(ok),
            "n_error": len(group) - len(ok),
        }

        if ok:
            ttft = [r["ttft_s"] for r in ok]
            tpot = [r["tpot_s"] for r in ok if r["tpot_s"] is not None]
            e2e = [r["e2e_s"] for r in ok]
            wall = max(r["t_done_s"] for r in ok) - min(
                r["t_send_s"] for r in ok
            )
            total_out = sum(r["completion_tokens"] for r in ok)
            entry.update(
                wall_s=wall,
                output_tokens_total=total_out,
                throughput_tok_s=total_out / wall if wall > 0 else None,
                requests_per_s=len(ok) / wall if wall > 0 else None,
                ttft_median_s=median(ttft),
                ttft_p90_s=percentile(ttft, 90),
                tpot_median_s=_median(tpot),
                e2e_median_s=median(e2e),
            )
            good = [
                r
                for r in ok
                if r["ttft_s"] <= limits["ttft_p90_s"]
                and r["tpot_s"] is not None
                and r["tpot_s"] <= limits["tpot_median_s"]
            ]
            entry["goodput_requests"] = len(good)
            entry["goodput_rps"] = len(good) / wall if wall > 0 else None

        ttft_p90 = entry.get("ttft_p90_s")
        tpot_median = entry.get("tpot_median_s")
        entry["limits_ok"] = bool(
            ttft_p90 is not None
            and tpot_median is not None
            and ttft_p90 <= limits["ttft_p90_s"]
            and tpot_median <= limits["tpot_median_s"]
        )

        level_record = level_records.get(clients) or {}
        entry["server_before"] = level_record.get("server_before")
        entry["server_after"] = level_record.get("server_after")
        entry["queue_waiting_after"] = _snapshot_value(
            level_record, "vllm:num_requests_waiting", "after"
        )
        entry["kv_cache_usage_after"] = _snapshot_value(
            level_record, "vllm:kv_cache_usage_perc", "after"
        )
        if entry["kv_cache_usage_after"] is None:
            entry["kv_cache_usage_after"] = _snapshot_value(
                level_record, "vllm:gpu_cache_usage_perc", "after"
            )
        levels[str(clients)] = entry

    within = [e["clients"] for e in levels.values() if e["limits_ok"]]
    capacity = (
        kv_cache_tokens // (prompt_tokens + output_tokens)
        if kv_cache_tokens
        else None
    )
    return {
        "n_planned": sum(e["n_planned"] for e in levels.values()),
        "n_ok": sum(e["n_ok"] for e in levels.values()),
        "n_error": sum(e["n_error"] for e in levels.values()),
        "limits": limits,
        "max_clients_within_limits": max(within) if within else None,
        "kv_cache_tokens": kv_cache_tokens,
        "kv_requests_capacity": capacity,
        "by_clients": levels,
    }


def config_summary(cfg: dict) -> str:
    """Выжимка конфигурации для терминала."""
    return ", ".join(f"{k}={v}" for k, v in cfg["exp3"].items())


def metrics_summary(metrics: dict) -> str:
    """Выжимка итогов для терминала."""
    parts = [
        f"{clients} кл.: {e['throughput_tok_s']:.0f} ток/с, "
        f"TTFT p90 {e['ttft_p90_s'] * 1000:.0f} мс"
        for clients, e in metrics["by_clients"].items()
        if "throughput_tok_s" in e
    ]
    return "; ".join(parts) or "нет успешных запросов"
