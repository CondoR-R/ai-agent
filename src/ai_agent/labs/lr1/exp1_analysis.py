"""Итоги эксперимента 1 из results.jsonl (чистые функции).

Все числа считаются из сохранённых записей, а не из памяти процесса
(Правила, п. 4.2). Знаменатель — все запланированные записи: ошибка
запроса остаётся в n_planned, но в медианы не входит.
"""

from itertools import pairwise
from statistics import median

import numpy as np

# подстроки имён счётчика попаданий в кеш префикса: vLLM и llama.cpp
HITS_MARKERS = ("prefix_cache_hits", "prompt_tokens_cached")


def _group(records: list[dict], **condition) -> list[dict]:
    """Записи без прогрева с заданными полями условия опыта."""
    return [
        r
        for r in records
        if not r["warmup"]
        and all(r["condition"].get(k) == v for k, v in condition.items())
    ]


# ---------- серия ttft_vs_length ----------


def summarize_by_length(
    records: list[dict],
    lengths: list[int],
    repeats: int,
    output_tokens: int,
) -> dict:
    """Медиана и диапазон TTFT и скорость prefill по каждой длине.

    Скорость prefill = prompt_tokens (из usage) / TTFT. TTFT включает ещё
    и первый шаг decode, поэтому на коротких запросах скорость занижена.

    params:
        records - записи results.jsonl
        lengths - длины запросов, токенов
        repeats - запланированное число запросов на длину (знаменатель)
        output_tokens - ожидаемая длина ответа

    return: {str(длина): сводка}
    """
    result: dict[str, dict] = {}
    for length in lengths:
        group = _group(records, prompt_tokens_target=length)
        ok = [r for r in group if r["status"] == "ok"]
        entry: dict = {
            "n_planned": repeats,
            "n_recorded": len(group),
            "n_ok": len(ok),
        }
        if ok:
            ttft = [r["ttft_s"] for r in ok]
            speed = [r["prompt_tokens"] / r["ttft_s"] for r in ok]
            entry.update(
                prompt_tokens_median=median([r["prompt_tokens"] for r in ok]),
                prompt_dev_max=max(
                    abs(r["prompt_tokens"] - length) for r in ok
                ),
                ttft_median_s=median(ttft),
                ttft_min_s=min(ttft),
                ttft_max_s=max(ttft),
                prefill_tok_s_median=median(speed),
                prefill_tok_s_min=min(speed),
                prefill_tok_s_max=max(speed),
                n_wrong_output_len=sum(
                    1 for r in ok if r["completion_tokens"] != output_tokens
                ),
            )
        result[str(length)] = entry
    return result


def scaling(by_length: dict, lengths: list[int]) -> dict:
    """Как меняется TTFT между соседними длинами.

    ttft_ratio > length_ratio — рост быстрее линейного.
    marginal_prefill_tok_s — скорость на добавленных токенах, без
    постоянной части TTFT.
    """
    result: dict[str, dict] = {}
    for a, b in pairwise(lengths):
        first, second = by_length[str(a)], by_length[str(b)]
        if "ttft_median_s" not in first or "ttft_median_s" not in second:
            continue
        dt = second["ttft_median_s"] - first["ttft_median_s"]
        dn = second["prompt_tokens_median"] - first["prompt_tokens_median"]
        result[f"{a}-{b}"] = {
            "length_ratio": second["prompt_tokens_median"]
            / first["prompt_tokens_median"],
            "ttft_ratio": second["ttft_median_s"] / first["ttft_median_s"],
            "marginal_prefill_tok_s": dn / dt if dt > 0 else None,
        }
    return result


def fit_ttft(by_length: dict, lengths: list[int]) -> dict | None:
    """Прямая TTFT = a + b * длина по медианам.

    a — постоянная часть (очередь, первый шаг decode, накладные расходы);
    1/b — скорость prefill без неё. Квадратичная часть внимания на
    длинных запросах делает прямую приближённой.
    """
    points = [
        (
            by_length[str(n)]["prompt_tokens_median"],
            by_length[str(n)]["ttft_median_s"],
        )
        for n in lengths
        if "ttft_median_s" in by_length[str(n)]
    ]
    if len(points) < 2:
        return None
    xs, ys = zip(*points, strict=False)
    slope, intercept = np.polyfit(xs, ys, 1)
    return {
        "slope_s_per_token": float(slope),
        "intercept_s": float(intercept),
        "prefill_tok_s": float(1 / slope) if slope > 0 else None,
    }


def summarize_length_run(records: list[dict], ecfg: dict) -> dict:
    """Итоги серии ttft_vs_length (metrics.json)."""
    lengths = ecfg["prompt_lengths"]
    by_length = summarize_by_length(
        records, lengths, ecfg["repeats"], ecfg["output_tokens"]
    )
    hits = [
        h
        for h in (hit_tokens(r) for r in records if not r["warmup"])
        if h is not None
    ]
    return {
        "n_planned": len(lengths) * ecfg["repeats"],
        "n_ok": sum(e["n_ok"] for e in by_length.values()),
        # по счётчику сервера: сколько токенов одного запроса пришло из кеша
        "cache_hit_tokens_max": max(hits) if hits else None,
        "n_wrong_output_len": sum(
            e.get("n_wrong_output_len", 0) for e in by_length.values()
        ),
        "prompt_dev_max": max(
            (
                e["prompt_dev_max"]
                for e in by_length.values()
                if "prompt_dev_max" in e
            ),
            default=None,
        ),
        "by_length": by_length,
        "scaling": scaling(by_length, lengths),
        "ttft_fit": fit_ttft(by_length, lengths),
    }


# ---------- серия prefix_cache ----------


def hit_tokens(record: dict) -> float | None:
    """Сколько токенов запроса сервер взял из кеша (по /metrics)."""
    for name, value in (record.get("server_delta") or {}).items():
        if any(marker in name for marker in HITS_MARKERS):
            return value
    return None


def summarize_cache_run(records: list[dict], ecfg: dict) -> dict:
    """Итоги серии prefix_cache: TTFT трёх одинаковых запросов.

    Первый запрос «холодный», второй и третий могут взять префикс из
    кеша. cold_start_ok проверяет по счётчику попаданий, что первый
    запрос действительно был холодным.
    """
    repeats = ecfg["repeats"]
    prompts: dict[str, dict] = {}
    for number in range(1, ecfg["n_prompts"] + 1):
        group = sorted(
            _group(records, prompt=number),
            key=lambda r: r["condition"]["attempt"],
        )
        ok = [r for r in group if r["status"] == "ok"]
        ttft = [r["ttft_s"] if r["status"] == "ok" else None for r in group]
        hits = [hit_tokens(r) for r in group]
        prompt_tokens = ok[0]["prompt_tokens"] if ok else None

        entry: dict = {
            "n_planned": repeats,
            "n_recorded": len(group),
            "n_ok": len(ok),
            "prompt_tokens": prompt_tokens,
            "ttft_s": ttft,
            "hit_tokens": hits,
        }
        for attempt, value in enumerate(ttft, start=1):
            if value is not None:
                entry[f"ttft_attempt{attempt}_s"] = value
        for k in (2, 3):
            if len(ttft) >= k and ttft[0] and ttft[k - 1]:
                entry[f"speedup_{k}"] = ttft[0] / ttft[k - 1]
        first_hit = hits[0] if hits else None
        entry["cold_start_ok"] = (
            None
            if first_hit is None or prompt_tokens is None
            else int(first_hit <= 0.1 * prompt_tokens)
        )
        prompts[str(number)] = entry

    def med(key: str) -> float | None:
        values = [e[key] for e in prompts.values() if key in e]
        return median(values) if values else None

    return {
        "n_planned": ecfg["n_prompts"] * repeats,
        "n_ok": sum(e["n_ok"] for e in prompts.values()),
        "speedup_2_median": med("speedup_2"),
        "speedup_3_median": med("speedup_3"),
        "prompts": prompts,
    }


# ---------- общее ----------


def summarize_run(mode: str, records: list[dict], ecfg: dict) -> dict:
    """Итоги серии по её режиму (mode из конфигурации)."""
    if mode == "ttft_vs_length":
        return summarize_length_run(records, ecfg)
    if mode == "prefix_cache":
        return summarize_cache_run(records, ecfg)
    raise ValueError(f"Неизвестный режим опыта 1: {mode!r}")


def config_summary(cfg: dict) -> str:
    """Выжимка конфигурации для терминала."""
    return ", ".join(f"{k}={v}" for k, v in cfg["exp1"].items())


def metrics_summary(metrics: dict) -> str:
    """Выжимка итогов для терминала."""
    if "by_length" in metrics:
        parts = [
            f"{length}: {e['ttft_median_s'] * 1000:.0f} мс ({e['n_ok']}/"
            f"{e['n_planned']})"
            for length, e in metrics["by_length"].items()
            if "ttft_median_s" in e
        ]
        return "TTFT (медиана) " + "; ".join(parts)
    parts = []
    for key in ("speedup_2_median", "speedup_3_median"):
        if metrics.get(key) is not None:
            parts.append(f"{key}={metrics[key]:.2f}")
    return ", ".join(parts) or "нет успешных запросов"
