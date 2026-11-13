"""Итоги эксперимента 2 из results.jsonl (чистые функции).

Все числа считаются из сохранённых записей (Правила, п. 4.2).
Знаменатель — все запланированные записи: ошибка запроса остаётся
в n_planned, но в медианы не входит.
"""

from statistics import median

import numpy as np

GIB = 2**30


def _group(records: list[dict], **condition) -> list[dict]:
    """Записи без прогрева с заданными полями условия опыта."""
    return [
        r
        for r in records
        if not r["warmup"]
        and all(r["condition"].get(k) == v for k, v in condition.items())
    ]


def _median(values: list[float]) -> float | None:
    return median(values) if values else None


def summarize_by_output(
    records: list[dict],
    lengths: list[int],
    repeats: int,
) -> dict:
    """Время ответа и скорость генерации по длине ответа.

    Полное время ответа ≈ TTFT + (N - 1) * TPOT, поэтому медиана e2e
    растёт с N почти линейно. Скорость decode считается по числу
    токенов из usage, а не по числу фрагментов потока.

    params:
        records - записи results.jsonl
        lengths - длины ответов, токенов
        repeats - запланированное число запросов на длину (знаменатель)

    return: {str(длина): сводка}
    """
    result: dict[str, dict] = {}
    for length in lengths:
        group = _group(records, output_tokens_target=length)
        ok = [r for r in group if r["status"] == "ok"]
        entry: dict = {
            "n_planned": repeats,
            "n_recorded": len(group),
            "n_ok": len(ok),
        }
        if ok:
            e2e = [r["e2e_s"] for r in ok]
            ttft = [r["ttft_s"] for r in ok]
            tpot = [r["tpot_s"] for r in ok if r["tpot_s"] is not None]
            decode = [
                (r["completion_tokens"] - 1) / (r["e2e_s"] - r["ttft_s"])
                for r in ok
                if r["e2e_s"] > r["ttft_s"]
            ]
            entry.update(
                completion_tokens_median=median(
                    [r["completion_tokens"] for r in ok]
                ),
                e2e_median_s=median(e2e),
                e2e_min_s=min(e2e),
                e2e_max_s=max(e2e),
                ttft_median_s=median(ttft),
                tpot_median_s=_median(tpot),
                tpot_min_s=min(tpot) if tpot else None,
                tpot_max_s=max(tpot) if tpot else None,
                decode_tok_s_median=_median(decode),
                n_wrong_output_len=sum(
                    1 for r in ok if r["completion_tokens"] != length
                ),
            )
        result[str(length)] = entry
    return result


def fit_e2e(by_output: dict, lengths: list[int]) -> dict | None:
    """Прямая e2e = a + b * N по медианам.

    Свободный член a должен быть близок к TTFT (очередь, prefill, первый
    шаг decode), наклон b — к TPOT (среднее время следующего токена).

    return: коэффициенты и их расхождение с медианами TTFT и TPOT
    """
    points = [
        (
            by_output[str(n)]["completion_tokens_median"],
            by_output[str(n)]["e2e_median_s"],
        )
        for n in lengths
        if "e2e_median_s" in by_output[str(n)]
    ]
    if len(points) < 2:
        return None
    xs, ys = zip(*points, strict=False)
    slope, intercept = np.polyfit(xs, ys, 1)

    ttft = [
        by_output[str(n)]["ttft_median_s"]
        for n in lengths
        if "ttft_median_s" in by_output[str(n)]
    ]
    tpot = [
        by_output[str(n)]["tpot_median_s"]
        for n in lengths
        if by_output[str(n)].get("tpot_median_s") is not None
    ]
    ttft_ref = median(ttft) if ttft else None
    tpot_ref = median(tpot) if tpot else None
    return {
        "slope_s_per_token": float(slope),
        "intercept_s": float(intercept),
        "ttft_ref_s": ttft_ref,
        "tpot_ref_s": tpot_ref,
        "intercept_minus_ttft_s": (
            None if ttft_ref is None else float(intercept) - ttft_ref
        ),
        "slope_minus_tpot_s": (
            None if tpot_ref is None else float(slope) - tpot_ref
        ),
    }


def decode_lower_bound_s(
    weights_gib: float | None, bandwidth_gb_s: float | None
) -> float | None:
    """Нижняя граница времени одного шага decode (п. 2.1 методички).

    За шаг модель читает из памяти все веса: t >= веса / пропускная
    способность памяти. Для Qwen2.5-1.5B в Q4_K_M это ≈1.04 ГиБ /
    68.25 ГБ/с ≈ 0.016 с на Apple M1.
    """
    if not weights_gib or not bandwidth_gb_s:
        return None
    return weights_gib * GIB / (bandwidth_gb_s * 1e9)


def summarize_run(records: list[dict], cfg: dict) -> dict:
    """Итоги эксперимента 2 (metrics.json).

    params:
        records - записи results.jsonl
        cfg - конфигурация запуска: exp2 и decode_memory
    """
    ecfg = cfg["exp2"]
    lengths = ecfg["output_lengths"]
    by_output = summarize_by_output(records, lengths, ecfg["repeats"])

    tpot = [
        e["tpot_median_s"]
        for e in by_output.values()
        if e.get("tpot_median_s") is not None
    ]
    tpot_median = median(tpot) if tpot else None
    decode = cfg.get("decode_memory") or {}
    lower_bound = decode_lower_bound_s(
        decode.get("weights_gib"), decode.get("memory_bandwidth_gb_s")
    )

    return {
        "n_planned": len(lengths) * ecfg["repeats"],
        "n_ok": sum(e["n_ok"] for e in by_output.values()),
        "n_wrong_output_len": sum(
            e.get("n_wrong_output_len", 0) for e in by_output.values()
        ),
        "tpot_median_s": tpot_median,
        "decode_lower_bound_s": lower_bound,
        "decode_lower_bound_tok_s": (
            None if not lower_bound else 1 / lower_bound
        ),
        "tpot_over_lower_bound": (
            None
            if not lower_bound or tpot_median is None
            else tpot_median / lower_bound
        ),
        "by_output": by_output,
        "e2e_fit": fit_e2e(by_output, lengths),
    }


def config_summary(cfg: dict) -> str:
    """Выжимка конфигурации для терминала."""
    return ", ".join(f"{k}={v}" for k, v in cfg["exp2"].items())


def metrics_summary(metrics: dict) -> str:
    """Выжимка итогов для терминала."""
    parts = [
        f"{length}: {e['e2e_median_s'] * 1000:.0f} мс "
        f"({e['n_ok']}/{e['n_planned']})"
        for length, e in metrics["by_output"].items()
        if "e2e_median_s" in e
    ]
    return "e2e (медиана) " + "; ".join(parts)
