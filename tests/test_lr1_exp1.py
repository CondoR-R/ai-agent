"""Проверки расчётов опыта 1 без сервера."""

import pytest

from ai_agent.labs.lr1 import exp1_analysis as analysis
from ai_agent.labs.lr1 import server


def _record(
    condition,
    ttft=0.1,
    prompt_tokens=256,
    completion_tokens=16,
    status="ok",
    warmup=False,
    delta=None,
):
    return {
        "id": "x",
        "condition": condition,
        "status": status,
        "error": None if status == "ok" else "ошибка",
        "warmup": warmup,
        "prompt_tokens": prompt_tokens if status == "ok" else None,
        "completion_tokens": completion_tokens if status == "ok" else None,
        "ttft_s": ttft if status == "ok" else None,
        "server_delta": delta or {},
    }


def _length_records():
    """2 длины по 3 запроса: 256 -> TTFT 0.1 с, 1024 -> TTFT 0.4 с."""
    records = [_record({"phase": "warmup"}, ttft=9.0, warmup=True)]
    for repeat, (t_short, t_long) in enumerate(
        [(0.10, 0.40), (0.11, 0.42), (0.09, 0.38)], start=1
    ):
        records.append(
            _record({"prompt_tokens_target": 256, "repeat": repeat},
                    ttft=t_short, prompt_tokens=256)
        )
        records.append(
            _record({"prompt_tokens_target": 1024, "repeat": repeat},
                    ttft=t_long, prompt_tokens=1024)
        )
    return records


ECFG_LENGTH = {
    "mode": "ttft_vs_length",
    "prompt_lengths": [256, 1024],
    "repeats": 3,
    "output_tokens": 16,
}


def test_by_length_median_range_and_warmup_excluded():
    by_length = analysis.summarize_by_length(
        _length_records(), [256, 1024], 3, 16
    )
    short = by_length["256"]
    assert short["n_ok"] == 3 and short["n_planned"] == 3
    assert short["ttft_median_s"] == pytest.approx(0.10)  # прогрев (9 с) не вошёл
    assert short["ttft_min_s"] == pytest.approx(0.09)
    assert short["ttft_max_s"] == pytest.approx(0.11)
    # скорость prefill = токены / TTFT, медиана по запросам
    assert short["prefill_tok_s_median"] == pytest.approx(256 / 0.10)


def test_failed_request_stays_in_denominator():
    records = _length_records()
    records.append(
        _record({"prompt_tokens_target": 256, "repeat": 4}, status="error")
    )
    by_length = analysis.summarize_by_length(records, [256, 1024], 5, 16)
    short = by_length["256"]
    assert short["n_planned"] == 5       # знаменатель: все запланированные
    assert short["n_recorded"] == 4
    assert short["n_ok"] == 3            # ошибка в медианы не входит
    assert short["ttft_median_s"] == pytest.approx(0.10)


def test_wrong_output_length_is_counted():
    records = [
        _record({"prompt_tokens_target": 256}, completion_tokens=16),
        _record({"prompt_tokens_target": 256}, completion_tokens=12),
    ]
    by_length = analysis.summarize_by_length(records, [256], 2, 16)
    assert by_length["256"]["n_wrong_output_len"] == 1


def test_scaling_and_fit_on_known_line():
    metrics = analysis.summarize_length_run(_length_records(), ECFG_LENGTH)
    pair = metrics["scaling"]["256-1024"]
    assert pair["length_ratio"] == pytest.approx(4.0)
    assert pair["ttft_ratio"] == pytest.approx(4.0)   # 0.1 -> 0.4: линейно
    # добавлено 768 токенов за 0.3 с
    assert pair["marginal_prefill_tok_s"] == pytest.approx(768 / 0.3)
    fit = metrics["ttft_fit"]
    assert fit["slope_s_per_token"] == pytest.approx(0.3 / 768)
    assert fit["intercept_s"] == pytest.approx(0.1 - 256 * 0.3 / 768)
    assert metrics["n_planned"] == 6 and metrics["n_ok"] == 6


def test_fit_needs_two_points():
    records = [_record({"prompt_tokens_target": 256})]
    by_length = analysis.summarize_by_length(records, [256], 1, 16)
    assert analysis.fit_ttft(by_length, [256]) is None


# ---------- серия prefix_cache ----------

ECFG_CACHE = {
    "mode": "prefix_cache",
    "prompt_tokens": 8192,
    "repeats": 3,
    "n_prompts": 1,
    "output_tokens": 16,
}


def _cache_records(first_hit=0.0):
    hits = "vllm:prefix_cache_hits_total"
    cond = {"prompt": 1, "prompt_tokens_target": 8192}
    return [
        _record({"phase": "warmup"}, warmup=True),
        _record({**cond, "attempt": 1}, ttft=2.0, prompt_tokens=8192,
                delta={hits: first_hit}),
        _record({**cond, "attempt": 2}, ttft=0.2, prompt_tokens=8192,
                delta={hits: 8176.0}),
        _record({**cond, "attempt": 3}, ttft=0.25, prompt_tokens=8192,
                delta={hits: 8176.0}),
    ]


def test_cache_speedup_and_cold_start():
    metrics = analysis.summarize_cache_run(_cache_records(), ECFG_CACHE)
    entry = metrics["prompts"]["1"]
    assert entry["speedup_2"] == pytest.approx(10.0)
    assert entry["speedup_3"] == pytest.approx(8.0)
    assert entry["hit_tokens"] == [0.0, 8176.0, 8176.0]
    assert entry["cold_start_ok"] == 1
    assert metrics["speedup_2_median"] == pytest.approx(10.0)


def test_cache_detects_warm_first_request():
    metrics = analysis.summarize_cache_run(
        _cache_records(first_hit=8000.0), ECFG_CACHE
    )
    assert metrics["prompts"]["1"]["cold_start_ok"] == 0


def test_cache_without_counters_gives_unknown_cold_start():
    records = [
        _record({"prompt": 1, "attempt": a}, ttft=0.5, prompt_tokens=8192)
        for a in (1, 2, 3)
    ]
    metrics = analysis.summarize_cache_run(records, ECFG_CACHE)
    assert metrics["prompts"]["1"]["cold_start_ok"] is None


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        analysis.summarize_run("другое", [], {})


# ---------- состояние кеширования сервера ----------


def test_prefix_caching_from_log_has_priority():
    info = server.parse_vllm_log("config: x=1, enable_prefix_caching=False, y=2")
    assert info["prefix_caching"] is False
    assert server.prefix_caching_state(info, "vllm serve m") == (
        False, "журнал vLLM")


def test_prefix_caching_from_command_and_default():
    empty = server.parse_vllm_log("")
    assert server.prefix_caching_state(
        empty, "vllm serve m --no-enable-prefix-caching")[0] is False
    assert server.prefix_caching_state(
        empty, "vllm serve m --enable-prefix-caching")[0] is True
    enabled, source = server.prefix_caching_state(empty, "vllm serve m")
    assert enabled is True and "умолчание" in source


def test_prefix_cache_counters_parsed():
    text = (
        'vllm:prefix_cache_queries_total{model_name="m"} 16384.0\n'
        'vllm:prefix_cache_hits_total{model_name="m"} 8176.0\n'
    )
    got = server.parse_metrics(text, server.EXP1_METRICS)
    assert got["vllm:prefix_cache_hits_total"] == 8176.0


def test_summaries_are_strings():
    metrics = analysis.summarize_length_run(_length_records(), ECFG_LENGTH)
    assert "256" in analysis.metrics_summary(metrics)
    cache = analysis.summarize_cache_run(_cache_records(), ECFG_CACHE)
    assert "speedup_2_median" in analysis.metrics_summary(cache)
    assert "mode=ttft_vs_length" in analysis.config_summary(
        {"exp1": ECFG_LENGTH})
