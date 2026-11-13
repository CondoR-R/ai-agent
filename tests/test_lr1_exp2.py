"""Проверки расчётов опыта 2 без сервера."""

import pytest

from ai_agent.labs.lr1 import exp2_analysis as analysis


def _record(
    output_tokens,
    e2e=1.0,
    ttft=0.1,
    completion=None,
    status="ok",
    warmup=False,
):
    completion = output_tokens if completion is None else completion
    tpot = None
    if status == "ok" and completion > 1:
        tpot = (e2e - ttft) / (completion - 1)
    return {
        "id": "x",
        "condition": {"output_tokens_target": output_tokens, "repeat": 1},
        "status": status,
        "error": None if status == "ok" else "ошибка",
        "warmup": warmup,
        "prompt_tokens": 1024 if status == "ok" else None,
        "completion_tokens": completion if status == "ok" else None,
        "ttft_s": ttft if status == "ok" else None,
        "tpot_s": tpot,
        "e2e_s": e2e if status == "ok" else None,
    }


def _linear_records(lengths=()):
    """e2e = 0.1 + (N - 1) * 0.02, то есть TTFT 0.1 с и TPOT 20 мс."""
    records = [_record(16, e2e=9.0, warmup=True)]
    for n in lengths:
        records.append(_record(n, e2e=0.1 + (n - 1) * 0.02, ttft=0.1))
    return records


def test_by_output_median_range_and_warmup_excluded():
    records = _linear_records((16, 64))
    records += [
        _record(16, e2e=2.0),
        _record(16, e2e=3.0),
        _record(64, e2e=4.0),
        _record(64, e2e=5.0),
    ]
    by_output = analysis.summarize_by_output(records, [16, 64], 3)

    short = by_output["16"]
    assert short["n_ok"] == 3 and short["n_planned"] == 3
    assert short["e2e_median_s"] == pytest.approx(2.0)  # прогрев (9 с) не вошёл
    assert short["e2e_min_s"] == pytest.approx(0.4)
    assert short["e2e_max_s"] == pytest.approx(3.0)
    assert short["tpot_median_s"] == pytest.approx(1.9 / 15)


def test_failed_request_stays_in_denominator():
    records = _linear_records((16,))
    records.append(_record(16, status="error"))
    by_output = analysis.summarize_by_output(records, [16], 5)
    short = by_output["16"]
    assert short["n_planned"] == 5
    assert short["n_recorded"] == 2
    assert short["n_ok"] == 1
    assert short["e2e_median_s"] == pytest.approx(0.4)


def test_wrong_output_length_is_counted():
    records = [
        _record(16, completion=16),
        _record(16, completion=12),
    ]
    by_output = analysis.summarize_by_output(records, [16], 2)
    assert by_output["16"]["n_wrong_output_len"] == 1


def test_fit_on_known_line():
    lengths = [16, 64, 256, 1024]
    metrics = analysis.summarize_run(
        _linear_records(lengths),
        {"exp2": {"output_lengths": lengths, "repeats": 1}},
    )
    fit = metrics["e2e_fit"]
    assert fit["slope_s_per_token"] == pytest.approx(0.02)
    # e2e = TTFT + (N - 1) * TPOT = (TTFT - TPOT) + N * TPOT
    assert fit["intercept_s"] == pytest.approx(0.08)
    assert fit["intercept_minus_ttft_s"] == pytest.approx(-0.02)
    assert fit["slope_minus_tpot_s"] == pytest.approx(0.0, abs=1e-12)


def test_fit_needs_two_points():
    by_output = analysis.summarize_by_output(_linear_records((16,)), [16], 1)
    assert analysis.fit_e2e(by_output, [16]) is None


def test_decode_lower_bound_and_tpot_ratio():
    bound = analysis.decode_lower_bound_s(7.45, 300.0)
    assert bound == pytest.approx(7.45 * analysis.GIB / 300e9)
    assert (
        0.026 < bound < 0.027
    )  # ≈ 27 мс для Qwen3-4B FP16 на T4 (пример из методички)

    lengths = [16, 64]
    metrics = analysis.summarize_run(
        _linear_records(lengths),
        {
            "exp2": {"output_lengths": lengths, "repeats": 1},
            "decode_memory": {
                "weights_gib": 7.45,
                "memory_bandwidth_gb_s": 300.0,
            },
        },
    )
    assert metrics["tpot_median_s"] == pytest.approx(0.02)
    assert metrics["decode_lower_bound_tok_s"] == pytest.approx(1 / bound)
    assert metrics["tpot_over_lower_bound"] == pytest.approx(0.02 / bound)


def test_lower_bound_absent_without_memory_config():
    metrics = analysis.summarize_run(
        _linear_records((16, 64)),
        {"exp2": {"output_lengths": [16, 64], "repeats": 1}},
    )
    assert metrics["decode_lower_bound_s"] is None
    assert metrics["tpot_over_lower_bound"] is None


def test_summaries_are_strings():
    metrics = analysis.summarize_run(
        _linear_records((16, 64)),
        {"exp2": {"output_lengths": [16, 64], "repeats": 1}},
    )
    assert "16" in analysis.metrics_summary(metrics)
    assert "output_lengths" in analysis.config_summary(
        {"exp2": {"output_lengths": [16, 64], "repeats": 1}}
    )
