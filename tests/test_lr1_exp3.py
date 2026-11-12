"""Проверки расчётов опыта 3 без сервера."""

import pytest

from ai_agent.labs.lr1 import exp3_analysis as analysis

CFG = {
    "exp3": {
        "prompt_tokens": 1024,
        "output_tokens": 128,
        "clients": [1, 2],
        "min_requests": 16,
        "requests_per_client": 4,
        "limits": {"ttft_p90_s": 2.0, "tpot_median_s": 0.1},
    }
}


def _request(
    clients,
    index,
    ttft=0.1,
    tpot=0.02,
    output_tokens=128,
    status="ok",
    t_send=0.0,
    duration=1.0,
):
    return {
        "id": f"k{clients}-r{index}",
        "condition": {
            "phase": "run",
            "clients": clients,
            "request": index,
        },
        "status": status,
        "error": None if status == "ok" else "ошибка",
        "warmup": False,
        "prompt_tokens": 1024 if status == "ok" else None,
        "completion_tokens": output_tokens if status == "ok" else None,
        "ttft_s": ttft if status == "ok" else None,
        "tpot_s": tpot if status == "ok" else None,
        "e2e_s": ttft + (output_tokens - 1) * tpot if status == "ok" else None,
        "t_send_s": t_send,
        "t_done_s": t_send + duration,
    }


def _level_record(clients, waiting, kv_usage):
    snapshot = {
        "vllm:num_requests_waiting": waiting,
        "vllm:kv_cache_usage_perc": kv_usage,
    }
    return {
        "id": f"level-{clients}",
        "condition": {"phase": "level", "clients": clients},
        "status": "ok",
        "warmup": False,
        "server_before": snapshot,
        "server_after": snapshot,
    }


def _records():
    records = [
        {
            "id": "warmup",
            "condition": {"phase": "warmup"},
            "status": "ok",
            "warmup": True,
            "ttft_s": 9.0,
            "tpot_s": 9.0,
            "e2e_s": 9.0,
            "completion_tokens": 128,
            "t_send_s": 0.0,
            "t_done_s": 9.0,
        }
    ]
    # уровень 1: 15 успешных и один сбой; окно 0.1 .. 2.5 с
    for index in range(15):
        records.append(
            _request(
                clients=1,
                index=index,
                t_send=index * 0.1,
                duration=1.0,
            )
        )
    records.append(_request(clients=1, index=15, status="error", t_send=1.5))
    records.append(_level_record(1, waiting=0.0, kv_usage=0.1))

    # уровень 2: 16 успешных, но TTFT выше предела
    for index in range(16):
        records.append(
            _request(
                clients=2,
                index=index,
                ttft=3.0,
                t_send=index * 0.1,
                duration=4.0,
            )
        )
    records.append(_level_record(2, waiting=5.0, kv_usage=0.9))
    return records


def test_requests_for_level():
    assert analysis.requests_for_level(1, 16, 4) == 16
    assert analysis.requests_for_level(8, 16, 4) == 32
    assert analysis.requests_for_level(32, 16, 4) == 128


def test_percentile_and_empty():
    assert analysis.percentile([1, 2, 3, 4], 90) == pytest.approx(3.7)
    assert analysis.percentile([], 90) is None


def test_levels_throughput_goodput_and_limits():
    metrics = analysis.summarize_level_run(
        _records(), CFG, kv_cache_tokens=60000
    )
    low = metrics["by_clients"]["1"]
    assert low["n_planned"] == 16
    assert low["n_ok"] == 15 and low["n_error"] == 1
    # окно: 0.0 .. 14*0.1+1.0 = 2.4 с; 15 * 128 токенов
    assert low["wall_s"] == pytest.approx(2.4)
    assert low["output_tokens_total"] == 15 * 128
    assert low["throughput_tok_s"] == pytest.approx(15 * 128 / 2.4)
    assert low["requests_per_s"] == pytest.approx(15 / 2.4)
    assert low["ttft_median_s"] == pytest.approx(0.1)
    assert low["ttft_p90_s"] == pytest.approx(0.1)
    assert low["tpot_median_s"] == pytest.approx(0.02)
    assert low["goodput_requests"] == 15
    assert low["goodput_rps"] == pytest.approx(15 / 2.4)
    assert low["limits_ok"] is True
    assert low["queue_waiting_after"] == 0.0
    assert low["kv_cache_usage_after"] == pytest.approx(0.1)

    high = metrics["by_clients"]["2"]
    assert high["n_ok"] == 16 and high["n_error"] == 0
    assert high["ttft_median_s"] == pytest.approx(3.0)
    assert high["goodput_requests"] == 0  # TTFT выше предела
    assert high["limits_ok"] is False
    assert high["queue_waiting_after"] == 5.0
    assert high["kv_cache_usage_after"] == pytest.approx(0.9)

    assert metrics["max_clients_within_limits"] == 1
    assert metrics["kv_requests_capacity"] == 60000 // (1024 + 128)
    assert metrics["n_planned"] == 32
    assert metrics["n_ok"] == 31
    assert metrics["n_error"] == 1


def test_goodput_checks_each_request_not_only_medians():
    records = [
        _request(clients=1, index=index, t_send=index * 0.1)
        for index in range(20)
    ]
    records[-1]["ttft_s"] = 5.0  # один запрос выпал, а p90 остался низким
    level = analysis.summarize_level_run(records, CFG, kv_cache_tokens=None)
    entry = level["by_clients"]["1"]
    assert entry["goodput_requests"] == 19
    assert entry["limits_ok"] is True


def test_no_successful_requests_gives_no_throughput():
    records = [_request(clients=1, index=0, status="error")]
    metrics = analysis.summarize_level_run(records, CFG, kv_cache_tokens=None)
    entry = metrics["by_clients"]["1"]
    assert entry["n_ok"] == 0 and entry["n_error"] == 1
    assert "throughput_tok_s" not in entry
    assert entry["limits_ok"] is False
    assert metrics["max_clients_within_limits"] is None
    assert metrics["kv_requests_capacity"] is None


def test_summaries_are_strings():
    metrics = analysis.summarize_level_run(
        _records(), CFG, kv_cache_tokens=60000
    )
    assert "ток/с" in analysis.metrics_summary(metrics)
    assert "clients" in analysis.config_summary(CFG)
