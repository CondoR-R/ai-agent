"""Проверки реализации ЛР1, не требующие сервера (п. 3.4).

Проверки против живого сервера (ровно 64 токена, длина запроса по usage,
ёмкость KV-кеша из журнала) выполняет scripts/lr1/passport.py.
"""

from types import SimpleNamespace

import pytest

from ai_agent import files
from ai_agent.labs.lr1 import data, prompts, server
from ai_agent.labs.lr1.endpoint import load_endpoint
from ai_agent.labs.lr1.timing import StreamRecorder, compute_timings


def _chunk(content=None, usage=None, finish=None):
    """Фрагмент потока в формате openai (без самой библиотеки)."""
    if content is None and finish is None:
        choices = []
    else:
        choices = [
            SimpleNamespace(
                delta=SimpleNamespace(content=content), finish_reason=finish
            )
        ]
    return SimpleNamespace(choices=choices, usage=usage)


# ---------- арифметика метрик ----------


def test_timings_on_given_marks():
    t = compute_timings(t0=0.0, t1=0.4, t_last=2.4, n_tokens=101)
    assert t["ttft_s"] == pytest.approx(0.400)
    assert t["tpot_s"] == pytest.approx(0.020)
    assert t["e2e_s"] == pytest.approx(2.4)


def test_tpot_undefined_for_one_token():
    assert compute_timings(0.0, 0.4, 0.4, 1)["tpot_s"] is None


def test_recorder_uses_usage_not_chunk_count():
    rec = StreamRecorder(t0=0.0)
    rec.feed(_chunk(content="a"), now=0.4)
    rec.feed(_chunk(content="bc"), now=1.4)  # в одном фрагменте два токена
    rec.feed(_chunk(content="d", finish="length"), now=2.4)
    rec.feed(
        _chunk(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=101)),
        now=2.5,
    )
    record = rec.record(
        request_id="r", condition={"n": 1}, warmup=False, t_end=2.5
    )
    assert record["status"] == "ok"
    assert record["completion_tokens"] == 101
    assert record["ttft_s"] == pytest.approx(0.4)
    assert record["tpot_s"] == pytest.approx(0.02)
    assert record["finish_reason"] == "length"
    assert record["n_chunks"] == 3


def test_recorder_flags_missing_usage_and_errors():
    rec = StreamRecorder(t0=0.0)
    rec.feed(_chunk(content="a"), now=0.1)
    no_usage = rec.record(request_id="r", condition=None, warmup=False, t_end=1)
    assert no_usage["status"] == "error"

    failed = StreamRecorder(t0=0.0).record(
        request_id="r",
        condition=None,
        warmup=False,
        t_end=1,
        error="APITimeoutError: timeout",
    )
    assert failed["status"] == "error" and failed["ttft_s"] is None


# ---------- KV-кеш и журнал сервера ----------

LOG = """
INFO Initializing a V1 LLM engine with config: dtype=torch.float16, x=1
INFO Available KV cache memory: 4.12 GiB
INFO GPU KV cache size: 29,984 tokens
INFO Maximum concurrency for 32,768 tokens per request: 0.91x
"""

LLAMA_LOG = """
llama_context: n_ctx                 = 32768
print_info: file type   = Q4_K - Medium
llama_kv_cache: size =  896.00 MiB ( 32768 cells,  28 layers,  1/1 seqs), K (f16):  448.00 MiB, V (f16):  448.00 MiB
"""


def test_kv_bytes_per_token_qwen3_4b():
    assert server.kv_bytes_per_token(36, 8, 128, 2) == 147_456
    # 32 тыс. токенов — 4,5 ГиБ (п. 2.2 методички)
    assert pytest.approx(4.5) == 32_768 * 147_456 / server.GIB


def test_parse_vllm_log():
    info = server.parse_vllm_log(LOG)
    assert info["kv_cache_tokens"] == 29_984
    assert info["available_kv_gib"] == pytest.approx(4.12)
    assert info["max_concurrency_x"] == pytest.approx(0.91)
    assert info["max_concurrency_tokens_per_request"] == 32_768
    assert info["weights_dtype"] == "float16"


def test_parse_vllm_log_takes_last_run_and_missing_is_none():
    info = server.parse_vllm_log("GPU KV cache size: 1,000 tokens\n" + LOG)
    assert info["kv_cache_tokens"] == 29_984
    assert server.parse_vllm_log("ничего")["kv_cache_tokens"] is None


def test_command_version_reads_stderr(monkeypatch):
    class Result:
        stdout = ""
        stderr = "version: 0.9.0 (build 12110)\n"

    monkeypatch.setattr(
        server.subprocess, "run", lambda *args, **kwargs: Result()
    )
    assert server._command_version("llama-server").startswith("0.9.0")


def test_parse_llama_log():
    info = server.parse_llama_log(LLAMA_LOG)
    assert info["kv_cache_tokens"] == 32_768
    assert info["available_kv_gib"] == pytest.approx(0.875)
    assert info["kv_cache_dtype"] == "f16"
    assert info["kv_cache_k_gib"] == pytest.approx(448 / 1024)
    assert info["weights_dtype"] == "Q4_K - Medium"
    assert server.parse_llama_log("ничего")["kv_cache_tokens"] is None
    assert server.parse_server_log(LLAMA_LOG)["kv_cache_dtype"] == "f16"

    parallel = LLAMA_LOG.replace(
        "( 32768 cells,  28 layers,  1/1 seqs)",
        "(  2048 cells,  28 layers, 32/32 seqs)",
    ).replace("896.00 MiB", "1792.00 MiB")
    info = server.parse_llama_log(parallel)
    assert info["kv_cache_tokens"] == 65_536
    assert info["available_kv_gib"] == pytest.approx(1.75)


def test_kv_capacity_check():
    ok = server.check_kv_capacity(29_984, 147_456, 4.12)
    assert ok["ok"]
    wrong_dtype = server.check_kv_capacity(29_984, 147_456 // 2, 4.12)
    assert not wrong_dtype["ok"]


def test_parse_metrics():
    text = (
        "# HELP vllm:num_requests_waiting x\n"
        'vllm:num_requests_waiting{model_name="m"} 3.0\n'
        'vllm:kv_cache_usage_perc{model_name="m"} 0.25\n'
        "other_metric 1\n"
    )
    got = server.parse_metrics(text, server.DEFAULT_METRICS)
    assert got == {
        "vllm:num_requests_waiting": 3.0,
        "vllm:kv_cache_usage_perc": 0.25,
    }


# ---------- endpoint.yaml ----------


def test_endpoint_requires_explicit_temperature(tmp_path):
    path = tmp_path / "endpoint.yaml"
    path.write_text("base_url: http://x/v1\nmodel: m\ngeneration: {top_p: 1}\n")
    with pytest.raises(ValueError, match="temperature"):
        load_endpoint(path)


def test_endpoint_request_kwargs(tmp_path):
    path = tmp_path / "endpoint.yaml"
    path.write_text(
        "base_url: http://x/v1\nmodel: m\n"
        "generation: {temperature: 0.0}\nextra_body: {top_k: -1}\n"
    )
    endpoint = load_endpoint(path)
    kwargs = endpoint.chat_kwargs(64, ignore_eos=True)
    assert kwargs["temperature"] == 0.0
    # stream и stream_options клиент передаёт в create явно
    assert "stream" not in kwargs and "stream_options" not in kwargs
    assert kwargs["extra_body"] == {"top_k": -1, "ignore_eos": True}
    assert "ignore_eos" not in endpoint.chat_kwargs(64).get("extra_body", {})
    assert endpoint.origin == "http://x"


# ---------- запросы заданной длины ----------


class WordTokenizer:
    """Токенизатор-заглушка: токен = слово (для проверки логики подгонки)."""

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids):
        return " ".join(ids)

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True
    ):
        body = "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
            for m in messages
        )
        return body + "<|im_start|>assistant\n"


def _builder():
    text = " ".join(f"w{i}" for i in range(5000))
    return prompts.PromptBuilder(text, WordTokenizer())


def test_prompt_has_requested_length_and_unique_label_first():
    builder = _builder()
    messages, n = builder.build(5, 300)
    assert n == 300 and builder.count(messages) == 300
    assert messages[0]["content"].startswith("[request 000005]")
    other, _ = builder.build(6, 300)
    assert other[0]["content"].startswith("[request 000006]")


def test_prompt_offsets_give_different_text():
    builder = _builder()
    a, _ = builder.build(1, 200, offset=builder.spread_offset(1, 200))
    b, _ = builder.build(2, 200, offset=builder.spread_offset(2, 200))
    assert (
        a[0]["content"].split("\n\n", 1)[1]
        != b[0]["content"].split("\n\n", 1)[1]
    )


def test_prompt_too_long_or_too_short_raises():
    builder = _builder()
    with pytest.raises(ValueError):
        builder.build(1, 2)
    with pytest.raises(ValueError):
        builder.build(1, 10_000)


# ---------- данные ----------


def test_strip_gutenberg():
    raw = (
        "header\r\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\r\n"
        "Chapter 1\r\nBody.\r\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\r\n"
        "license"
    )
    assert data.strip_gutenberg(raw) == "Chapter 1\nBody.\n"
    with pytest.raises(ValueError):
        data.strip_gutenberg("нет маркеров")


def test_jsonl_roundtrip(tmp_path):
    path = tmp_path / "results.jsonl"
    files.append_jsonl(path, {"id": "a", "x": 1})
    files.append_jsonl(path, {"id": "б", "x": None})
    assert files.read_jsonl(path) == [
        {"id": "a", "x": 1},
        {"id": "б", "x": None},
    ]
