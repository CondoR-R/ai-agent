"""Эксперимент 1 ЛР1: длина запроса и TTFT, кеширование префикса (п. 4.1)

Две серии, у них разное состояние сервера (поэтому два запуска):
- ttft_vs_length: кеширование префикса выключено, запросы разной длины;
- prefix_cache: кеширование включено, один и тот же запрос трижды.
"""

from pathlib import Path

from ai_agent import console, files, metadata

from . import data, exp1_analysis, prompts, server
from .client import make_client, timed_chat
from .endpoint import Endpoint, load_endpoint

# номера прогревочных запросов не пересекаются с номерами измеряемых
WARMUP_REQUEST_ID = 900_000


def _snapshot(endpoint: Endpoint) -> dict:
    """Снимок счётчиков сервера; пустой, если /metrics недоступен."""
    try:
        return server.fetch_metrics(endpoint, server.EXP1_METRICS)
    except OSError:
        return {}


def _delta(before: dict, after: dict) -> dict:
    return {k: after[k] - before[k] for k in after if k in before}


class _Session:
    """Всё, что нужно для отправки запросов серии."""

    def __init__(
        self,
        endpoint: Endpoint,
        client,
        builder: prompts.PromptBuilder,
        results_path: Path,
        output_tokens: int,
    ) -> None:
        self.endpoint = endpoint
        self.client = client
        self.builder = builder
        self.results_path = results_path
        self.output_tokens = output_tokens

    def build(self, request_id: int, n_tokens: int, offset: int = 0):
        """Запрос нужной длины; допуск 0,2% (но не меньше 4 токенов)."""
        return self.builder.build(
            request_id,
            n_tokens,
            offset=offset,
            tolerance=max(4, n_tokens // 500),
        )

    def send(
        self,
        request_id: str,
        messages: list[dict],
        condition: dict,
        warmup: bool = False,
    ) -> dict:
        """Один запрос; счётчики сервера снимаются вне интервала замера.

        Запросы идут строго по одному, поэтому разность счётчиков до и
        после относится именно к этому запросу.
        """
        before = _snapshot(self.endpoint)
        record = timed_chat(
            self.client,
            self.endpoint,
            messages,
            self.output_tokens,
            request_id=request_id,
            condition=condition,
            ignore_eos=True,
            warmup=warmup,
        )
        record["server_delta"] = _delta(before, _snapshot(self.endpoint))
        files.append_jsonl(self.results_path, record)

        if record["status"] == "ok":
            console.info(
                f"{request_id}: TTFT={record['ttft_s'] * 1000:.0f} мс, "
                f"запрос={record['prompt_tokens']} ток., "
                f"ответ={record['completion_tokens']} ток."
            )
        else:
            console.info(f"{request_id}: ОШИБКА {record['error']}")
        return record


def _open_session(cfg: dict, run_dir: Path) -> _Session:
    """Сервер, паспорт в metadata.json, данные, токенизатор, клиент."""
    ecfg = cfg["exp1"]
    endpoint = load_endpoint(files.PROJECT_ROOT / cfg["endpoint_path"])
    info = server.collect_server_info(endpoint, cfg["server"])

    enabled, source = server.prefix_caching_state(
        info["kv_cache"], info["server"]["launch_command"]
    )
    expected = ecfg.get("prefix_caching")
    if expected is not None and enabled != expected:
        raise ValueError(
            f"Кеширование префикса на сервере: {enabled} ({source}), а "
            f"серия требует {expected}. Перезапустите сервер "
            "(scripts/lr1/start_vllm.sh с нужным флагом)"
        )

    metadata.add(
        run_dir,
        {
            "endpoint": endpoint.to_record(),
            **info,
            "prefix_caching": {"enabled": enabled, "source": source},
        },
    )
    console.info(
        f"сервер: {info['server']['name']} {info['server']['version']}, "
        f"кеширование префикса: {'вкл' if enabled else 'выкл'} ({source})"
    )

    text = data.load_verified_text()
    tokenizer = prompts.load_tokenizer(
        endpoint.model, info["model"]["revision"]
    )
    return _Session(
        endpoint,
        make_client(endpoint),
        prompts.PromptBuilder(text, tokenizer),
        run_dir / files.RESULTS_FILENAME,
        ecfg["output_tokens"],
    )


def _warmup(session: _Session, ecfg: dict) -> None:
    """Прогревочные запросы перед серией (в статистику не входят).

    По одному на каждую длину из warmup_tokens. Первый запрос новой длины
    может идти заметно дольше остальных (компиляция ядер, автотюнинг), и
    без прогрева на этой длине разовая добавка попала бы в первую
    измеряемую запись.
    """
    sizes = ecfg["warmup_tokens"]
    if isinstance(sizes, int):
        sizes = [sizes]
    for index, n_tokens in enumerate(sizes):
        messages, _ = session.build(WARMUP_REQUEST_ID + index, n_tokens)
        session.send(
            f"warmup-{n_tokens}",
            messages,
            {"phase": "warmup", "prompt_tokens_target": n_tokens},
            warmup=True,
        )


def _run_ttft_vs_length(ecfg: dict, session: _Session) -> None:
    """5 запросов на каждую длину, по одному.

    Порядок круговой (повтор 1 по всем длинам, затем повтор 2, ...):
    так длина запроса не смешивается с дрейфом частот и температуры GPU.
    """
    _warmup(session, ecfg)
    request_number = 1
    for repeat in range(1, ecfg["repeats"] + 1):
        for length in ecfg["prompt_lengths"]:
            messages, expected = session.build(request_number, length)
            session.send(
                f"len{length}-rep{repeat}",
                messages,
                {
                    "prompt_tokens_target": length,
                    "prompt_tokens_expected": expected,
                    "repeat": repeat,
                },
            )
            request_number += 1


def _run_prefix_cache(ecfg: dict, session: _Session) -> None:
    """Один и тот же запрос несколько раз подряд.

    Сообщения собираются один раз и отправляются без изменений: у всех
    попыток одна и та же метка и один и тот же текст.
    """
    _warmup(session, ecfg)
    length = ecfg["prompt_tokens"]
    for number in range(1, ecfg["n_prompts"] + 1):
        offset = session.builder.spread_offset(number, length)
        messages, expected = session.build(number, length, offset=offset)
        for attempt in range(1, ecfg["repeats"] + 1):
            session.send(
                f"prompt{number}-attempt{attempt}",
                messages,
                {
                    "prompt": number,
                    "attempt": attempt,
                    "prompt_tokens_target": length,
                    "prompt_tokens_expected": expected,
                },
            )


RUNNERS = {
    "ttft_vs_length": _run_ttft_vs_length,
    "prefix_cache": _run_prefix_cache,
}


def exp1_lab(cfg: dict, run_dir: Path) -> None:
    """Запуск одной серии опыта 1 по cfg['exp1']['mode'].

    Итоги metrics.json считаются из results.jsonl после серии.
    """
    ecfg = cfg["exp1"]
    mode = ecfg["mode"]
    if mode not in RUNNERS:
        raise ValueError(
            f"Неизвестный режим {mode!r}; доступны {sorted(RUNNERS)}"
        )

    session = _open_session(cfg, run_dir)
    RUNNERS[mode](ecfg, session)

    records = files.read_jsonl(session.results_path)
    metrics = exp1_analysis.summarize_run(mode, records, ecfg)
    files.write_json(run_dir / files.METRICS_FILENAME, metrics)

    console.info(
        f"запросов: {metrics['n_ok']} из {metrics['n_planned']} успешно"
    )
    if metrics.get("n_wrong_output_len"):
        console.info(
            f"ВНИМАНИЕ: длина ответа не равна {ecfg['output_tokens']} у "
            f"{metrics['n_wrong_output_len']} запросов"
        )
    if metrics.get("cache_hit_tokens_max"):
        console.info(
            "ВНИМАНИЕ: сервер взял из кеша префикса до "
            f"{metrics['cache_hit_tokens_max']:.0f} токенов за один запрос"
        )
    for number, entry in metrics.get("prompts", {}).items():
        if entry["cold_start_ok"] == 0:
            console.info(
                f"ВНИМАНИЕ: запрос {number} не был холодным: кеш уже "
                f"содержал его префикс ({entry['hit_tokens'][0]} токенов)"
            )
