"""Эксперимент 2 ЛР1: длина ответа и скорость генерации (п. 4.2)

Один запрос на 1024 токена, ответы 16, 64, 256 и 1024 токена с
ignore_eos, по 5 запросов на длину. Ответ ровно заданной длины
проверяется по usage.completion_tokens.
"""

from pathlib import Path

from ai_agent import console, files, metadata

from . import data, exp2_analysis, prompts, server
from .client import make_client, timed_chat
from .endpoint import load_endpoint

# номер прогревочного запроса не пересекается с номерами измеряемых
WARMUP_REQUEST_ID = 900_000


def exp2_lab(cfg: dict, run_dir: Path) -> None:
    """Запускает серию опыта 2 и пишет metrics.json.

    progress: прогрев, затем круговой порядок «повтор × длина ответа»,
    чтобы длина не смешивалась с дрейфом частот и температуры GPU.
    """
    ecfg = cfg["exp2"]
    results_path = run_dir / files.RESULTS_FILENAME

    endpoint = load_endpoint(files.PROJECT_ROOT / cfg["endpoint_path"])
    info = server.collect_server_info(endpoint, cfg["server"])
    metadata.add(
        run_dir,
        {"endpoint": endpoint.to_record(), **info},
    )
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
    client = make_client(endpoint)

    warmup_messages, _ = builder.build(WARMUP_REQUEST_ID, ecfg["prompt_tokens"])
    warmup = timed_chat(
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
    for repeat in range(1, ecfg["repeats"] + 1):
        for output_tokens in ecfg["output_lengths"]:
            offset = builder.spread_offset(
                request_number, ecfg["prompt_tokens"]
            )
            messages, expected = builder.build(
                request_number,
                ecfg["prompt_tokens"],
                offset=offset,
            )
            record = timed_chat(
                client,
                endpoint,
                messages,
                output_tokens,
                request_id=f"out{output_tokens}-rep{repeat}",
                condition={
                    "phase": "run",
                    "prompt_tokens_target": ecfg["prompt_tokens"],
                    "prompt_tokens_expected": expected,
                    "output_tokens_target": output_tokens,
                    "repeat": repeat,
                },
                ignore_eos=True,
            )
            files.append_jsonl(results_path, record)
            if record["status"] == "ok":
                console.info(
                    f"ответ {output_tokens}: e2e="
                    f"{record['e2e_s'] * 1000:.0f} мс, "
                    f"TTFT={record['ttft_s'] * 1000:.0f} мс, "
                    f"TPOT={record['tpot_s'] * 1000:.0f} мс, "
                    f"токенов={record['completion_tokens']}"
                )
            else:
                console.info(f"ответ {output_tokens}: ОШИБКА {record['error']}")
            request_number += 1

    records = files.read_jsonl(results_path)
    metrics = exp2_analysis.summarize_run(records, cfg)
    files.write_json(run_dir / files.METRICS_FILENAME, metrics)

    console.info(
        f"запросов: {metrics['n_ok']} из {metrics['n_planned']} успешно"
    )
    if metrics["n_wrong_output_len"]:
        console.info(
            "ВНИМАНИЕ: длина ответа не совпала с целевой у "
            f"{metrics['n_wrong_output_len']} запросов"
        )
    fit = metrics["e2e_fit"]
    if fit:
        console.info(
            "прямая e2e: свободный член "
            f"{fit['intercept_s'] * 1000:.0f} мс, наклон "
            f"{fit['slope_s_per_token'] * 1000:.1f} мс/токен"
        )
