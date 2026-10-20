"""Работа с файлами results.jsonl и metrics.json"""

import json
from collections.abc import Callable
from pathlib import Path

from .paths import METRICS_FILENAME, RESULTS_FILENAME


def append_result(run_dir: Path, record: dict, required_fields: tuple[str, ...]) -> None:
    """Добавляет строку записи в results.jsonl в конец файла

    params:
        run_dir - путь к текущей директории runs/<run_id>/
        record - словарь с записью в results.jsonl
        required_fields - кортеж ожидаемых полей
    """
    validate_result(record, required_fields)
    with open(run_dir / RESULTS_FILENAME, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def load_results(run_dir: Path) -> list[dict]:
    """Чтение results.jsonl

    params:
        run_dir - путь к текущей директории runs/<run_id>/

    return: список словарей с записями.

    raise:
        ValueError, при ошибки чтения строки
    """
    results: list[dict] = []
    with open(run_dir / RESULTS_FILENAME, 'r', encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"Ошибка при чтении строки {i}: {err.msg}. Строка: {line[:100]!r}")
            results.append(record)
    return results


def validate_result(record: dict, required_fields: tuple[str, ...]) -> None:
    """Валидация записи для results.jsonl
    params:
        record - словарь с записью
        required_fields - кортеж ожидаемых полей

    raise:
        TypeError, при неправильном типе record
        ValueError, при отсутствие обязательного поля
    """
    # record - словарь
    if not isinstance(record, dict):
        raise TypeError("record должен иметь тип dict")
    # Наличие обязательных полей
    for field in required_fields:
        if field not in record:
            raise ValueError(f"Отсутствует обязательное поле {field}")


def compute_metrics(
    run_dir: Path,
    metric_fn: Callable[[list[dict]], dict],
) -> None:
    """Вычисление метрик и запись в файл metrics.json

    params:
        run_dir - путь к текущей директории runs/<run_id>/
        metric_fn - функция вычисления метрик из записей
    """
    records = load_results(run_dir)
    metrics = metric_fn(records)
    metrics_path = run_dir / METRICS_FILENAME
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
