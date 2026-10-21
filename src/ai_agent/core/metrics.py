"""Работа с файлом metrics.json в runs/<run_id>/"""

from pathlib import Path
from .paths import METRICS_FILENAME, RESULTS_FILENAME
import json
from typing import Callable
from . import results


def compute(
    run_dir: Path,
    metric_fn: Callable[[list[dict]], dict],
) -> dict:
    """Вычисление метрик

    params:
        run_dir - путь к текущей директории runs/<run_id>/
        metric_fn - функция вычисления метрик из записей

    return: словарь с подсчитанными метриками
    """
    records = results.load(run_dir / RESULTS_FILENAME)
    metrics = metric_fn(records)
    return metrics


def write(run_dir: Path, metrics: dict) -> None:
    """Запись метрик в файл metrics.json

    params:
        run_dir - путь к текущей директории runs/<run_id>/
        metrics - записываемый словарь
    """
    metrics_path = run_dir / METRICS_FILENAME
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
