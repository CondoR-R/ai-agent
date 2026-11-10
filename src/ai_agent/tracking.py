"""Журнал MLflow: запись запуска каркаса в MLflow Tracking."""

import json
from pathlib import Path
from typing import Any

import mlflow

from . import files

# константы определены в files.py; имена оставлены для совместимости
MLFLOW_DB = files.MLFLOW_DB
MLFLOW_TRACKING_URI = files.MLFLOW_TRACKING_URI
MLFLOW_ARTIFACTS_DIR = files.MLFLOW_ARTIFACTS_DIR

# поля конфигурации, которые не логируются как параметры
EXCLUDED_PARAMS = ("tracked_libs",)
# итоговые метрики логируются с префиксом, чтобы не смешиваться с кривой эпох
FINAL_PREFIX = "final_"


def setup(experiment: str) -> str:
    """Подключает хранилище и выбирает эксперимент MLflow.

    Расположение артефактов задаётся явно: иначе оно зависит от каталога,
    из которого запущен процесс.

    params:
        experiment - имя эксперимента (например, lr0)

    return: идентификатор эксперимента
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    existing = mlflow.get_experiment_by_name(experiment)
    if existing is None:
        experiment_id = mlflow.create_experiment(
            experiment, artifact_location=MLFLOW_ARTIFACTS_DIR.as_uri()
        )
    else:
        experiment_id = existing.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Разворачивает вложенный словарь: {'train': {'lr': 1}} -> {'train.lr': 1}

    params:
        data - вложенный словарь
        prefix - префикс ключей

    return: плоский словарь
    """
    flat: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def start(run_id: str, cfg: dict, series: str, run_dir: Path) -> str:
    """Открывает запись MLflow, пишет параметры конфигурации и теги.

    Коммит берётся из metadata.json запуска. Если уже есть активная
    запись (родитель поиска Optuna), новая вкладывается в неё.

    params:
        run_id - идентификатор запуска, имя записи
        cfg - итоговая конфигурация
        series - серия эксперимента
        run_dir - каталог запуска

    return: идентификатор записи MLflow
    """
    meta = json.loads((run_dir / files.METADATA_FILENAME).read_text("utf-8"))

    setup(cfg["experiment"])
    active_run = mlflow.start_run(
        run_name=run_id, nested=mlflow.active_run() is not None
    )

    params = {k: v for k, v in cfg.items() if k not in EXCLUDED_PARAMS}
    mlflow.log_params(flatten(params))

    tags = {
        "work": cfg["experiment"],
        "series": series,
        "commit": meta["commit"],
        "run_id": run_id,
    }
    if cfg.get("seed") is not None:
        tags["seed"] = cfg["seed"]
    mlflow.set_tags(tags)
    return active_run.info.run_id


def log_metrics(metrics: dict[str, Any], step: int | None = None) -> None:
    """Логирует числовые метрики в активную запись MLflow.

    Без активной записи ничего не делает, поэтому код лабораторной
    можно запускать и вне каркаса.

    params:
        metrics - {имя: значение}; нечисловые значения пропускаются
        step - номер шага (эпохи), если метрика меняется по ходу запуска
    """
    if mlflow.active_run() is None:
        return
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            mlflow.log_metric(key, value, step=step)


def log_epoch(record: dict[str, Any]) -> None:
    """Логирует метрики одной эпохи (step = номер эпохи).

    params:
        record - запись эпохи: {epoch, train_loss, val_loss, val_acc, time_epoch}
    """
    log_metrics(
        {k: v for k, v in record.items() if k != "epoch"}, step=record["epoch"]
    )


def finish(run_dir: Path, error: str | None = None) -> None:
    """Логирует итоговые метрики, загружает файлы запуска, закрывает запись.

    Вложенные словари в metrics.json разворачиваются в имена через точку:
    {"by_len": {"256": {"ttft": 0.1}}} -> final_by_len.256.ttft.

    При падении запись закрывается со статусом FAILED, а файлы,
    записанные до падения, всё равно загружаются.

    params:
        run_dir - каталог запуска
        error - текст ошибки, если запуск упал
    """
    if mlflow.active_run() is None:
        return

    metrics_path = run_dir / files.METRICS_FILENAME
    if metrics_path.exists():
        final = flatten(json.loads(metrics_path.read_text("utf-8")))
        mlflow.log_metrics(
            {
                f"{FINAL_PREFIX}{k}": v
                for k, v in final.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
        )

    if error is not None:
        mlflow.set_tag("error_message", error)
    mlflow.log_artifacts(str(run_dir))
    mlflow.end_run(status="FAILED" if error else "FINISHED")
