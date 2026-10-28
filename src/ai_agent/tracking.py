"""Журнал MLflow: запись запуска каркаса в MLflow Tracking."""

import json
from pathlib import Path
from typing import Any

import mlflow

from . import files

MLFLOW_DB = files.PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB.as_posix()}"
MLFLOW_ARTIFACTS_DIR = files.PROJECT_ROOT / "mlartifacts"

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


def start(run_id: str, cfg: dict, series: str, run_dir: Path) -> None:
    """Открывает запись MLflow, пишет параметры конфигурации и теги.

    Коммит берётся из metadata.json запуска. Если уже есть активная
    запись (родитель поиска Optuna), новая вкладывается в неё.

    params:
        run_id - идентификатор запуска, имя записи
        cfg - итоговая конфигурация
        series - серия эксперимента
        run_dir - каталог запуска
    """
    meta = json.loads((run_dir / files.METADATA_FILENAME).read_text("utf-8"))

    setup(cfg["experiment"])
    mlflow.start_run(run_name=run_id, nested=mlflow.active_run() is not None)

    params = {k: v for k, v in cfg.items() if k not in EXCLUDED_PARAMS}
    mlflow.log_params(flatten(params))
    mlflow.set_tags(
        {
            "work": cfg["experiment"],
            "series": series,
            "seed": cfg["seed"],
            "commit": meta["commit"],
            "run_id": run_id,
        }
    )


def log_epoch(record: dict[str, Any]) -> None:
    """Логирует метрики одной эпохи (step = номер эпохи).

    Без активной записи ничего не делает, поэтому обучение можно
    запускать и вне каркаса.

    params:
        record - запись эпохи: {epoch, train_loss, val_loss, val_acc, time_epoch}
    """
    if mlflow.active_run() is None:
        return
    step = record["epoch"]
    for key, value in record.items():
        if key != "epoch" and isinstance(value, (int, float)):
            mlflow.log_metric(key, value, step=step)


def finish(run_dir: Path, error: str | None = None) -> None:
    """Логирует итоговые метрики, загружает файлы запуска, закрывает запись.

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
        final = json.loads(metrics_path.read_text("utf-8"))
        mlflow.log_metrics(
            {
                f"{FINAL_PREFIX}{k}": v
                for k, v in final.items()
                if isinstance(v, (int, float))
            }
        )

    if error is not None:
        mlflow.set_tag("error_message", error)
    mlflow.log_artifacts(str(run_dir))
    mlflow.end_run(status="FAILED" if error else "FINISHED")
