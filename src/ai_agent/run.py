"""Один запуск эксперимента"""

import time
from collections.abc import Callable
from pathlib import Path

from . import console, files, metadata, reproducibility, tracking, utils

# файлы, которые должны появиться в каталоге запуска
ARTIFACT_FILES = (
    files.CONFIG_FILENAME,
    files.METADATA_FILENAME,
    "history.csv",
    "predictions.csv",
    files.METRICS_FILENAME,
)


def _config_summary(cfg: dict) -> str:
    """Краткая выжимка конфигурации для вывода в терминал."""
    model = cfg["model"]
    train = cfg["train"]
    return (
        f"model={model['n_layers']}×{model['hidden']} "
        f"(dropout={model['dropout']}), train={train['epochs']} эпох, "
        f"batch={train['batch_size']}, lr={train['lr']}"
    )


def _metrics_summary(metrics: dict) -> str:
    """Форматирует итоговые метрики для вывода."""
    parts = []
    if "val_acc" in metrics:
        parts.append(f"val_acc={metrics['val_acc']:.4f}")
    if "val_loss" in metrics:
        parts.append(f"val_loss={metrics['val_loss']:.4f}")
    if "test_acc" in metrics:
        parts.append(f"test_acc={metrics['test_acc']:.4f}")
    if "train_time" in metrics:
        parts.append(f"train_time={metrics['train_time']:.2f} с")
    return ", ".join(parts)


def _artifacts(run_dir: Path) -> str:
    """Перечисляет созданные файлы каталога запуска."""
    return ", ".join(
        name for name in ARTIFACT_FILES if (run_dir / name).exists()
    )


def _error_detail(exc: BaseException) -> str:
    """Форматирует исключение как 'Тип: сообщение'."""
    message = str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _handle_failure(
    exc: BaseException,
    step_number: int,
    step_title: str,
    run_dir: Path | None,
    mlflow_run_id: str | None,
) -> None:
    """Печатает ошибку и сохраняет всё, что успели записать."""
    console.error(
        f"Шаг {step_number}/{console.TOTAL_STEPS} ({step_title}): "
        f"{_error_detail(exc)}"
    )
    if run_dir is None:
        console.info("артефакты не созданы")
        return

    console.info(f"каталог: {run_dir}")

    if (run_dir / files.METADATA_FILENAME).exists():
        try:
            metadata.update(run_dir, "failed")
        except Exception:
            pass
        else:
            console.info("metadata.json: failed")

    error_path = run_dir / files.ERROR_FILENAME
    try:
        files.write_error(error_path, exc)
    except Exception:
        pass
    else:
        console.info(f"трейсбек: {error_path}")

    try:
        tracking.finish(run_dir, error=_error_detail(exc))
    except Exception:
        console.info("MLflow: не удалось закрыть запись")
    else:
        if mlflow_run_id is not None:
            console.info(
                f"MLflow: run_id={mlflow_run_id}, статус FAILED"
            )


def run(
    config_path: Path,
    cli_cfg: dict,
    exp: int,
    series: str,
    sha256: dict[str, str],
    lab_fn: Callable[[dict, Path, bool], None],
    eval_test: bool,
) -> Path:
    """Запуск эксперимента

    params:
        config_path - путь к файлу конфигурации
        cli_cfg - словарь с переопрелениями конфигурации из CLI
        exp - номер эксперимента
        series - серия эксперимента
        sha256 - словарь с sha256 файлов
        lab_fn - функция запуска лабораторной
        eval_test - флаг тестовых данных

    return: путь к директории запуска
    """
    started = time.perf_counter()
    run_dir: Path | None = None
    mlflow_run_id: str | None = None
    step_number = 1
    step_title = "Git"

    console.step(1, "Git: проверка чистого дерева")
    try:
        utils.check_git()
    except SystemExit as exc:
        lines = [
            line.strip() for line in str(exc).splitlines() if line.strip()
        ]
        if lines:
            console.error(lines[0])
            for line in lines[1:]:
                console.info(line)
        raise SystemExit(1) from None

    try:
        step_number, step_title = 2, "конфиг"
        console.step(2, "Конфиг: чтение и слияние с CLI")
        file_cfg = files.read_yaml(config_path)
        cfg = utils.merge_cfgs(file_cfg, cli_cfg)
        console.info(
            f"experiment={cfg['experiment']}, series={series}, "
            f"exp={exp}, eval_test={eval_test}"
        )
        if cli_cfg:
            console.info(f"CLI-переопределения: {cli_cfg}")

        step_number, step_title = 3, "каталог запуска"
        console.step(3, "Создание каталога запуска")
        run_id = files.get_run_id(cfg["experiment"], exp)
        run_dir = files.create_run_dir(run_id)
        console.info(f"runs/{run_id}/")

        step_number, step_title = 4, "config.yaml"
        console.step(4, "Сохранение config.yaml")
        files.write_yaml(run_dir / files.CONFIG_FILENAME, cfg)
        console.info(_config_summary(cfg))

        step_number, step_title = 5, "воспроизводимость"
        console.step(5, "Воспроизводимость")
        reproducibility.set_seed(
            cfg["seed"], cfg["runtime"]["num_threads"]
        )
        console.info(
            f"seed={cfg['seed']}, "
            f"num_threads={cfg['runtime']['num_threads']}, "
            "deterministic=True"
        )

        step_number, step_title = 6, "metadata.json"
        console.step(6, "metadata.json")
        meta = metadata.init(
            run_dir,
            run_id,
            series,
            cfg,
            ("torch", "scikit-learn", "optuna", "mlflow"),
            sha256,
        )
        versions = meta["versions"]
        console.info(
            f"commit={meta['commit'][:7]}, python={versions['python']}, "
            f"torch={versions['torch']}"
        )
        console.info(
            f"sklearn={versions['scikit-learn']}, "
            f"optuna={versions['optuna']}, mlflow={versions['mlflow']}"
        )

        step_number, step_title = 7, "MLflow"
        console.step(7, "MLflow")
        mlflow_run_id = tracking.start(run_id, cfg, series, run_dir)
        console.info(
            f"эксперимент={cfg['experiment']}, run_id={mlflow_run_id}, "
            "sqlite:///mlflow.db"
        )

        step_number, step_title = 8, "лабораторная"
        console.step(8, "Лабораторная: старт")
        lab_started = time.perf_counter()
        lab_fn(cfg, run_dir, eval_test)
        console.info(
            "лабораторная: готово за "
            f"{time.perf_counter() - lab_started:.2f} с"
        )

        metadata.update(run_dir, "completed")
        tracking.finish(run_dir)
    except BaseException as exc:
        _handle_failure(
            exc, step_number, step_title, run_dir, mlflow_run_id
        )
        raise

    elapsed = time.perf_counter() - started
    metrics_path = run_dir / files.METRICS_FILENAME
    metrics = files.read_json(metrics_path) if metrics_path.exists() else {}

    console.ok(f"Запуск завершён за {elapsed:.2f} с")
    if metrics:
        console.info(f"метрики: {_metrics_summary(metrics)}")
    else:
        console.info("метрики: metrics.json не найден")
    console.info(f"артефакты: {_artifacts(run_dir)}")
    console.info(f"каталог: {run_dir}")
    console.info(f"MLflow: run_id={mlflow_run_id}, статус FINISHED")

    return run_dir
