"""Один запуск эксперимента"""

import time
from collections.abc import Callable
from pathlib import Path

from . import console, files, metadata, reproducibility, tracking, utils

# библиотеки, версии которых пишутся в metadata.json в любом запуске
DEFAULT_LIBS = ("mlflow",)


def _config_summary(cfg: dict) -> str:
    """Выжимка конфигурации по умолчанию: скалярные поля верхнего уровня.

    Лабораторная может передать свою функцию через config_summary.
    """
    return ", ".join(
        f"{key}={value}"
        for key, value in cfg.items()
        if not isinstance(value, (dict, list))
    )


def _metrics_summary(metrics: dict) -> str:
    """Выжимка итоговых метрик по умолчанию: числа верхнего уровня.

    Лабораторная может передать свою функцию через metrics_summary.
    """
    return ", ".join(
        f"{key}={value:.4g}"
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )


def _artifacts(run_dir: Path) -> str:
    """Перечисляет файлы, которые лежат в каталоге запуска."""
    return ", ".join(sorted(p.name for p in run_dir.iterdir() if p.is_file()))


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
            console.info(f"MLflow: run_id={mlflow_run_id}, статус FAILED")


def run(
    config_path: Path,
    cli_cfg: dict,
    exp: int,
    series: str,
    sha256: dict[str, str],
    lab_fn: Callable[[dict, Path], None],
    *,
    libs: tuple[str, ...] = (),
    config_summary: Callable[[dict], str] | None = None,
    metrics_summary: Callable[[dict], str] | None = None,
) -> Path:
    """Запуск эксперимента

    params:
        config_path - путь к файлу конфигурации
        cli_cfg - словарь с переопрелениями конфигурации из CLI
        exp - номер эксперимента
        series - серия эксперимента
        sha256 - словарь с sha256 файлов
        lab_fn - функция лабораторной: lab_fn(cfg, run_dir). Всё, что нужно
                 сверх этого (например, eval_test в ЛР0), привязывается
                 через functools.partial
        libs - библиотеки лабораторной, версии которых пишутся в metadata
               (к ним добавляются mlflow и tracked_libs из конфигурации)
        config_summary - функция cfg -> строка для вывода в терминал
        metrics_summary - функция metrics -> строка для вывода в терминал

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
        lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
        if lines:
            console.error(lines[0])
            for line in lines[1:]:
                console.info(line)
        raise SystemExit(1) from None

    try:
        step_number, step_title = 2, "конфиг"
        console.step(2, "Конфиг: чтение и слияние с CLI")
        file_cfg = files.read_yaml(config_path)
        utils.check_overrides(file_cfg, cli_cfg)
        cfg = utils.merge_cfgs(file_cfg, cli_cfg)
        console.info(
            f"experiment={cfg['experiment']}, series={series}, exp={exp}"
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
        console.info((config_summary or _config_summary)(cfg))

        step_number, step_title = 5, "воспроизводимость"
        console.step(5, "Воспроизводимость")
        seed = cfg.get("seed")
        num_threads = cfg.get("runtime", {}).get("num_threads", 1)
        if seed is None:
            console.info(
                "seed не задан в конфигурации: генераторы не фиксируются"
            )
        else:
            reproducibility.set_seed(seed, num_threads)
            console.info(
                f"seed={seed}, num_threads={num_threads}, deterministic=True"
            )

        step_number, step_title = 6, "metadata.json"
        console.step(6, "metadata.json")
        all_libs = tuple(
            dict.fromkeys(
                (*DEFAULT_LIBS, *libs, *(cfg.get("tracked_libs") or ()))
            )
        )
        meta = metadata.init(run_dir, run_id, series, cfg, all_libs, sha256)
        versions = meta["versions"]
        console.info(
            f"commit={meta['commit'][:7]}, python={versions['python']}"
        )
        console.info(
            ", ".join(f"{k}={v}" for k, v in versions.items() if k != "python")
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
        lab_fn(cfg, run_dir)
        console.info(
            f"лабораторная: готово за {time.perf_counter() - lab_started:.2f} с"
        )

        metadata.update(run_dir, "completed")
        tracking.finish(run_dir)
    except BaseException as exc:
        _handle_failure(exc, step_number, step_title, run_dir, mlflow_run_id)
        raise

    elapsed = time.perf_counter() - started
    metrics_path = run_dir / files.METRICS_FILENAME
    metrics = files.read_json(metrics_path) if metrics_path.exists() else {}

    console.ok(f"Запуск завершён за {elapsed:.2f} с")
    if metrics:
        console.info(
            f"метрики: {(metrics_summary or _metrics_summary)(metrics)}"
        )
    else:
        console.info("метрики: metrics.json не найден")
    console.info(f"артефакты: {_artifacts(run_dir)}")
    console.info(f"каталог: {run_dir}")
    console.info(f"MLflow: run_id={mlflow_run_id}, статус FINISHED")

    return run_dir
