"""Оркестратор запуска: одна команда — один run.

Собирает вместе инфраструктуру (paths, git_utils, config,
metadata, results, reproducibility) и предметную логику,
переданную извне через функции prepare_fn, load_records,
process_one, compute_metrics_fn. Модуль не знает, что именно
обрабатывается — изображения, утверждения SciFact или задачи
агента. Это делает его общим для всех лабораторных.
"""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import (
    ConfigDict,
    load_config,
    merge_configs,
    save_config,
    validate_config,
)
from .git_utils import check_clean_status, get_commit_hash
from .metadata import (
    collect_versions,
    update_status,
    write_error,
    write_metadata,
)
from .paths import (
    CONFIG_FILENAME,
    create_run_id_dir,
    get_run_id,
)
from .reproducibility import (
    enable_deterministic_mode,
    set_num_threads,
    set_seed,
)
from .results import append_result, compute_metrics


def _log_to_mlflow_stub(run_dir: Path, cfg: ConfigDict, work: str, series: str) -> None:
    """Заглушка регистрации в MLflow: ничего не делает."""
    return None


def run_experiment(
    config_path: str | Path,
    work: str,
    series: str,
    prepare_fn: Callable[[ConfigDict], Any],
    load_records: Callable[[ConfigDict], list],
    process_one: Callable[[Any, Any], dict],
    compute_metrics_fn: Callable[[list[dict]], dict],
    required_fields: tuple[str, ...],
    data_hashes: dict[str, str],
    cli_overrides: dict | None = None,
) -> Path:
    """Запуск одного эксперимента: одна конфигурация — один run.

    Оркестрирует полный цикл: проверяет git и конфиг, создаёт каталог
    ``runs/<run_id>/``, сохраняет итоговую конфигурацию и метаданные,
    прогоняет записи через ``process_one`` с инкрементальной записью
    результатов, считает метрики из сохранённого файла и фиксирует
    статус запуска. Не знает, что именно обрабатывается — изображения,
    утверждения или задачи агента.

    params:
        config_path - путь к YAML-файлу конфигурации
        work - имя работы (``lr0``, ``lr1``, ..., ``lr8``)
        series - имя серии (``base_seeds``, ``final``, ``search``)
        prepare_fn - функция подготовки, вызывается один раз с конфигом
        load_records - функция, возвращающая список записей
        process_one - функция обработки одной записи,
            получает запись и результат ``prepare_fn``
        compute_metrics_fn - функция подсчёта метрик по списку записей
        required_fields - кортеж обязательных полей каждой записи
        data_hashes - словарь ``путь → SHA-256`` для входных файлов
            (сохраняется в ``metadata.json`` как ``data_sha256``)
        cli_overrides - переопределения конфигурации из CLI,
            по умолчанию ``None`` (без переопределений)

    return: путь к каталогу запуска ``runs/<run_id>/``

    raise:
        SystemExit - репозиторий грязный (есть незакоммиченные файлы),
            проверка выполняется до создания каталога
        ValueError - конфигурация невалидна, проверка выполняется
            до создания каталога
        Exception - любая ошибка на этапе подготовки или обработки;
            traceback сохраняется в ``error.txt``, статус меняется
            на ``failed``, затем исключение пробрасывается дальше
    """
    # 1. Проверки до создания каталога
    check_clean_status()
    commit = get_commit_hash()

    base_cfg = load_config(config_path)
    cfg = merge_configs(base_cfg, cli_overrides or {})
    validate_config(cfg)

    # 2. Воспроизводимость
    set_seed(cfg["seed"])
    set_num_threads(cfg["runtime"]["num_threads"])
    enable_deterministic_mode()

    # 3. Создание каталога и первичных файлов
    run_id = get_run_id(work, series, cfg["seed"])
    run_dir = create_run_id_dir(run_id)

    save_config(run_dir / CONFIG_FILENAME, cfg)

    metadata_dict = {
        "run_id": run_id,
        "work": work,
        "series": series,
        "status": "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "commit": commit,
        "device": cfg["runtime"]["device"],
        "seed": cfg["seed"],
        "split_seed": cfg["split_seed"],
        "num_threads": cfg["runtime"]["num_threads"],
        "data_sha256": dict(data_hashes),
        "versions": collect_versions(cfg["runtime"]["tracked_libs"]),
    }
    write_metadata(run_dir, metadata_dict)

    # 4. Основная работа (ошибки логируются в каталог run)
    try:
        context = prepare_fn(cfg)
        records = load_records(cfg)

        for record in records:
            result = process_one(record, context)
            append_result(run_dir, result, required_fields)

        compute_metrics(run_dir, compute_metrics_fn)
        update_status(run_dir, "completed")

    except BaseException as exc:
        write_error(run_dir, exc)
        try:
            update_status(run_dir, "failed")
        except Exception:
            pass
        try:
            _log_to_mlflow_stub(run_dir, cfg, work, series)
        except Exception:
            pass
        raise

    # 5. Регистрация в MLflow (пока заглушка)
    _log_to_mlflow_stub(run_dir, cfg, work, series)

    return run_dir
