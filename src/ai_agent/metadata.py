"""Работа с metadata.json"""

from pathlib import Path

from . import files, utils


def init(
    run_dir: Path,
    run_id: str,
    series: str,
    cfg: dict,
    libs: tuple[str, ...],
    sha256: dict[str, str],
):
    """Запись начального состояния metadata.json

    params:
        run_dir - директория запуска
        run_id - id запуска
        series - серия запуска
        cfg - словарь конфигурации
        libs - кортеж библиотек
        sha256 - словарь с sha256 файлов, ключ - название файла
    """
    m = get_metadata(
        run_id,
        series,
        cfg["runtime"]["num_threads"],
        cfg["seed"],
        libs,
        sha256,
    )
    files.write_json(run_dir / files.METADATA_FILENAME, m)


def get_metadata(
    run_id: str,
    series: str,
    num_threads: int,
    seed: int,
    libs: tuple[str, ...],
    sha256: dict[str, str],
) -> dict:
    """Получение начального словаря для metadata.json

    params:
        run_id - id запуска
        series - серия запуска
        num_threads - число потоков
        seed - сид случайности
        libs - список с названиями используемых библиотек
        sha256 - словарь с sha256 файлов, ключ - название файла

    return: словарь metadata со статусом running
    """
    return {
        "run_id": run_id,
        "series": series,
        "status": "running",
        "started_at": utils.time_now(),
        "finished_at": "-",
        "commit": utils.get_commit_hash(),
        "versions": utils.get_versions(libs),
        "num_threads": num_threads,
        "seed": seed,
        "sha256": sha256,
    }
