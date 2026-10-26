"""Работа с файлом <run_id>/metadata.json"""

import json
import platform
from importlib.metadata import version
from pathlib import Path

from ai_agent import git_utils, time


def get_versions(libs: list[str]) -> dict[str, str]:
    """Получение версий Python и библиотек

    params:
        libs - список библиотек
    return: словарь с версиями
    """
    v = {
        "python": platform.version(),
    }
    for lib in libs:
        v[lib] = version(lib)
    return v


def get_metadata(
    run_id: str,
    series: str,
    num_threads: int,
    seed: int,
    libs: list[str],
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
        "started_at": time.now(),
        "finished_at": "-",
        "commit": git_utils.get_commit_hash(),
        "versions": get_versions(libs),
        "num_threads": num_threads,
        "seed": seed,
        "sha256": sha256,
    }


def write(path: Path, metadata: dict) -> None:
    """Запись в файл metadata.json

    params:
        path - путь к файлу
        metadata - записываемый словарь
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def update(path: Path): ...
