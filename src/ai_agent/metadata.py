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
) -> dict:
    """Запись начального состояния metadata.json

    params:
        run_dir - директория запуска
        run_id - id запуска
        series - серия запуска
        cfg - словарь конфигурации
        libs - кортеж библиотек
        sha256 - словарь с sha256 файлов, ключ - название файла

    return: записанный словарь metadata
    """
    m = get_metadata(
        run_id,
        series,
        cfg.get("runtime", {}).get("num_threads"),
        cfg.get("seed"),
        libs,
        sha256,
    )
    files.write_json(run_dir / files.METADATA_FILENAME, m)
    return m


def update(run_dir: Path, status: str) -> None:
    """Обновление статуса в metadata.json

    params:
        run_dir - директория запуска
        status - обновленный статус (completed или failed)
    """
    metadata = files.read_json(run_dir / files.METADATA_FILENAME)
    metadata["status"] = status
    metadata["finished_at"] = utils.time_now()
    files.write_json(run_dir / files.METADATA_FILENAME, metadata)


def add(run_dir: Path, fields: dict) -> None:
    """Добавляет в metadata.json поля, известные только во время запуска.

    Например, ЛР1 после обращения к серверу записывает адрес, точное имя
    модели, ревизию, формат весов и параметры генерации (Правила, п. 3).
    Уже существующие поля (run_id, status, commit, ...) перезаписать нельзя.

    params:
        run_dir - директория запуска
        fields - добавляемые поля

    raise:
        ValueError, если поле уже есть в metadata.json
    """
    path = run_dir / files.METADATA_FILENAME
    metadata = files.read_json(path)
    clash = sorted(set(fields) & set(metadata))
    if clash:
        raise ValueError(f"Поля уже есть в metadata.json: {clash}")
    metadata.update(fields)
    files.write_json(path, metadata)


def get_metadata(
    run_id: str,
    series: str,
    num_threads: int | None,
    seed: int | None,
    libs: tuple[str, ...],
    sha256: dict[str, str],
) -> dict:
    """Получение начального словаря для metadata.json

    params:
        run_id - id запуска
        series - серия запуска
        num_threads - число потоков (None, если не задано в конфигурации)
        seed - сид случайности (None, если не задан в конфигурации)
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
