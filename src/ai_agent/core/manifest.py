"""Чтение и обновление data/manifest.json."""

import hashlib
import json
import os
from pathlib import Path

from .paths import PROJECT_ROOT

MANIFEST_DATASETS_KEY = "datasets"

ENTRY_FIELDS = ("name", "source", "version", "selection", "files")
SELECTION_FIELDS = ("total", "taken", "excluded")
FILE_FIELDS = ("path", "sha256", "records")


def file_sha256(path: str | Path) -> str:
    """Вычисляет SHA-256 файла

    params:
        path - путь к файлу

    return: hash-строку для файла
    """
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def validate_entry(entry: dict) -> None:
    """Проверяет запись набора данных (Правила, п. 5)

    Ожидаемая структура:
        name - уникальное имя набора
        source - источник (пакет, репозиторий)
        version - версия пакета или ревизия репозитория
        selection - {total, taken, excluded[, reason]}: сколько было,
            взято, исключено и почему
        files - список {path, sha256, records}: итоговые файлы

    params:
        entry - проверяемая запись

    raise:
        TypeError, при неправильном типе данных
        ValueError, при недопустимых значениях
        KeyError, при отсутствии обязательного поля
    """
    if not isinstance(entry, dict):
        raise TypeError("Запись манифеста должна быть словарём (dict)")

    for field in ENTRY_FIELDS:
        if field not in entry:
            raise KeyError(f"Отсутствует обязательное поле {field}")

    sel = entry["selection"]
    if not isinstance(sel, dict):
        raise TypeError("Поле selection должно быть словарём (dict)")
    for field in SELECTION_FIELDS:
        if field not in sel:
            raise KeyError(f"Отсутствует обязательное поле selection.{field}")
    if sel["taken"] + sel["excluded"] != sel["total"]:
        raise ValueError(
            "В selection должно выполняться taken + excluded = total, "
            f"получено {sel['taken']} + {sel['excluded']} != {sel['total']}"
        )
    if sel["excluded"] > 0 and not sel.get("reason"):
        raise ValueError("При excluded > 0 нужно указать selection.reason")

    files = entry["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("Поле files должно быть непустым списком")
    for item in files:
        for field in FILE_FIELDS:
            if field not in item:
                raise KeyError(f"Отсутствует обязательное поле files[].{field}")
        if len(item["sha256"]) != 64:
            raise ValueError(f"Некорректный SHA-256 у файла {item['path']}")
        if item["records"] <= 0:
            raise ValueError(f"Пустой набор: {item['path']}")


def load(path: str | Path) -> dict:
    """Читает манифест данных из файла.

    Если файла нет, возвращает пустой скелет.

    params:
        path - путь к data/manifest.json

    return: словарь с ключом ``datasets`` (список записей)

    raise:
        ValueError - верхний уровень не словарь, datasets не список
            или имена наборов повторяются
        json.JSONDecodeError - файл существует, но невалидный JSON
    """
    p = Path(path)
    if not p.exists():
        return {MANIFEST_DATASETS_KEY: []}
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(
            f"manifest.json должен содержать объект на верхнем уровне, "
            f"получено: {type(data).__name__}"
        )
    datasets = data.setdefault(MANIFEST_DATASETS_KEY, [])
    if not isinstance(datasets, list):
        raise ValueError(f"Поле {MANIFEST_DATASETS_KEY} должно быть списком")
    names = [d.get("name") for d in datasets]
    if len(names) != len(set(names)):
        raise ValueError("В манифесте повторяются имена наборов данных")
    return data


def _write(path: Path, manifest: dict) -> None:
    """Записывает манифест атомарно: через временный файл и замену"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def update(path: str | Path, name: str, entry: dict) -> None:
    """Добавляет или заменяет запись набора данных по имени.

    Запись проверяется до записи в файл. При замене позиция
    записи в списке сохраняется.

    params:
        path - путь к data/manifest.json
        name - имя набора (уникальный идентификатор)
        entry - словарь с полями набора (без name)

    raise:
        TypeError, ValueError - если запись не прошла validate_entry
    """
    p = Path(path)
    new_entry = {"name": name, **entry}
    validate_entry(new_entry)

    manifest = load(p)
    datasets = manifest[MANIFEST_DATASETS_KEY]
    for i, d in enumerate(datasets):
        if d.get("name") == name:
            datasets[i] = new_entry
            break
    else:
        datasets.append(new_entry)

    _write(p, manifest)


def find_entry(path: str | Path, name: str) -> dict | None:
    """Ищет запись набора в манифесте по имени.

    params:
        path - путь к data/manifest.json
        name - имя набора

    return: словарь с полями набора или None, если не найдено
    """
    for d in load(path)[MANIFEST_DATASETS_KEY]:
        if d.get("name") == name:
            return d
    return None


def verify_files(entry: dict, base_dir: Path = PROJECT_ROOT) -> None:
    """Проверяет, что файлы набора не изменились: пересчитывает SHA-256
    и сравнивает с записанным в манифесте.

    params:
        entry - запись набора из манифеста
        base_dir - каталог, от которого отсчитываются пути files[].path

    raise:
        FileNotFoundError, если файла нет
        ValueError, если хеш не совпал
    """
    for item in entry["files"]:
        actual = file_sha256(base_dir / item["path"])
        if actual != item["sha256"]:
            raise ValueError(
                f"Файл {item['path']} изменён: ожидался SHA-256 "
                f"{item['sha256'][:12]}…, получен {actual[:12]}…"
            )
