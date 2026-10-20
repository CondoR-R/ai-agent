"""Чтение и обновление data/manifest.json."""

import hashlib
import json
from pathlib import Path

BLOCK_SIZE = 8192  # размер блока файла

MANIFEST_DATASETS_KEY = "datasets"


def file_sha256(path: str | Path) -> str:
    """Вычисляет SHA-256 файла

    params:
        path - путь к файлу

    return: hash-строку для файла
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(BLOCK_SIZE)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def load_manifest(path: str | Path) -> dict:
    """Читает манифест данных из файла.

    Если файла нет, возвращает пустой скелет — это позволяет
    первому запуску создания данных не падать.

    params:
        path - путь к data/manifest.json

    return: словарь с ключом ``datasets`` (список записей)

    raise:
        ValueError - файл существует, но верхний уровень не словарь
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
    data.setdefault(MANIFEST_DATASETS_KEY, [])
    return data


def update_manifest(path: str | Path, name: str, entry: dict) -> None:
    """Добавляет или заменяет запись набора данных по имени.

    Замена, а не добавление: повторный запуск скрипта подготовки
    не должен раздувать манифест дубликатами.

    params:
        path - путь к data/manifest.json
        name - имя набора (уникальный идентификатор)
        entry - словарь с полями набора (source, version, rule,
            sizes, files и т. д.)
    """
    p = Path(path)
    manifest = load_manifest(p)
    datasets = [
        d for d in manifest[MANIFEST_DATASETS_KEY] if d.get("name") != name
    ]
    datasets.append({"name": name, **entry})
    manifest[MANIFEST_DATASETS_KEY] = datasets

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def find_entry(path: str | Path, name: str) -> dict | None:
    """Ищет запись набора в манифесте по имени.

    params:
        path - путь к data/manifest.json
        name - имя набора

    return: словарь с полями набора или None, если не найдено
    """
    for d in load_manifest(path)[MANIFEST_DATASETS_KEY]:
        if d.get("name") == name:
            return d
    return None
