"""Чтение и обновление data/manifest.json."""

import hashlib
from pathlib import Path

BLOCK_SIZE = 8192  # размер блока файла


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
