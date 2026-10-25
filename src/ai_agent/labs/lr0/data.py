"""Разбиение данных, сохранение разбиения, запись в manifest.json"""

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from ai_agent import paths


def load_data() -> tuple[np.ndarray, np.ndarray]:
    """Загрузка датасета из load_digits. Признаки делятся на 16.

    return: кортеж нормализованных признаков и меток
    """
    features, labels = load_digits(return_X_y=True)
    features = np.asarray(features, dtype=np.float32) / 16.0
    labels = np.asarray(labels, dtype=np.int64)
    return features, labels


def get_split(
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.25,
    split_seed: int = 0,
) -> dict[str, list[int]]:
    """Разбиение на train/val/test со стратификацией

    params:
        y - массив меток
        test_size - размер тестовой выборки от всей (по умолчанию 0.2)
        val_size - размер валидационной выборки от остатка,
            после отделения тестовой (по умолчанию 0.25)
        split_seed - сид сплита (по умолчанию 0)

    return: словарь со сплитом
    """
    idx = np.arange(len(y))
    rest, test = train_test_split(
        idx, test_size=test_size, stratify=y, random_state=split_seed
    )
    train, val = train_test_split(
        rest, test_size=val_size, stratify=y[rest], random_state=split_seed
    )
    return {
        "train": train.tolist(),
        "val": val.tolist(),
        "test": test.tolist(),
    }


def save_split(path: Path, split: dict[str, list[int]]) -> None:
    """Сохраняет разбиение в JSON по переданному пути

    params:
        path - путь к файлу
        split - словарь с разбиением
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(split, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def file_sha256(path: str | Path) -> str:
    """Вычисляет SHA-256 файла

    params:
        path - путь к файлу

    return: hash-строку для файла
    """
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_manifest(manifest: dict, path: Path = paths.MANIFEST_PATH) -> None:
    """Записывает информацию о данных в manifest.json

    params:
        manifest - словрь с данными для записи
        path - путь к файлу (по умолчанию paths.MANIFEST_PATH)"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def validate_split(y: np.ndarray, split: dict[str, list[int]]):
    """Проверка разбиения: части не пересекаются, в сумме дают 1797, в каждой 10 классов

    params:
        y - массив меток
        split - словарь с разбиением

    raise: ValueError, при некорректных значениях
    """
    parts = ['train', 'val', 'test']
    all_idx = np.concatenate(
        [np.asarray(split[p], dtype=np.int64) for p in parts]
    )
    if len(all_idx) != len(np.unique(all_idx)):
        raise ValueError("Индексы присутствуют в разных частях: утечка данных")

    if len(all_idx) != len(y):
        raise ValueError(
            f"В частях {len(all_idx)} объектов, должно быть {len(y)}"
        )

    all_classes = set(np.unique(y).tolist())
    for p in parts:
        part_classes = set(y[split[p]].tolist())
        if part_classes != all_classes:
            missing = sorted(all_classes - part_classes)
            raise ValueError(f"В части {p} пропущены классы: {missing}")

