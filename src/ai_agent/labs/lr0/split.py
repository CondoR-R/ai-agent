"""Разбиение данных и проверки корректности."""

import numpy as np
from sklearn.model_selection import train_test_split


def make_stratified_split(
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.25,
    random_state: int = 0,
) -> dict[str, list[int]]:
    """Стратифицированное разбиение: сначала test, затем val от остатка.

    params:
        y - массив меток
        test_size - доля тестовых индексов от всего датасета
        val_size - доля валидационных индексов от тренировочной выборки
        random_state - сид разбиения

    return: словарь с индексами для train, val, test.

    Для load_digits даёт 1077 / 360 / 360.
    """
    idx = np.arange(len(y))
    rest, test = train_test_split(
        idx, test_size=test_size, stratify=y, random_state=random_state
    )
    train, val = train_test_split(
        rest, test_size=val_size, stratify=y[rest], random_state=random_state
    )

    return {
        "train": train.tolist(),
        "val": val.tolist(),
        "test": test.tolist(),
    }


def validate(split: dict[str, list[int]], y: np.ndarray) -> None:
    """Проверка разбиения.

    Проверяет: есть все части (train, val, test) и они не пусты;
    индексы лежат в диапазоне [0, len(y)); части не пересекаются и
    в сумме дают все len(y) объектов; в каждой части есть все классы.

    params:
        split - словарь с индексами train, val и test
        y - массив меток

    raise:
        ValueError, при отсутствии частей, пустых частях, выходе индексов
        за диапазон, пересечении частей, неверной сумме размеров
        и отсутствии классов в какой-либо части
    """
    parts = ("train", "val", "test")
    for p in parts:
        if p not in split:
            raise ValueError(f"Пропущена часть: {p}")
        if not len(split[p]):
            raise ValueError(f"Пустая часть: {p}")

    all_idx = np.concatenate(
        [np.asarray(split[p], dtype=np.int64) for p in parts]
    )

    if all_idx.min() < 0 or all_idx.max() >= len(y):
        raise ValueError(
            f"Индексы вне диапазона [0, {len(y)}): "
            f"min={all_idx.min()}, max={all_idx.max()}"
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
