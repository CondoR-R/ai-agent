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


def validate_split(split: dict[str, list[int]], y: np.ndarray) -> None:
    """Проверка сплита:
    есть все части (train, val, test),
    части не пересекаются, в сумме N элементов, в каждой есть все классы

    params:
        split - словарь с train, val и test разбиениями,
        y - массив меток

    raise:
        ValueError, при отсутсвии обязательных полей или пустые, некорректные значения
    """
    parts = ("train", "val", "test")
    for p in parts:
        # проверка на наличие train val test
        if p not in split:
            raise ValueError(f"Пропущено поле: {p}")
        # проверка на пустую часть
        if not len(split[p]):
            raise ValueError(f"Пустое поле: {p}")

    # проверка на пересечение
    all_idx = np.concatenate(
        [np.asarray(split[p], dtype=np.int64) for p in parts]
    )
    if len(all_idx) != len(np.unique(all_idx)):
        raise ValueError("Значения присутвуют в разных частях, утечка данных")

    # провека на правильную длину
    if len(all_idx) != len(y):
        raise ValueError(
            f"Есть {len(all_idx)} объектов в выборках, должно быть: {len(y)}"
        )

    # проверка на наличие всех классов в каждой части
    set_classes = set(np.unique(y))
    for p in parts:
        set_part = set(y[split[p]])
        if set_classes != set_part:
            raise ValueError(
                f"выборка {p} должна содержать все классы, пропущены: {set_classes - set_part}"
            )
