"""Тесты для src/ai_agent/data/loaders.py."""

import numpy as np

from ai_agent.data.loaders import load_digits_dataset


def test_shapes():
    """Корректный размер"""
    X, y = load_digits_dataset()
    assert X.shape == (1797, 64)
    assert y.shape == (1797,)


def test_dtypes():
    """Проверка типов"""
    X, y = load_digits_dataset()
    assert X.dtype == np.float32
    assert y.dtype == np.int64


def test_X_range():
    """Значения в пределах [0, 1]"""
    X, _ = load_digits_dataset()
    assert X.min() >= 0.0
    assert X.max() <= 1.0


def test_X_max_is_exactly_one():
    """Максимальное значение равно 1.0"""
    # Оригинальные значения 0..16, деление на 16 даёт ровно 1.0
    # на максимуме. Если max != 1.0, деление сделано не на 16.
    X, _ = load_digits_dataset()
    assert X.max() == 1.0


def test_labels_are_all_digits():
    """10 классов от 0 до 9"""
    _, y = load_digits_dataset()
    assert set(y.tolist()) == set(range(10))


def test_reproducible():
    # Загрузка не должна зависеть от глобального состояния
    X1, y1 = load_digits_dataset()
    X2, y2 = load_digits_dataset()
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)
