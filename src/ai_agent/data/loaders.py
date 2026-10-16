"""Загрузчик наборов данных"""

import numpy as np
from sklearn.datasets import load_digits


def load_digits_dataset() -> tuple[np.ndarray, np.ndarray]:
    """load_digits: 1797 изображений 8x8, 64 признака, метки 0..9.

    Признаки приводятся к [0, 1] делением на 16.
    Масштаб известен заранее, статистики по данным не вычисляются.
    """
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float32) / 16.0
    y = np.asarray(y, dtype=np.int64)
    return X, y
