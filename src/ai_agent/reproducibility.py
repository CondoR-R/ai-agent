"""Утилиты для воспроизводимости экспериментов"""

import os
import random

import numpy as np


def enable_determinism(num_threads: int = 1) -> None:
    """Включает детерминированные алгоритмы и фиксирует число потоков.

    torch.use_deterministic_algorithms(True) заставляет PyTorch использовать
    только детерминированные реализации операций. Если операция не имеет
    такой реализации, будет выброшено исключение (а не тихая недетермини-
    рованность). Это полезно для отладки, но может замедлить обучение.

    torch.set_num_threads(1) убирает недетерминизм, связанный с
    параллельными редукциями в CPU-операциях (например, в matmul).
    """
    import torch

    # Переменная окружения для cuBLAS: нужна для детерминизма на GPU
    # (задать ДО инициализации CUDA; безопасно — просто no-op, если CUDA нет).
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_num_threads(num_threads)


def set_seed(seed: int, num_threads: int = 1) -> None:
    """Фиксирует seed для random, NumPy и PyTorch и включает детерминизм.

    Если PyTorch не установлен (например, клиент к серверу модели в ЛР1+),
    фиксируются только random и NumPy.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    enable_determinism(num_threads=num_threads)


def make_generator(seed: int):
    """Создаёт torch.Generator с заданным seed — для DataLoader."""
    import torch

    g = torch.Generator()
    g.manual_seed(seed)
    return g
