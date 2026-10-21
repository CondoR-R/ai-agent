"""Фиксация параметров для воспроизводимости"""

import os
import random

import numpy as np


def set_seed(seed: int = 0) -> None:
    """Фиксирует seed для random, numpy и pytorch

    params:
        seed - сид генерации
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_num_threads(num_threads: int = 1) -> None:
    """Устанавливает число потоков в pytorch

    params:
        num_threads - число потоков
    """
    import torch

    torch.set_num_threads(num_threads)


def enable_deterministic_mode() -> None:
    """Включает только детерминированные алгоритмы"""
    import torch

    # нужна для детерминизма cuBLAS на GPU, на CPU безвредна
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
