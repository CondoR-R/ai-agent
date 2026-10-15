"""Фиксация параметров для воспроизводимости"""
import random

import numpy as np
import torch


def set_seed(seed: int = 0) -> None:
    """Фиксирует seed для встроенного модуля random, библиотек
    numpy и pytorch
    
    params:
        seed - сид генерации
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_num_threads(num_threads: int = 1) -> None:
    """Устанавливает число потоков в pytorch
    
    params:
        num_threads - число потоков
    """
    torch.set_num_threads(num_threads)


def enable_deterministic_mode() -> None:
    """Включает выбор только операций детерминированных алгоритмов"""
    torch.use_deterministic_algorithms(True)
