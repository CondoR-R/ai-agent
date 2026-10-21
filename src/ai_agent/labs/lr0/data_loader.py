"""Создание DataLoader с учетом фиксированного генератора
случайных чисел для воспроизводимости
"""

import torch
from torch.utils.data import DataLoader, Dataset


def create_dataloader(dataset: Dataset, bs: int, seed: int) -> DataLoader:
    """Создает DataLoader с фиксированным генератором случайных чисел

    params:
        dataset - объект Dataset
        bs - размер батча
        seed - сид генерации

    return: DataLoader
    """
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=bs, shuffle=True, generator=g, num_workers=0
    )
    return loader
