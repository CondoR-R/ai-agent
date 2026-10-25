"""Обучение модели"""

import torch
from torch.utils.data import DataLoader
from torch import nn

from .model import MLP


def train(model: MLP, train_cfg: dict, loader: DataLoader):
    """Обучение модели
    
    params:
        model - обучаемая модель
        train_cfg - поле train из конфигурации
        loader - даталоадер
    """
    optimizer = torch.optim.Adam(model.parameters(), train_cfg['lr'], weight_decay=train_cfg['weight_decay'])
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(1, train_cfg['epochs'] + 1):
        model.train()
        for x, y in loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
