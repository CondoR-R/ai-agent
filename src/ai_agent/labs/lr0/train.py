"""Обучение модели"""

import time
from collections.abc import Generator

import torch
from torch import nn
from torch.utils.data import DataLoader

from .model import MLP


def accuracy(logits, labels) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()


def train(
    model: MLP,
    train_cfg: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
) -> Generator:
    """Обучение модели

    params:
        model - обучаемая модель
        train_cfg - поле train из конфигурации
        train_loader - тренировочная выборка
        val_loader - валидационная выборка
        device - устройство, на котором будут выполнены вычисления
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, train_cfg["epochs"] + 1):
        start = time.perf_counter()
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, device
        )
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        time_epoch = time.perf_counter() - start

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "time_epoch": time_epoch,
        }
        yield record


def train_epoch(
    model: MLP,
    loader: DataLoader,
    optimizer: torch.optim.Adam,
    loss_fn: nn.CrossEntropyLoss,
    device: str,
) -> float:
    """Одна эпоха обучения

    return: loss эпохи
    """
    model.train()
    train_loss = 0
    total_samples = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        train_loss += loss.item() * bs
        total_samples += bs

    return train_loss / total_samples


@torch.no_grad
def evaluate(
    model: MLP, loader: DataLoader, loss_fn: nn.CrossEntropyLoss, device
) -> tuple[float, float]:
    """Один прогон валидационной или тестовой выборки

    return: кортеж из val_loss, val_acc
    """
    model.eval()
    total_loss = 0
    total_acc = 0
    total_samples = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = loss_fn(logits, y)
        acc = accuracy(logits, y)

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_acc += acc * bs
        total_samples += bs

    return total_loss / total_samples, total_acc / total_samples
