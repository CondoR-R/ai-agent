"""Цикл обучения MLP для ЛР0: эпохи, метрики по эпохам, предсказания.

Обучает MLP на load_digits с CrossEntropyLoss
и Adam. Возвращает структуры (history_rows, predictions).
"""

import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ai_agent.models.mlp import MLP


def train_epoch(
    model: MLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Один проход по train с обновлением весов.

    params:
        model - обучаемая модель
        loader - DataLoader с train-данными, shuffle=True
        optimizer - оптимизатор (Adam)
        criterion - функция потерь (CrossEntropyLoss)
        device - устройство (cpu или cuda)

    return: средний train loss за эпоху, посчитанный по всем примерам
    """
    model.train()
    total_loss = 0
    total_samples = 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        total_samples += len(y_batch)
    return total_loss / total_samples


def evaluate(
    model: MLP,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int]]:
    """Один проход по val или test без обновления весов.

    params:
        model - обученная модель
        loader - DataLoader с val/test-данными, shuffle=False
        criterion - функция потерь (CrossEntropyLoss)
        device - устройство (cpu или cuda)

    return: кортеж (loss, accuracy, predictions)
        loss - средний loss по всем примерам
        accuracy - доля верных предсказаний
        predictions - предсказанные классы в порядке следования loader
    """
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_samples = 0
        correct = 0
        preds: list[int] = []
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            batch_preds = logits.argmax(dim=1)

            total_loss += loss.item() * len(X_batch)
            total_samples += len(X_batch)
            correct += (batch_preds == y_batch).sum().item()
            preds.extend(batch_preds.cpu().tolist())
    return (total_loss / total_samples, correct / total_samples, preds)


def train_epochs(
    model: MLP,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    val_indices: list[int],
    train_cfg: dict,
    runtime_cfg: dict,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Обучает модель заданное число эпох и собирает метрики.

    params:
        model - экземпляр MLP
        X_train, y_train - обучающие данные (float32 / int64)
        X_val, y_val - валидационные данные (float32 / int64)
        val_indices - индексы строк X_val в исходном датасете
            (обычно split["val"] из data/split.json)
        train_cfg - секция train из конфига
            (lr, weight_decay, batch_size, epochs)
        runtime_cfg - секция runtime из конфига
            (device, num_threads)
        seed - сид для генератора DataLoader

    return: кортеж (history_rows, predictions)
        history_rows - по строке на эпоху:
            epoch, train_loss, val_loss, val_acc, epoch_time_sec
        predictions - по строке на пример val:
            index, y_true, y_pred (за последнюю эпоху)
    """
    if len(val_indices) != len(X_val):
        raise ValueError(
            f"val_indices длины {len(val_indices)} не совпадает с X_val длины {len(X_val)}"
        )

    device = torch.device(runtime_cfg["device"])
    model.to(device)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    history_rows: list[dict] = []
    predictions: list[dict] = []
    for epoch in range(1, train_cfg["epochs"] + 1):
        t0 = time.perf_counter()
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, val_preds = evaluate(
            model, val_loader, criterion, device
        )
        t1 = time.perf_counter()
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch_time_sec": t1 - t0,
            }
        )
        predictions = [
            {
                "index": val_indices[i],
                "y_true": int(y_val[i]),
                "y_pred": val_preds[i],
            }
            for i in range(len(val_preds))
        ]
    return history_rows, predictions
