"""Работа с файлом <run_id>/predictions.csv"""

import torch
from torch.utils.data import DataLoader

from .model import MLP
from pathlib import Path

import csv


@torch.no_grad()
def get_preds(
    model: MLP, loader: DataLoader, idx: list[int], split: str, device: str
) -> list[dict]:
    """Получение списка записей для predictions.csv

    params:
        model - модель MLP
        loader - val/test выборка
        idx - список индекcов
        split - тип разбиения (val/test)
        device - устройство, на котором будут вычисления
    """
    model.eval()
    records: list[dict] = []
    offset = 0

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().tolist()
        y_list = y.tolist()

        for j in range(len(y_list)):
            k = offset + j
            records.append(
                {
                    "split": split,
                    "index": int(idx[k]),
                    "y_true": int(y_list[j]),
                    "y_pred": int(preds[j]),
                }
            )
        offset += len(y_list)

    return records




def write(path: Path, preds: list[dict]) -> None:
    """Записывает предсказания в predictions.csv.

    Файл создаётся заново (перезапись). Заголовок берётся из ключей
    первой записи. Пустой список → создаётся пустой файл (без заголовка,
    т.к. не из чего его вывести).

    params:
        path  - путь к файлу
        preds - список словарей вида {split, index, y_true, y_pred}
    """
    path = Path(path)

    with open(path, "w", newline="", encoding="utf-8") as f:
        if not preds:
            return

        writer = csv.DictWriter(f, fieldnames=list(preds[0].keys()))
        writer.writeheader()
        writer.writerows(preds)
