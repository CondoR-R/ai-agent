"""Обучение MLP на load_digits: цикл обучения, оценка, history и predictions."""

import csv
import math
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from ai_agent.core import metrics as core_metrics
from ai_agent.core import reproducibility

from .data import DigitsData
from .model import MLP

HISTORY_FILENAME = "history.csv"
PREDICTIONS_FILENAME = "predictions.csv"

HISTORY_FIELDS = ("epoch", "train_loss", "val_loss", "val_acc", "epoch_time")
PREDICTIONS_FIELDS = ("split", "index", "y_true", "y_pred")

Parts = dict[str, tuple[Tensor, Tensor]]


@dataclass(frozen=True)
class EpochResult:
    """Результат одной эпохи"""

    epoch: int
    train_loss: float
    val_loss: float
    val_acc: float
    epoch_time: float


# метрики


def accuracy(
    y_true: Sequence[int] | np.ndarray, y_pred: Sequence[int] | np.ndarray
) -> float:
    """Доля верных предсказаний

    params:
        y_true - верные ответы
        y_pred - предсказания

    return: точность в [0, 1]

    raise:
        ValueError, если длины не совпадают или массивы пусты
    """
    a = np.asarray(y_true)
    b = np.asarray(y_pred)
    if a.shape != b.shape:
        raise ValueError(f"Размеры не совпадают: {a.shape} и {b.shape}")
    if a.size == 0:
        raise ValueError("Нельзя посчитать точность на пустом наборе")
    return float(np.mean(a == b))


# данные


def prepare_parts(data: DigitsData, device: torch.device) -> Parts:
    """Переводит части данных в тензоры на нужном устройстве

    params:
        data - данные ЛР0 с разбиением
        device - устройство для тензоров

    return: словарь {часть: (X float32, y long)}
    """
    parts: Parts = {}
    for name in data.split:
        X, y = data.part(name)
        parts[name] = (
            torch.from_numpy(X).float().to(device),
            torch.from_numpy(y).long().to(device),
        )
    return parts


# обучение


def evaluate(
    model: nn.Module, X: Tensor, y: Tensor, loss_fn: nn.Module
) -> tuple[float, Tensor]:
    """Оценка модели на одной части данных (режим eval, без градиентов)

    params:
        model - модель
        X, y - признаки и метки
        loss_fn - функция потерь

    return: кортеж (потери, предсказанные классы)
    """
    model.eval()
    with torch.no_grad():
        logits = model(X)
        loss = loss_fn(logits, y).item()
        pred = logits.argmax(dim=1)
    return loss, pred


class Trainer:
    """Обучение по конфигурации: фиксирует seed, создаёт модель и
    оптимизатор, отдаёт результаты по эпохам.

    Порядок важен: seed фиксируется до создания модели, иначе начальные
    веса не зависят от seed.

    Для Optuna:
        for r in Trainer(cfg, parts, seed).epochs():
            trial.report(r.val_acc, step=r.epoch)
    """

    def __init__(self, cfg: dict, parts: Parts, seed: int) -> None:
        """
        params:
            cfg - итоговая конфигурация (секции model, train, runtime)
            parts - тензоры частей (prepare_parts)
            seed - seed обучения
        """
        runtime = cfg["runtime"]
        self.device = torch.device(runtime["device"])
        self.train_cfg = cfg["train"]
        self.parts = parts

        reproducibility.set_seed(seed)
        reproducibility.set_num_threads(runtime["num_threads"])
        reproducibility.enable_deterministic_mode()

        m = cfg["model"]
        self.model = MLP(
            n_layers=m["n_layers"], hidden=m["hidden"], dropout=m["dropout"]
        ).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.train_cfg["lr"],
            weight_decay=self.train_cfg["weight_decay"],
        )
        # отдельный генератор порядка мини-батчей с тем же seed
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def _train_one_epoch(self) -> float:
        """Одна эпоха по train

        return: средние потери по эпохе, взвешенные размером батча
        """
        X, y = self.parts["train"]
        n = len(y)
        batch_size = self.train_cfg["batch_size"]

        self.model.train()
        order = torch.randperm(n, generator=self.generator)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size].to(self.device)
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.model(X[idx]), y[idx])
            loss.backward()
            self.optimizer.step()
            total += loss.item() * len(idx)
        return total / n

    def epochs(self) -> Iterator[EpochResult]:
        """Генератор результатов по эпохам.

        Время эпохи: обучение плюс оценка на val; загрузка данных
        и запись файлов в интервал не входят.
        """
        X_val, y_val = self.parts["val"]
        for epoch in range(1, self.train_cfg["epochs"] + 1):
            t0 = time.perf_counter()

            train_loss = self._train_one_epoch()
            val_loss, val_pred = evaluate(
                self.model, X_val, y_val, self.loss_fn
            )
            val_acc = accuracy(y_val.cpu().numpy(), val_pred.cpu().numpy())

            epoch_time = time.perf_counter() - t0

            yield EpochResult(epoch, train_loss, val_loss, val_acc, epoch_time)


# файлы


def append_history(path: Path, result: EpochResult) -> None:
    """Дописывает строку эпохи в history.csv (заголовок при создании)

    params:
        path - путь к history.csv
        result - результат эпохи
    """
    is_new = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(
            {
                "epoch": result.epoch,
                "train_loss": result.train_loss,
                "val_loss": result.val_loss,
                "val_acc": result.val_acc,
                "epoch_time": result.epoch_time,
            }
        )
        f.flush()


def prediction_rows(
    split: str, indices: Sequence[int], y_true: Tensor, y_pred: Tensor
) -> list[dict]:
    """Строки для predictions.csv

    params:
        split - имя части (val или test)
        indices - глобальные индексы изображений в load_digits
        y_true - верные ответы
        y_pred - предсказания
    """
    t = y_true.cpu().tolist()
    p = y_pred.cpu().tolist()
    return [
        {"split": split, "index": i, "y_true": a, "y_pred": b}
        for i, a, b in zip(indices, t, p, strict=True)
    ]


def write_predictions(path: Path, rows: list[dict]) -> None:
    """Записывает predictions.csv целиком

    params:
        path - путь к predictions.csv
        rows - строки с полями PREDICTIONS_FIELDS
    """
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTIONS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def compute_metrics(run_dir: Path) -> dict:
    """Считает итоговые метрики из history.csv и predictions.csv

    Точность пересчитывается из predictions.csv и сверяется
    с val_acc последней эпохи в history.csv.

    params:
        run_dir - каталог запуска

    return: {val_acc, val_loss, train_time[, test_acc]}

    raise:
        ValueError, если history пуст, в predictions нет val
        или точности расходятся
    """
    history = _read_csv(run_dir / HISTORY_FILENAME)
    if not history:
        raise ValueError("history.csv пуст")
    predictions = _read_csv(run_dir / PREDICTIONS_FILENAME)

    def split_acc(name: str) -> float | None:
        rows = [r for r in predictions if r["split"] == name]
        if not rows:
            return None
        return accuracy(
            [int(r["y_true"]) for r in rows], [int(r["y_pred"]) for r in rows]
        )

    val_acc = split_acc("val")
    if val_acc is None:
        raise ValueError("В predictions.csv нет строк val")

    last = history[-1]
    if not math.isclose(val_acc, float(last["val_acc"]), abs_tol=1e-9):
        raise ValueError(
            f"val_acc из predictions ({val_acc}) не совпадает "
            f"с val_acc последней эпохи ({last['val_acc']})"
        )

    metrics = {
        "val_acc": val_acc,
        "val_loss": float(last["val_loss"]),
        "train_time": sum(float(r["epoch_time"]) for r in history),
    }
    test_acc = split_acc("test")
    if test_acc is not None:
        metrics["test_acc"] = test_acc
    return metrics


# запуск


def run_training(
    run_dir: Path, cfg: dict, data: DigitsData, eval_test: bool = False
) -> dict:
    """Обучение с записью history.csv, predictions.csv и metrics.json

    Test оценивается один раз, после последней эпохи, и только
    при eval_test=True.

    params:
        run_dir - каталог запуска
        cfg - итоговая конфигурация
        data - данные ЛР0
        eval_test - оценить ли итоговую модель на test

    return: словарь метрик (тот же, что записан в metrics.json)
    """
    device = torch.device(cfg["runtime"]["device"])
    parts = prepare_parts(data, device)
    trainer = Trainer(cfg, parts, cfg["seed"])

    history_path = run_dir / HISTORY_FILENAME
    for result in trainer.epochs():
        append_history(history_path, result)

    rows: list[dict] = []
    names = ("val", "test") if eval_test else ("val",)
    for name in names:
        X, y = parts[name]
        _, pred = evaluate(trainer.model, X, y, trainer.loss_fn)
        rows += prediction_rows(name, data.split[name], y, pred)
    write_predictions(run_dir / PREDICTIONS_FILENAME, rows)

    metrics = compute_metrics(run_dir)
    core_metrics.write(run_dir, metrics)
    return metrics
