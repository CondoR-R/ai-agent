"""Эксперимент для lr0"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from ai_agent import files, reproducibility, tracking

from . import data, metrics, train
from .model import MLP, build_model


@dataclass
class Prepared:
    """Всё, что нужно для обучения одной конфигурации"""

    model: MLP
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    device: str


def load_inputs() -> tuple[np.ndarray, np.ndarray, dict]:
    """Данные и разбиение ЛР0

    return: кортеж (features, labels, split)
    """
    features, labels = data.load_data()
    split = files.read_json(files.DATA_DIR / "split.json")
    return features, labels, split


def prepare(cfg: dict, features, labels, split: dict) -> Prepared:
    """Загрузчики и модель для одной конфигурации.

    Порядок создания генератора, загрузчиков и модели общий для запуска
    и для trial Optuna: от него зависит повторяемость результата.
    Перед вызовом должен быть вызван reproducibility.set_seed.

    params:
        cfg - итоговая конфигурация
        features, labels - все данные
        split - индексы train, val, test

    return: объект Prepared с моделью и загрузчиками
    """
    train_features, train_labels = (
        features[split["train"]],
        labels[split["train"]],
    )
    val_features, val_labels = features[split["val"]], labels[split["val"]]
    test_features, test_labels = features[split["test"]], labels[split["test"]]

    device = cfg["runtime"]["device"]
    generator = reproducibility.make_generator(cfg["seed"])
    train_loader = data.get_loader(
        train_features,
        train_labels,
        cfg["train"]["batch_size"],
        generator,
        shuffle=True,
    )
    val_loader = data.get_loader(val_features, val_labels, 1, generator)
    test_loader = data.get_loader(test_features, test_labels, 1, generator)
    model = build_model(cfg["model"], device)
    return Prepared(model, train_loader, val_loader, test_loader, device)


def lr0_run(cfg: dict, run_dir: Path, eval_test: bool) -> None:
    """Один запуск lr0

    params:
        cfg - словарь конфигурации
        run_dir - директория запуска
        eval_test - флаг тестовых данных
    """
    # проверка lr >= 0
    if cfg["train"]["lr"] < 0:
        raise ValueError("lr должен быть положительным")
    # получение данных
    features, labels, split = load_inputs()
    ctx = prepare(cfg, features, labels, split)

    # обучение модели
    history_path = run_dir / "history.csv"
    for record in train.train(
        ctx.model, cfg["train"], ctx.train_loader, ctx.val_loader, ctx.device
    ):
        files.update_csv(history_path, record)
        tracking.log_epoch(record)

    # predictions
    preds = train.get_preds(
        ctx.model, ctx.val_loader, split["val"], "val", ctx.device
    )
    if eval_test:
        preds.extend(
            train.get_preds(
                ctx.model, ctx.test_loader, split["test"], "test", ctx.device
            )
        )
    predictions_path = run_dir / "predictions.csv"
    files.write_csv(predictions_path, preds)

    # metrics
    h = files.read_csv(
        history_path,
        cast={
            "val_loss": float,
            "time_epoch": float,
        },
    )
    val_loss = metrics.get_val_loss(h)
    train_time = metrics.get_train_time(h)
    p = files.read_csv(
        predictions_path,
        cast={
            "y_true": int,
            "y_pred": int,
        },
    )
    val_acc = metrics.get_acc(p, "val")

    metrics_dict = {
        "val_acc": val_acc,
        "val_loss": val_loss,
        "train_time": train_time,
    }
    if eval_test:
        metrics_dict["test_acc"] = metrics.get_acc(p, "test")

    files.write_json(run_dir / files.METRICS_FILENAME, metrics_dict)
