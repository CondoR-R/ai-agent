"""Работа с файлом <run_id>/metrics.json"""
from pathlib import Path
import json


def write(path: Path, metrics: dict[str, float]) -> None:
    """Запись в файл metrics.json.

    params:
        path    - путь к файлу (например, runs/<run_id>/metrics.json)
        metrics - словарь с метриками (ключи — имена метрик,
                  значения — строки/числа, сериализуемые в JSON)
    """
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def get_val_loss(records: list[dict]) -> float:
    """Вычисление итоговых потерь на val
    
    params:
        records - список записей из history.csv

    return: итоговые потери
    """
    return records[-1]['val_loss']


def get_train_time(records: list[dict]) -> float:
    """Вычисление времени обучения

    params:
        records - список записей из history.csv

    return: время обучения
    """
    train_time = 0.0
    for record in records:
        train_time += record["time_epoch"]
    return train_time


def get_acc(records: list[dict], split: str) -> float:
    """Получение итогового accuracy из predictions.csv

    params:
        records - список записей из predictions.csv
        split - test или val выборка

    return: accuracy для выборки
    """
    correct = 0
    samples = 0
    for record in records:
        if record['split'] != split:
            continue

        y_true = record['y_true']
        y_pred = record['y_pred']

        if y_true == y_pred:
            correct += 1

        samples += 1

    if samples == 0:
        samples = 1

    return correct / samples
