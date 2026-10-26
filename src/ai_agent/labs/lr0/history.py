"""Работа с файлом <run_id>/history.csv"""

import csv
from pathlib import Path


def update(path: Path, record: dict) -> None:
    """Добавление записи в history.csv
    Если первый возов - создаст файл

    params:
        path - путь к файлу
        record - записываемая строка
            (словарь вида: {epoch, train_loss, val_loss, val_acc, epoch_time})
    """
    is_new = not path.exists() or path.stat().st_size == 0

    if is_new:
        header = list(record.keys())
    else:
        with path.open("r", newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), list(record.keys()))

    mode = "w" if is_new else "a"
    with path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow(record)
