import copy
import csv
import math

import numpy as np
import pytest
import torch

from ai_agent.labs.lr0.data import DigitsData
from ai_agent.labs.lr0.train import (
    HISTORY_FILENAME,
    PREDICTIONS_FILENAME,
    Trainer,
    accuracy,
    compute_metrics,
    prepare_parts,
    run_training,
)


# ---------- helpers ----------


def _make_cfg(epochs: int = 2) -> dict:
    """Минимальный конфиг для Trainer / run_training."""
    return {
        "model": {"n_layers": 1, "hidden": 16, "dropout": 0.1},
        "train": {
            "lr": 1e-3,
            "epochs": epochs,
            "batch_size": 32,
            "weight_decay": 0.0,
        },
        "runtime": {"device": "cpu", "num_threads": 1},
        "seed": 0,
    }


def _read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_history_without_time(path):
    """history.csv как список словарей без колонки epoch_time."""
    return [
        {k: v for k, v in row.items() if k != "epoch_time"}
        for row in _read_csv_rows(path)
    ]


# ---------- фикстуры ----------


@pytest.fixture(scope="module")
def tiny_data():
    """Быстрый синтетический набор для тестов Trainer (без I/O)."""
    rng = np.random.default_rng(0)
    n = 120
    X = rng.random((n, 64), dtype=np.float32)
    y = rng.integers(0, 10, n, dtype=np.int64)
    idx = np.arange(n)
    rng.shuffle(idx)
    split = {
        "train": idx[:60].tolist(),
        "val": idx[60:90].tolist(),
        "test": idx[90:].tolist(),
    }
    return DigitsData(X=X, y=y, split=split, split_sha256="0" * 64)


@pytest.fixture(scope="module")
def real_data():
    """Настоящий load_digits с фиксированным разбиением 1077/360/360.

    Разбиение строится здесь напрямую (не через get_split), чтобы
    тесты не касались data/split.json и data/manifest.json.
    """
    from ai_agent.labs.lr0.data import load_raw

    X, y = load_raw()
    idx = np.arange(len(y))
    np.random.default_rng(0).shuffle(idx)
    split = {
        "train": idx[:1077].tolist(),
        "val": idx[1077:1437].tolist(),
        "test": idx[1437:].tolist(),
    }
    return DigitsData(X=X, y=y, split=split, split_sha256="0" * 64)


# ---------- accuracy ----------


def test_accuracy_basic():
    assert accuracy([1, 2, 3, 3], [1, 2, 3, 4]) == 0.75


def test_accuracy_length_mismatch():
    with pytest.raises(ValueError, match="Размеры"):
        accuracy([1, 2], [1])


@pytest.mark.parametrize(
    "y_true,y_pred", [([], []), (np.array([]), np.array([]))]
)
def test_accuracy_empty(y_true, y_pred):
    with pytest.raises(ValueError, match="пуст"):
        accuracy(y_true, y_pred)


# ---------- Trainer: повторяемость ----------


def test_trainer_same_seed_same_results(tiny_data):
    """Два Trainer с seed=0 дают равные EpochResult по всем полям кроме epoch_time."""
    cfg = _make_cfg(epochs=2)
    parts = prepare_parts(tiny_data, torch.device("cpu"))

    # Конструируем каждый Trainer отдельно, чтобы __init__ сбросил seed
    r1 = list(Trainer(cfg, parts, seed=0).epochs())
    r2 = list(Trainer(cfg, parts, seed=0).epochs())

    assert len(r1) == len(r2) == 2
    for a, b in zip(r1, r2):
        assert a.epoch == b.epoch
        assert a.train_loss == b.train_loss
        assert a.val_loss == b.val_loss
        assert a.val_acc == b.val_acc


def test_trainer_different_seed_differs(tiny_data):
    """Другой seed даёт другие значения метрик (все поля кроме epoch_time)."""
    cfg = _make_cfg(epochs=2)
    parts = prepare_parts(tiny_data, torch.device("cpu"))

    r0 = list(Trainer(cfg, parts, seed=0).epochs())
    r1 = list(Trainer(cfg, parts, seed=1).epochs())

    assert any(
        a.train_loss != b.train_loss or a.val_loss != b.val_loss
        for a, b in zip(r0, r1)
    )


# ---------- run_training: воспроизводимость ----------


def test_run_training_reproducible_files(tmp_path, tiny_data):
    """Два запуска с seed=0 дают одинаковые predictions.csv и history.csv
    (без колонки epoch_time)."""
    cfg = _make_cfg(epochs=2)

    dir1 = tmp_path / "run1"
    dir1.mkdir()
    dir2 = tmp_path / "run2"
    dir2.mkdir()

    run_training(dir1, copy.deepcopy(cfg), tiny_data, eval_test=False)
    run_training(dir2, copy.deepcopy(cfg), tiny_data, eval_test=False)

    # predictions.csv — побайтово равны
    p1 = (dir1 / PREDICTIONS_FILENAME).read_bytes()
    p2 = (dir2 / PREDICTIONS_FILENAME).read_bytes()
    assert p1 == p2

    # history.csv — совпадает без колонки epoch_time
    h1 = _read_history_without_time(dir1 / HISTORY_FILENAME)
    h2 = _read_history_without_time(dir2 / HISTORY_FILENAME)
    assert h1 == h2


# ---------- eval_test ----------


def test_no_test_rows_without_eval_test(tmp_path, tiny_data):
    run_training(tmp_path, _make_cfg(epochs=1), tiny_data, eval_test=False)
    rows = _read_csv_rows(tmp_path / PREDICTIONS_FILENAME)
    assert {r["split"] for r in rows} == {"val"}


def test_test_rows_with_eval_test(tmp_path, real_data):
    """С eval_test=True в predictions.csv ровно 360 строк test."""
    run_training(tmp_path, _make_cfg(epochs=1), real_data, eval_test=True)
    rows = _read_csv_rows(tmp_path / PREDICTIONS_FILENAME)
    test_rows = [r for r in rows if r["split"] == "test"]
    assert len(test_rows) == 360


# ---------- compute_metrics ----------


def test_compute_metrics_detects_tampered_prediction(tmp_path, real_data):
    """Подмена одного предсказания в predictions.csv → ValueError про val_acc."""
    run_training(tmp_path, _make_cfg(epochs=2), real_data, eval_test=False)

    path = tmp_path / PREDICTIONS_FILENAME
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # меняем предсказание у заведомо корректной val-строки
    target = next(
        r for r in rows if r["split"] == "val" and r["y_true"] == r["y_pred"]
    )
    target["y_pred"] = str((int(target["y_pred"]) + 1) % 10)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="val_acc"):
        compute_metrics(tmp_path)


# ---------- history.csv ----------


def test_history_row_count(tmp_path, tiny_data):
    """history.csv: заголовок + ровно epochs строк."""
    epochs = 3
    run_training(tmp_path, _make_cfg(epochs=epochs), tiny_data, eval_test=False)

    with open(tmp_path / HISTORY_FILENAME, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == epochs
