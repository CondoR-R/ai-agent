import copy

import pytest

from ai_agent.core import config as core_config
from ai_agent.labs.lr0.config_checks import validate_lr0


# ---------- fixtures ----------


@pytest.fixture
def valid_config() -> core_config.ConfigDict:
    """
    Минимально валидный конфиг ЛР0.

    Подгоните под реальную схему из core_config.validate: если обязательные
    поля внутри model/runtime/experiment отличаются — дополните здесь.
    """
    return {
        "experiment": {"name": "lr0"},
        "split_seed": 42,
        "seed": 42,
        "model": {},  # при необходимости добавьте обязательные поля
        "train": {"lr": 0.001, "epochs": 10},
        "runtime": {},  # при необходимости добавьте обязательные поля
    }


# ---------- happy path ----------


def test_valid_config_passes(valid_config):
    """Корректный конфиг не должен бросать исключений."""
    validate_lr0(valid_config)  # если бросит — тест упадёт


# ---------- lr ----------


@pytest.mark.parametrize("bad_lr", [-1, 0, -0.001, -1e-9])
def test_negative_or_zero_lr_raises(valid_config, bad_lr):
    """lr <= 0 должен отвергаться."""
    cfg = copy.deepcopy(valid_config)
    cfg["train"]["lr"] = bad_lr

    with pytest.raises(ValueError, match=r"train\.lr"):
        validate_lr0(cfg)


# ---------- epochs ----------


@pytest.mark.parametrize("bad_epochs", [0, -1, -100])
def test_zero_or_negative_epochs_raises(valid_config, bad_epochs):
    """epochs < 1 должен отвергаться."""
    cfg = copy.deepcopy(valid_config)
    cfg["train"]["epochs"] = bad_epochs

    with pytest.raises(ValueError, match=r"train\.epochs"):
        validate_lr0(cfg)


# ---------- train как словарь ----------


def test_train_not_dict_raises(valid_config):
    """Если train не словарь — ValueError с явным сообщением."""
    cfg = copy.deepcopy(valid_config)
    cfg["train"] = ["lr", "epochs"]  # не dict

    with pytest.raises(ValueError, match="словар"):
        validate_lr0(cfg)


# ---------- отсутствующие поля ----------


@pytest.mark.parametrize("missing_key", ["lr", "epochs"])
def test_missing_train_field_raises(valid_config, missing_key):
    """Отсутствие train.lr или train.epochs → ValueError."""
    cfg = copy.deepcopy(valid_config)
    del cfg["train"][missing_key]

    with pytest.raises(ValueError, match=rf"train\.{missing_key}"):
        validate_lr0(cfg)


# ---------- обязательные поля верхнего уровня ----------


@pytest.mark.parametrize(
    "field",
    ["experiment", "split_seed", "seed", "model", "train", "runtime"],
)
def test_missing_required_field_raises(valid_config, field):
    """Пропуск любого обязательного поля ЛР0 отвергается core_config.validate."""
    cfg = copy.deepcopy(valid_config)
    del cfg[field]

    with pytest.raises(ValueError):
        validate_lr0(cfg)
