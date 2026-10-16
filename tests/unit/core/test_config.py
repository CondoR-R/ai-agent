"""Тесты для src/ai_agent/core/config.py."""

from pathlib import Path

import pytest
import yaml

from ai_agent.core.config import (
    load_config,
    merge_configs,
    save_config,
    validate_config,
)


def _write_yaml(tmp_path: Path, name: str, content: str) -> Path:
    """Помощник: пишет YAML-строку в файл внутри tmp_path."""
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture
def valid_config() -> dict:
    """Минимальный корректный конфиг для тестов."""
    return {
        "experiment": "lr0",
        "split_seed": 0,
        "seed": 0,
        "model": {"n_layers": 1, "hidden": 64, "dropout": 0.0},
        "train": {"lr": 0.001, "weight_decay": 0.0, "batch_size": 64, "epochs": 30},
        "runtime": {"device": "cpu", "num_threads": 1},
    }


# --- load_config ---


def test_load_valid_yaml(tmp_path: Path):
    """Получаем словарь с ожидаемыми ключами"""
    p = _write_yaml(
        tmp_path,
        "cfg.yaml",
        "experiment: lr0\nseed: 42\nmodel:\n  hidden: 64\n",
    )
    cfg = load_config(p)
    assert cfg["experiment"] == "lr0"
    assert cfg["seed"] == 42
    assert cfg["model"]["hidden"] == 64


def test_load_empty_file_returns_empty_dict(tmp_path: Path):
    """При пустом файле возвращает {}"""
    p = _write_yaml(tmp_path, "empty.yaml", "")
    assert load_config(p) == {}


def test_load_missing_file_raises(tmp_path: Path):
    """Файл не найден"""
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_load_invalid_yaml_raises(tmp_path: Path):
    """Битый YAML"""
    p = _write_yaml(tmp_path, "bad.yaml", "key: [unclosed\n")
    with pytest.raises(yaml.YAMLError):
        load_config(p)


# --- merge_configs ---


def test_merge_overrides_top_level(valid_config):
    """Значения из cli переопределяют исходный конфиг"""
    merged = merge_configs(valid_config, {"seed": 42})
    assert merged["seed"] == 42


def test_merge_preserves_other_fields(valid_config):
    """После объединения нетронутые поля не меняются"""
    merged = merge_configs(valid_config, {"seed": 42})
    assert merged["experiment"] == valid_config["experiment"]
    assert merged["train"] == valid_config["train"]


def test_merge_does_not_mutate_source(valid_config):
    """Исходный словарь не мутирует"""
    original_seed = valid_config["seed"]
    merged = merge_configs(valid_config, {"seed": 42})
    new_lr = 999.0
    merged["train"]["lr"] = new_lr
    # После мутации результата исходный словарь должен остаться нетронутым
    assert valid_config["seed"] == original_seed
    assert valid_config["train"]["lr"] != new_lr


def test_merge_returns_new_object(valid_config):
    """Возвращаемый словарь - новый объект"""
    merged = merge_configs(valid_config, {})
    assert merged is not valid_config


def test_merge_nested_dict_is_deepcopy(valid_config):
    """Проверка глубокого копирования:
    вложенные словари не общие с исходным
    """
    merged = merge_configs(valid_config, {})
    assert merged["train"] is not valid_config["train"]
    assert merged["model"] is not valid_config["model"]


# --- validate_config ---


def test_validate_ok(valid_config):
    validate_config(valid_config)  # не должно бросить


def test_validate_missing_required_field(valid_config):
    """Отсутствие обязательного поля"""
    del valid_config["model"]
    with pytest.raises(ValueError, match="model"):
        validate_config(valid_config)


def test_validate_lr_negative(valid_config):
    """Отрицательный lr"""
    valid_config["train"]["lr"] = -0.001
    with pytest.raises(ValueError, match="lr"):
        validate_config(valid_config)


def test_validate_lr_zero(valid_config):
    """Нулевой lr"""
    valid_config["train"]["lr"] = 0.0
    with pytest.raises(ValueError, match="lr"):
        validate_config(valid_config)


def test_validate_missing_lr(valid_config):
    """lr отсутствует"""
    del valid_config["train"]["lr"]
    with pytest.raises(ValueError, match="lr"):
        validate_config(valid_config)


# --- save_config ---


def test_save_and_load_roundtrip(tmp_path: Path, valid_config):
    """При чтении и записи в питон попадает словарь"""
    p = tmp_path / "out.yaml"
    save_config(p, valid_config)
    loaded = load_config(p)
    assert loaded == valid_config


def test_save_preserves_key_order(tmp_path: Path):
    """Порядок словаря не меняется после сохранения"""
    cfg = {"zebra": 1, "apple": 2, "mango": 3}
    p = tmp_path / "out.yaml"
    save_config(p, cfg)
    text = p.read_text(encoding="utf-8")
    # Порядок в файле должен совпадать с порядком в словаре
    assert text.index("zebra") < text.index("apple") < text.index("mango")


def test_save_keeps_unicode(tmp_path: Path):
    """Корректная обработка utf-8"""
    cfg = {"note": "привет"}
    p = tmp_path / "out.yaml"
    save_config(p, cfg)
    text = p.read_text(encoding="utf-8")
    assert "привет" in text
    assert "\\u" not in text


def test_save_uses_block_style(tmp_path: Path):
    """Блочный стиль: вложенный словарь пишется через отступы,
    а не как {hidden: 64, n_layers: 1}
    """
    cfg = {"model": {"hidden": 64, "n_layers": 1}}
    p = tmp_path / "out.yaml"
    save_config(p, cfg)
    text = p.read_text(encoding="utf-8")

    assert "{" not in text
    assert "hidden: 64" in text
