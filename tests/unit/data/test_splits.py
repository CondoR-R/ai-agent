"""Тесты для src/ai_agent/data/splits.py."""

from copy import deepcopy

import numpy as np
import pytest

from ai_agent.data.loaders import load_digits_dataset
from ai_agent.data.splits import make_stratified_split, validate_split


@pytest.fixture
def digits():
    """Загруженные данные load_digits — один раз на весь модуль."""
    return load_digits_dataset()


# --- make_stratified_split ---


def test_split_sizes(digits):
    """Правильные размеры train, val, test после сплита"""
    _, y = digits
    split = make_stratified_split(y)
    assert len(split["train"]) == 1077
    assert len(split["val"]) == 360
    assert len(split["test"]) == 360


def test_split_covers_all(digits):
    """Длина датасета не меняется после сплита"""
    _, y = digits
    split = make_stratified_split(y)
    total = len(split["train"]) + len(split["val"]) + len(split["test"])
    assert total == len(y)


def test_split_no_overlap(digits):
    """Отсутвуют пересечения"""
    _, y = digits
    split = make_stratified_split(y)
    train = set(split["train"])
    val = set(split["val"])
    test = set(split["test"])
    assert train & val == set()
    assert train & test == set()
    assert val & test == set()


def test_split_reproducible(digits):
    """При одинаковом random_state сплиты будут одинаковые"""
    _, y = digits
    s1 = make_stratified_split(y, random_state=0)
    s2 = make_stratified_split(y, random_state=0)
    assert s1 == s2


def test_split_different_random_state(digits):
    """Разные random_state дадут разное разделение"""
    _, y = digits
    s1 = make_stratified_split(y, random_state=0)
    s2 = make_stratified_split(y, random_state=1)
    # хотя бы одна часть должна отличаться
    assert s1 != s2


# --- validate_split: положительный сценарий ---


def test_validate_ok(digits):
    _, y = digits
    split = make_stratified_split(y)
    validate_split(split, y)  # не должно бросить


# --- validate_split: негативные сценарии ---


def test_validate_missing_part(digits):
    """Отсутвует одна из частей"""
    _, y = digits
    split = make_stratified_split(y)
    del split["val"]
    with pytest.raises(ValueError, match="val"):
        validate_split(split, y)


def test_validate_empty_part(digits):
    """Есть пустая часть"""
    _, y = digits
    split = make_stratified_split(y)
    split["val"] = []
    with pytest.raises(ValueError, match="val"):
        validate_split(split, y)


def test_validate_overlap(digits):
    """Есть пересечения"""
    _, y = digits
    split = make_stratified_split(y)
    # перенесём первый индекс train в val, не убирая из train
    split["val"].append(split["train"][0])
    with pytest.raises(ValueError):
        validate_split(split, y)


def test_validate_wrong_total(digits):
    """Длина после сплита изменилась"""
    _, y = digits
    split = make_stratified_split(y)
    # удалим один индекс из train, сумма станет N-1
    split["train"].pop()
    with pytest.raises(ValueError):
        validate_split(split, y)


def test_validate_missing_class(digits):
    """Сплит не страцифицирован"""
    _, y = digits
    split = make_stratified_split(y)
    # заберём из test все индексы с меткой 9 и отдадим их в train
    test_idx = split["test"]
    keep = [i for i in test_idx if y[i] != 9]
    moved = [i for i in test_idx if y[i] == 9]
    split["test"] = keep
    split["train"].extend(moved)
    with pytest.raises(ValueError, match="test"):
        validate_split(split, y)
