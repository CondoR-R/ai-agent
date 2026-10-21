import numpy as np
import pytest
from sklearn.datasets import load_digits

from ai_agent.labs.lr0.split import make_stratified_split, validate


@pytest.fixture(scope="module")
def digits():
    """Загружаем load_digits один раз на весь модуль."""
    X, y = load_digits(return_X_y=True)
    return X, y


# ---------- базовые свойства ----------


def test_digits_split_sizes_and_valid(digits):
    """На load_digits сплит даёт 1077 / 360 / 360 и проходит validate."""
    _, y = digits

    split = make_stratified_split(y)

    assert len(split["train"]) == 1077
    assert len(split["val"]) == 360
    assert len(split["test"]) == 360

    # не должно бросить
    validate(split, y)


def test_same_random_state_gives_identical_split(digits):
    """Один и тот же random_state даёт идентичный сплит."""
    _, y = digits

    a = make_stratified_split(y, random_state=42)
    b = make_stratified_split(y, random_state=42)

    assert a == b


def test_different_random_state_gives_different_split(digits):
    """Другой random_state даёт другой сплит."""
    _, y = digits

    a = make_stratified_split(y, random_state=42)
    b = make_stratified_split(y, random_state=43)

    assert a != b


# ---------- утечка индексов ----------


def test_duplicate_index_across_parts_raises_leak(digits):
    """Один индекс в двух частях → ValueError про утечку данных."""
    _, y = digits

    split = make_stratified_split(y)

    leaked = split["val"][0]
    split["train"] = split["train"] + [leaked]

    with pytest.raises(ValueError, match="утечк"):
        validate(split, y)


# ---------- выход за диапазон ----------


@pytest.mark.parametrize(
    "bad_index_kind",
    ["negative", "out_of_upper_bound"],
)
def test_out_of_range_index_raises(digits, bad_index_kind):
    """Индекс -1 или len(y) → ValueError про диапазон."""
    _, y = digits

    split = make_stratified_split(y)
    bad_index = -1 if bad_index_kind == "negative" else len(y)
    split["test"] = split["test"] + [bad_index]

    with pytest.raises(ValueError, match="диапазон"):
        validate(split, y)


# ---------- пропавшие классы ----------


def test_missing_class_in_val_raises(digits):
    """
    Убираем из val все объекты одного класса и перекладываем их в train
    (сумма сходится, утечки нет) → validate должен ругаться на классы.
    """
    _, y = digits

    split = make_stratified_split(y)
    val = split["val"]

    # класс, который точно присутствует в val
    target_class = min({int(y[i]) for i in val})

    moved = [i for i in val if int(y[i]) == target_class]
    assert moved, "в val должен быть хотя бы один объект выбранного класса"

    split["val"] = [i for i in val if int(y[i]) != target_class]
    split["train"] = split["train"] + moved

    # санити-чек: сумма и отсутствие утечек не нарушены
    all_idx = np.concatenate(
        [np.asarray(split[p], dtype=np.int64) for p in ("train", "val", "test")]
    )
    assert len(all_idx) == len(np.unique(all_idx)) == len(y)

    with pytest.raises(ValueError, match="класс"):
        validate(split, y)


# ---------- пропорции классов ----------


def test_class_proportions_close_to_overall(digits):
    """
    Доли классов в каждой части отличаются от общих не более чем на один объект.

    Это мягкая проверка качества стратификации: sklearn распределяет классы
    пропорционально, расхождение из-за округления — не больше одного объекта.
    """
    _, y = digits
    split = make_stratified_split(y)

    classes = np.unique(y)
    total = len(y)

    for part_name in ("train", "val", "test"):
        part_idx = np.asarray(split[part_name], dtype=np.int64)
        part_y = y[part_idx]
        part_size = len(part_idx)

        for c in classes:
            expected = (y == c).sum() * part_size / total
            actual = int((part_y == c).sum())
            assert abs(actual - expected) <= 1, (
                f"part={part_name}, class={int(c)}: "
                f"expected≈{expected:.2f}, got {actual}"
            )
