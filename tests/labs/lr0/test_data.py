import json

import numpy as np
import pytest

from ai_agent.labs.lr0.data import (
    DATASET_NAME,
    load_raw,
    get_split,
    read_split,
)


@pytest.fixture(scope="module")
def digits_y():
    """Метки load_digits — нужны всем тестам get_split."""
    from sklearn.datasets import load_digits

    _, y = load_digits(return_X_y=True)
    return np.asarray(y, dtype=np.int64)


@pytest.fixture
def paths(tmp_path):
    """Пути к split.json и manifest.json внутри tmp_path + base_dir."""
    return {
        "split_path": tmp_path / "split.json",
        "manifest_path": tmp_path / "manifest.json",
        "base_dir": tmp_path,
    }


# ---------- создание ----------


def test_first_call_creates_split_and_manifest(paths, digits_y):
    """Первый вызов get_split создаёт файл, манифест и даёт 1077/360/360."""
    split, sha = get_split(digits_y, split_seed=0, **paths)

    assert len(split["train"]) == 1077
    assert len(split["val"]) == 360
    assert len(split["test"]) == 360

    assert paths["split_path"].exists()
    assert paths["manifest_path"].exists()

    # sha из файла совпадает с возвращённым
    import hashlib

    assert sha == hashlib.sha256(paths["split_path"].read_bytes()).hexdigest()

    # в манифесте есть запись
    manifest = json.loads(paths["manifest_path"].read_text(encoding="utf-8"))
    names = [d["name"] for d in manifest["datasets"]]
    assert names == [DATASET_NAME]


# ---------- повторный вызов ----------


def test_second_call_returns_same_split_and_hash(paths, digits_y):
    """Повторный вызов возвращает то же разбиение и тот же хеш."""
    split1, sha1 = get_split(digits_y, split_seed=0, **paths)
    split2, sha2 = get_split(digits_y, split_seed=0, **paths)

    assert split1 == split2
    assert sha1 == sha2


# ---------- подмена файла ----------


def test_modified_split_file_raises_changed(paths, digits_y):
    """Изменение split.json на один байт → ValueError про «изменён»."""
    get_split(digits_y, split_seed=0, **paths)

    # убираем завершающий перевод строки — содержимое остаётся валидным JSON,
    # но SHA-256 меняется ровно на один байт
    original = paths["split_path"].read_bytes()
    assert original.endswith(b"\n")
    paths["split_path"].write_bytes(original[:-1])

    with pytest.raises(ValueError, match="изменён"):
        get_split(digits_y, split_seed=0, **paths)


# ---------- несовпадение seed ----------


def test_different_split_seed_raises(paths, digits_y):
    """При существующем split.json другой split_seed → ValueError."""
    get_split(digits_y, split_seed=0, **paths)

    with pytest.raises(ValueError, match="split_seed"):
        get_split(digits_y, split_seed=1, **paths)


# ---------- запись удалена ----------


def test_missing_manifest_entry_raises(paths, digits_y):
    """split.json есть, а записи в манифесте нет → ValueError."""
    get_split(digits_y, split_seed=0, **paths)

    # очищаем datasets, оставляя структуру файла
    manifest = json.loads(paths["manifest_path"].read_text(encoding="utf-8"))
    manifest["datasets"] = []
    paths["manifest_path"].write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="манифест"):
        get_split(digits_y, split_seed=0, **paths)


# ---------- load_raw ----------


def test_load_raw_shapes_and_scale():
    """load_raw: (1797, 64), признаки в [0, 1]."""
    X, y = load_raw()

    assert X.shape == (1797, 64)
    assert y.shape == (1797,)

    # признаки масштабированы делением на 16 — исходный диапазон [0, 16]
    assert X.dtype == np.float32
    assert float(X.min()) >= 0.0
    assert float(X.max()) <= 1.0
