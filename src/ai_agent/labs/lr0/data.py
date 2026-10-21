"""Данные ЛР0: load_digits, разбиение, split.json и запись в манифест."""

import json
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits

from ai_agent.core import manifest
from ai_agent.core.paths import DATA_DIR, MANIFEST_PATH, PROJECT_ROOT

from .split import make_stratified_split, validate

DATASET_NAME = "digits"
SPLIT_PATH = DATA_DIR / "split.json"

TEST_SIZE = 0.2  # доля test от всего набора
VAL_SIZE = 0.25  # доля val от остатка после отделения test
FEATURE_SCALE = 16.0  # признаки load_digits лежат в [0, 16]


@dataclass(frozen=True)
class DigitsData:
    """Данные ЛР0: признаки, метки, разбиение и хеш split.json"""

    X: np.ndarray
    y: np.ndarray
    split: dict[str, list[int]]
    split_sha256: str

    def part(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Признаки и метки одной части

        params:
            name - train, val или test

        return: кортеж (X, y) для указанной части
        """
        idx = self.split[name]
        return self.X[idx], self.y[idx]


def load_raw() -> tuple[np.ndarray, np.ndarray]:
    """Загружает load_digits и масштабирует признаки делением на 16.

    Масштаб известен заранее, статистики по данным не вычисляются.

    return: кортеж (X float32 формы (1797, 64), y формы (1797,))
    """
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float32) / 16.0
    y = np.asarray(y, dtype=np.int64)
    return X, y


def write_split(path: Path, split: dict[str, list[int]]) -> None:
    """Записывает разбиение в JSON.

    Формат фиксирован (indent=2, перевод строки в конце, ключи по алфавиту):
    хеш считается по байтам, поэтому любое изменение форматирования
    изменит SHA-256.

    params:
        path - путь к split.json
        split - словарь с индексами train, val, test
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(split, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_split(path: Path) -> dict[str, list[int]]:
    """Читает разбиение из JSON

    params:
        path - путь к split.json

    return: словарь с индексами train, val, test
    """
    return json.loads(path.read_text(encoding="utf-8"))


def _build_entry(
    split: dict[str, list[int]],
    split_seed: int,
    split_path: Path,
    base_dir: Path,
) -> dict:
    """Собирает запись для манифеста (формат: manifest.validate_entry)"""
    total = sum(len(idx) for idx in split.values())
    return {
        "source": "scikit-learn load_digits",
        "version": f"scikit-learn=={version('scikit-learn')}",
        "selection": {"total": total, "taken": total, "excluded": 0},
        "rule": (
            f"stratified split: test {TEST_SIZE:.0%} of all, "
            f"val {VAL_SIZE:.0%} of the rest, split_seed={split_seed}"
        ),
        "split_seed": split_seed,
        "sizes": {name: len(idx) for name, idx in split.items()},
        "files": [
            {
                "path": split_path.relative_to(base_dir).as_posix(),
                "sha256": manifest.file_sha256(split_path),
                "records": total,
            }
        ],
    }


def get_split(
    y: np.ndarray,
    split_seed: int,
    split_path: Path = SPLIT_PATH,
    manifest_path: Path = MANIFEST_PATH,
    base_dir: Path = PROJECT_ROOT,
) -> tuple[dict[str, list[int]], str]:
    """Возвращает разбиение: создаёт его при первом запуске, иначе читает
    и сверяет с манифестом.

    Если split.json нет: строит разбиение, проверяет, пишет файл и запись
    в манифест. Если есть: читает, проверяет, сверяет SHA-256 и split_seed
    с записью в манифесте. Файл никогда не пересоздаётся молча.

    params:
        y - массив меток
        split_seed - сид разбиения
        split_path - путь к split.json
        manifest_path - путь к data/manifest.json
        base_dir - каталог, от которого записываются пути в манифесте

    return: кортеж (разбиение, SHA-256 файла split.json)

    raise:
        ValueError, если файл есть, а записи в манифесте нет, хеш или
        split_seed не совпали либо разбиение не прошло validate_split
    """
    if not split_path.exists():
        split = make_stratified_split(y, TEST_SIZE, VAL_SIZE, split_seed)
        validate(split, y)
        write_split(split_path, split)
        manifest.update(
            manifest_path,
            DATASET_NAME,
            _build_entry(split, split_seed, split_path, base_dir),
        )
        return split, manifest.file_sha256(split_path)

    split = read_split(split_path)
    validate(split, y)

    entry = manifest.find_entry(manifest_path, DATASET_NAME)
    if entry is None:
        raise ValueError(
            f"{split_path.name} существует, но записи '{DATASET_NAME}' "
            "в манифесте нет. Удалите split.json, чтобы создать разбиение заново."
        )
    manifest.verify_files(entry, base_dir)
    if entry.get("split_seed") != split_seed:
        raise ValueError(
            f"split_seed в конфигурации ({split_seed}) не совпадает с "
            f"split_seed разбиения в манифесте ({entry.get('split_seed')})"
        )
    return split, manifest.file_sha256(split_path)


def load(split_seed: int = 0) -> DigitsData:
    """Загружает данные ЛР0 вместе с разбиением

    params:
        split_seed - сид разбиения (из конфигурации)

    return: DigitsData с X, y, разбиением и хешем split.json
    """
    X, y = load_raw()
    split, sha = get_split(y, split_seed)
    return DigitsData(X=X, y=y, split=split, split_sha256=sha)
