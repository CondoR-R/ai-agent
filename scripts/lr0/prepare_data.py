"""Скрипт подготовки данных"""

from importlib.metadata import version

from ai_agent import paths
from ai_agent.labs.lr0 import data

TEST_SIZE = 0.2
VAL_SIZE = 0.25
SPLIT_SEED = 0


def main():
    # получение датасета и его разбиение
    _, labels = data.load_data()
    split = data.get_split(labels, TEST_SIZE, VAL_SIZE, SPLIT_SEED)
    data.validate_split(labels, split)

    # сохранение разбиения
    split_path = paths.DATA_DIR / "split.json"
    data.save_split(split_path, split)
    split_sha256 = data.file_sha256(split_path)

    # сохранение в manifest.json
    manifest = {
        "source": "scikit-learn",
        "version": version("scikit-learn"),
        "rule": {
            "test_size": TEST_SIZE,
            "val_size": VAL_SIZE,
            "split_seed": SPLIT_SEED,
            "stratify": "y",
        },
        "sizes": {
            "train": len(split["train"]),
            "val": len(split["val"]),
            "test": len(split["test"]),
        },
        "sha256": split_sha256,
    }
    data.write_manifest(manifest, paths.MANIFEST_PATH)


if __name__ == "__main__":
    main()
