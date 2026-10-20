"""Подготовка данных для ЛР0: разбиение load_digits и запись в манифест."""

import json

import sklearn

from ai_agent.core.paths import PROJECT_ROOT
from ai_agent.data.loaders import load_digits_dataset
from ai_agent.data.manifests import file_sha256, update_manifest
from ai_agent.data.splits import make_stratified_split, validate_split


def prepare_data() -> None:
    """Загружает load_digits, делает разбиение, пишет split.json и манифест."""
    test_size = 0.2
    val_size = 0.25
    random_state = 0

    # загрузка и разбиение датасета
    _, y = load_digits_dataset()
    split = make_stratified_split(
        y, test_size=test_size, val_size=val_size, random_state=random_state
    )
    validate_split(split, y)

    # запись сплита в файл
    split_path = PROJECT_ROOT / "data" / "split.json"
    if split_path.exists():
        print(
            "split.json уже существует, удалите вручную, если хотите пересоздать"
        )
        return

    split_path.write_text(
        json.dumps(split, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # создание записи в manifest
    sha = file_sha256(split_path)
    n_total = len(y)
    n_taken = len(split["train"]) + len(split["val"]) + len(split["test"])
    entry = {
        "source": "scikit-learn",
        "version": sklearn.__version__,
        "rule": f"test={test_size * 100}% (stratify, split_seed={random_state}); val={val_size * 100}% of the rest (stratify, split_seed={random_state})",
        "n_total": n_total,
        "n_taken": n_taken,
        "n_excluded": n_total - n_taken,
        "sizes": {
            "train": len(split["train"]),
            "val": len(split["val"]),
            "test": len(split["test"]),
        },
        "files": {"data/split.json": sha},
    }

    # запись в manifest.json
    manifest_path = PROJECT_ROOT / "data" / "manifest.json"
    update_manifest(manifest_path, "lr0_load_digits", entry)

    # вывод результатов для самопроверки
    print(f"Путь к split.json: {split_path}")
    print(
        f"Размеры: train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}"
    )
    print(f"sha файла: {sha}")
    print(f"Путь к манифесту: {manifest_path}")


def main() -> None:
    """Точка входа скрипта."""
    prepare_data()


if __name__ == "__main__":
    main()
