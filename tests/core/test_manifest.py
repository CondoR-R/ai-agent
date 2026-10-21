import json

import pytest

from ai_agent.core import manifest


def _entry(version: str = "1.0") -> dict:
    """Валидный payload для update (без name — его подставляет update)."""
    return {
        "source": "pypi:pkg==1.0",
        "version": version,
        "selection": {"total": 10, "taken": 10, "excluded": 0},
        "files": [
            {"path": "data/raw/f.txt", "sha256": "a" * 64, "records": 5},
        ],
    }


def test_update_twice_same_name_keeps_single_entry(tmp_path):
    """update дважды с одним именем оставляет одну запись"""
    path = tmp_path / "manifest.json"

    manifest.update(path, "ds1", _entry(version="1.0"))
    manifest.update(path, "ds1", _entry(version="2.0"))

    datasets = manifest.load(path)["datasets"]
    assert len(datasets) == 1
    assert datasets[0]["name"] == "ds1"
    assert datasets[0]["version"] == "2.0"


@pytest.mark.parametrize(
    "bad_entry",
    [
        # без files — validate_entry упадёт на отсутствующем поле
        {
            "source": "s",
            "version": "1",
            "selection": {"total": 1, "taken": 1, "excluded": 0},
        },
        # пустой список files
        {
            "source": "s",
            "version": "1",
            "selection": {"total": 1, "taken": 1, "excluded": 0},
            "files": [],
        },
        # records: 0
        {
            "source": "s",
            "version": "1",
            "selection": {"total": 1, "taken": 1, "excluded": 0},
            "files": [
                {"path": "f.txt", "sha256": "a" * 64, "records": 0},
            ],
        },
    ],
    ids=["no_files_key", "empty_files", "zero_records"],
)
def test_invalid_entry_rejected_and_manifest_untouched(tmp_path, bad_entry):
    """Запись без files или с records: 0 отклоняется,
    а файл манифеста при этом не изменяется"""
    path = tmp_path / "manifest.json"

    # создаём валидное состояние
    manifest.update(path, "existing", _entry())
    before = path.read_text(encoding="utf-8")

    with pytest.raises((ValueError, KeyError, TypeError)):
        manifest.update(path, "bad", bad_entry)

    assert path.read_text(encoding="utf-8") == before
    # атомарная запись не должна оставить .tmp
    assert not (tmp_path / "manifest.json.tmp").exists()


def test_load_duplicate_names_raises(tmp_path):
    """load на файле с повторяющимися именами бросает ValueError"""
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"datasets": [{"name": "ds1"}, {"name": "ds1"}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="имена"):
        manifest.load(path)


def test_verify_files_detects_one_byte_change(tmp_path):
    """verify_files ловит изменение файла на один байт"""
    target = tmp_path / "data" / "file.txt"
    target.parent.mkdir(parents=True)
    target.write_text("hello", encoding="utf-8")

    entry = {
        "name": "ds1",
        "files": [
            {
                "path": "data/file.txt",
                "sha256": manifest.file_sha256(target),
                "records": 1,
            },
        ],
    }

    # исходный файл проходит проверку
    manifest.verify_files(entry, base_dir=tmp_path)

    # меняем ровно один байт: 'o' -> 'p'
    target.write_text("hellp", encoding="utf-8")

    with pytest.raises(ValueError, match="изменён"):
        manifest.verify_files(entry, base_dir=tmp_path)
