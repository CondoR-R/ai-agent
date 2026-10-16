"""Тесты для src/ai_agent/core/metadata.py."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from ai_agent.core.metadata import (
    REQUIRED_FIELDS,
    update_status,
    validate_metadata,
    write_error,
    write_metadata,
)
from ai_agent.core.paths import METADATA_FILENAME


def _metadata_path(run_dir: Path) -> Path:
    """Путь к metadata.json внутри run_dir."""
    return run_dir / METADATA_FILENAME


def _read_metadata(run_dir: Path) -> dict:
    """Читает metadata.json из run_dir."""
    return json.loads(_metadata_path(run_dir).read_text(encoding="utf-8"))


# --- validate_metadata: корректные словари ---


def test_validate_running_ok(valid_metadata):
    """Корректный словарь со status=running проходит валидацию"""
    validate_metadata(valid_metadata, "running")


def test_validate_without_expected_status(valid_metadata):
    """Без expected_status проверяется только базовое"""
    validate_metadata(valid_metadata)


def test_validate_completed_ok(valid_metadata):
    """status=completed с заполненным finished_at проходит"""
    m = deepcopy(valid_metadata)
    m["status"] = "completed"
    m["finished_at"] = "2026-10-16T10:00:00"
    validate_metadata(m, "completed")


def test_validate_failed_ok(valid_metadata):
    """status=failed с заполненным finished_at проходит"""
    m = deepcopy(valid_metadata)
    m["status"] = "failed"
    m["finished_at"] = "2026-10-16T10:00:00"
    validate_metadata(m, "failed")


def test_validate_running_with_finished_at_none(valid_metadata):
    """finished_at=None при running считается отсутствием"""
    m = deepcopy(valid_metadata)
    m["finished_at"] = None
    validate_metadata(m, "running")


# --- validate_metadata: ошибки ---


def test_validate_not_a_dict():
    """Не словарь → TypeError"""
    with pytest.raises(TypeError):
        validate_metadata([1, 2, 3])


def test_validate_none_is_not_a_dict():
    """None → TypeError"""
    with pytest.raises(TypeError):
        validate_metadata(None)


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_validate_missing_required_field(valid_metadata, field):
    """Отсутствие любого обязательного поля → ValueError"""
    m = deepcopy(valid_metadata)
    del m[field]
    with pytest.raises(ValueError, match=field):
        validate_metadata(m)


def test_validate_unknown_status(valid_metadata):
    """Неизвестный статус → ValueError"""
    m = deepcopy(valid_metadata)
    m["status"] = "pending"
    with pytest.raises(ValueError, match="status"):
        validate_metadata(m)


def test_validate_expected_status_mismatch(valid_metadata):
    """expected_status не совпадает с фактическим → ValueError"""
    with pytest.raises(ValueError, match="expected_status"):
        validate_metadata(valid_metadata, "completed")


def test_validate_completed_without_finished_at(valid_metadata):
    """completed без finished_at → ValueError"""
    m = deepcopy(valid_metadata)
    m["status"] = "completed"
    with pytest.raises(ValueError, match="finished_at"):
        validate_metadata(m)


def test_validate_completed_with_finished_at_none(valid_metadata):
    """completed с finished_at=None → ValueError"""
    m = deepcopy(valid_metadata)
    m["status"] = "completed"
    m["finished_at"] = None
    with pytest.raises(ValueError, match="finished_at"):
        validate_metadata(m)


def test_validate_failed_without_finished_at(valid_metadata):
    """failed без finished_at → ValueError"""
    m = deepcopy(valid_metadata)
    m["status"] = "failed"
    with pytest.raises(ValueError, match="finished_at"):
        validate_metadata(m)


def test_validate_running_with_finished_at(valid_metadata):
    """running с заполненным finished_at → ValueError"""
    m = deepcopy(valid_metadata)
    m["finished_at"] = "2026-10-16T10:00:00"
    with pytest.raises(ValueError, match="finished_at"):
        validate_metadata(m, "running")


# --- write_metadata ---


def test_write_metadata_creates_file(run_dir, valid_metadata):
    """write_metadata создаёт metadata.json"""
    write_metadata(run_dir, valid_metadata)
    assert _metadata_path(run_dir).exists()


def test_write_metadata_roundtrip(run_dir, valid_metadata):
    """Записанный словарь читается обратно без изменений"""
    write_metadata(run_dir, valid_metadata)
    assert _read_metadata(run_dir) == valid_metadata


def test_write_metadata_rejects_non_running(run_dir, valid_metadata):
    """write_metadata ожидает status=running"""
    m = deepcopy(valid_metadata)
    m["status"] = "completed"
    m["finished_at"] = "2026-10-16T10:00:00"
    with pytest.raises(ValueError):
        write_metadata(run_dir, m)


def test_write_metadata_does_not_write_on_invalid(run_dir, valid_metadata):
    """При ошибке валидации файл не создаётся"""
    m = deepcopy(valid_metadata)
    del m["commit"]
    with pytest.raises(ValueError):
        write_metadata(run_dir, m)
    assert not _metadata_path(run_dir).exists()


def test_write_metadata_overwrites_existing(run_dir, valid_metadata):
    """Повторный вызов перезаписывает файл новыми данными"""
    write_metadata(run_dir, valid_metadata)
    m2 = deepcopy(valid_metadata)
    m2["seed"] = 999
    write_metadata(run_dir, m2)
    assert _read_metadata(run_dir)["seed"] == 999


# --- update_status ---


def test_update_status_preserves_other_fields(run_dir, valid_metadata):
    """Все обязательные поля остаются на месте после update"""
    write_metadata(run_dir, valid_metadata)
    update_status(run_dir, "completed")
    loaded = _read_metadata(run_dir)
    for field in REQUIRED_FIELDS:
        assert field in loaded
    assert loaded["seed"] == valid_metadata["seed"]
    assert loaded["commit"] == valid_metadata["commit"]
    assert loaded["versions"] == valid_metadata["versions"]


def test_update_status_rejects_running(run_dir, valid_metadata):
    """running нельзя поставить через update_status"""
    write_metadata(run_dir, valid_metadata)
    with pytest.raises(ValueError, match="running"):
        update_status(run_dir, "running")


def test_update_status_rejects_unknown(run_dir, valid_metadata):
    """Неизвестный статус не принимается"""
    write_metadata(run_dir, valid_metadata)
    with pytest.raises(ValueError, match="pending"):
        update_status(run_dir, "pending")


# --- write_error ---


def test_write_error_preserves_other_fields(run_dir, valid_metadata):
    """Регресс-тест: метаданные не теряются при записи ошибки"""
    write_metadata(run_dir, valid_metadata)
    write_error(run_dir, RuntimeError("boom"))
    loaded = _read_metadata(run_dir)
    for field in REQUIRED_FIELDS:
        assert field in loaded
    assert loaded["commit"] == valid_metadata["commit"]
