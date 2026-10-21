import json

import pytest

from ai_agent.core import metadata
from ai_agent.core.metadata import update_status


# ---------- отвергаемые статусы ----------


@pytest.mark.parametrize(
    "bad_status",
    ["running", "unknown"],
)
def test_non_final_status_rejected(tmp_path, bad_status):
    """
    Любой статус, не входящий в FINAL_STATUSES, должен отвергаться.

    Проверка статуса выполняется ДО чтения metadata.json, поэтому файл
    в run_dir намеренно отсутствует — это дополнительно гарантирует,
    что ошибка возникает именно из-за статуса, а не из-за файловой системы.
    """
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()

    with pytest.raises(ValueError, match=bad_status or "некорректен"):
        update_status(run_dir, bad_status)


def test_running_status_rejected_explicitly(tmp_path):
    """Отдельный явный тест на 'running' — самый частый неверный статус."""
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()

    with pytest.raises(ValueError, match="running"):
        update_status(run_dir, "running")


def test_rejection_does_not_touch_metadata(tmp_path):
    """При некорректном статусе существующий metadata.json не должен меняться."""
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()
    metadata_path = run_dir / metadata.METADATA_FILENAME
    original = {"status": "pending", "run_id": "run_001"}
    metadata_path.write_text(json.dumps(original), encoding="utf-8")

    with pytest.raises(ValueError):
        update_status(run_dir, "running")

    assert json.loads(metadata_path.read_text(encoding="utf-8")) == original


def test_rejection_does_not_create_metadata(tmp_path):
    """При некорректном статусе и отсутствующем metadata.json файл не создаётся."""
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()

    with pytest.raises(ValueError):
        update_status(run_dir, "running")

    assert not (run_dir / metadata.METADATA_FILENAME).exists()


# ---------- положительный контроль: финальные статусы принимаются ----------


@pytest.mark.parametrize("final_status", ["completed", "failed"])
def test_final_status_accepted(tmp_path, monkeypatch, final_status):
    """
    completed / failed проходят проверку статуса и записываются в файл.

    Чтение/валидация/запись metadata.json подменяются, чтобы тест проверял
    ТОЛЬКО логику выбора статуса, а не схему metadata.json.
    """
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()
    metadata_path = run_dir / metadata.METADATA_FILENAME
    metadata_path.write_text(
        json.dumps({"run_id": "run_001"}), encoding="utf-8"
    )

    monkeypatch.setattr(metadata, "validate", lambda *_a, **_kw: None)
    monkeypatch.setattr(metadata, "_now", lambda: "2026-01-01T00:00:00Z")

    update_status(run_dir, final_status)

    written = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert written["status"] == final_status
    assert written["finished_at"] == "2026-01-01T00:00:00Z"
