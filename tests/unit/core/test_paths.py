"""Тесты для src/ai_agent/core/paths.py."""

from ai_agent.core.paths import create_run_id_dir


# --- create_run_id_dir ---


def test_create_run_id_dir_when_missing(tmp_path):
    """Если каталогов нет, они создаются вместе с родителями"""
    nested = tmp_path / "a" / "b" / "runs"
    result = create_run_id_dir("test_run", runs_dir=nested)
    assert result.exists()
    assert result.is_dir()
    assert result == nested / "test_run"


def test_create_run_id_dir_when_exists(tmp_path):
    """Повторный вызов не падает и возвращает тот же путь"""
    first = create_run_id_dir("test_run", runs_dir=tmp_path)
    second = create_run_id_dir("test_run", runs_dir=tmp_path)
    assert first == second
    assert first.exists()
