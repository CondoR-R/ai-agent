import subprocess

import pytest

import ai_agent.core.git_utils as git_utils
from ai_agent.core.git_utils import check_clean_status


# ---------- helpers ----------


def _git(*args: str, cwd) -> None:
    """Настоящий git для подготовки репозитория в тестах."""
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _make_run_git(cwd):
    """
    Возвращает замену для _run_git, работающую в указанной директории.

    Повторяет контракт оригинала:
      - возвращает stripped stdout
      - при ошибке git бросает RuntimeError
    """

    def _run_git(*args, **kwargs):
        kwargs["cwd"] = cwd
        try:
            return subprocess.run(
                ["git", *args],
                check=True,
                capture_output=True,
                text=True,
                **kwargs,
            ).stdout.strip()
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(exc.stderr) from exc

    return _run_git


# ---------- fixtures ----------


@pytest.fixture
def git_repo(tmp_path, monkeypatch):
    """Временный git-репозиторий с одним коммитом + подмена _run_git на него."""
    _git("init", cwd=tmp_path)
    _git("config", "user.email", "test@example.com", cwd=tmp_path)
    _git("config", "user.name", "Test User", cwd=tmp_path)

    (tmp_path / ".gitkeep").write_text("")
    _git("add", ".", cwd=tmp_path)
    _git("commit", "-m", "initial", cwd=tmp_path)

    # Ключевой момент: заставляем _run_git работать с tmp_path
    monkeypatch.setattr(git_utils, "_run_git", _make_run_git(tmp_path))
    return tmp_path


# ---------- tests ----------


def test_clean_repo_does_not_raise(git_repo):
    """В чистом репозитории функция не должна бросать SystemExit."""
    check_clean_status()


def test_untracked_file_raises_system_exit(git_repo):
    (git_repo / "new_file.txt").write_text("hello")

    with pytest.raises(SystemExit) as exc_info:
        check_clean_status()

    assert "new_file.txt" in str(exc_info.value.code)


def test_modified_tracked_file_raises_system_exit(git_repo):
    (git_repo / ".gitkeep").write_text("changed")

    with pytest.raises(SystemExit) as exc_info:
        check_clean_status()

    assert ".gitkeep" in str(exc_info.value.code)


def test_staged_file_raises_system_exit(git_repo):
    (git_repo / "staged.txt").write_text("staged")
    _git("add", "staged.txt", cwd=git_repo)

    with pytest.raises(SystemExit) as exc_info:
        check_clean_status()

    assert "staged.txt" in str(exc_info.value.code)


def test_deleted_tracked_file_raises_system_exit(git_repo):
    (git_repo / ".gitkeep").unlink()

    with pytest.raises(SystemExit) as exc_info:
        check_clean_status()

    assert ".gitkeep" in str(exc_info.value.code)


def test_not_a_git_repo_raises_runtime_error(tmp_path, monkeypatch):
    """В каталоге без .git git вернёт ошибку → функция должна бросить RuntimeError."""
    monkeypatch.setattr(git_utils, "_run_git", _make_run_git(tmp_path))

    with pytest.raises(RuntimeError):
        check_clean_status()
