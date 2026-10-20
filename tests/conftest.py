"""Общие фикстуры для тестов."""

from pathlib import Path

import pytest
import subprocess


@pytest.fixture
def valid_metadata() -> dict:
    """Корректный словарь metadata со статусом running.

    Используется как база: тесты на «проходит» берут его как есть,
    тесты на «падает» копируют и мутируют нужное поле.
    """
    return {
        "run_id": "lr0_base_seed0_2026-10-16_09-03-07",
        "work": "lr0",
        "series": "base_seeds",
        "status": "running",
        "started_at": "2026-10-16T09:03:07",
        "commit": "a" * 40,
        "device": "cpu",
        "seed": 0,
        "split_seed": 0,
        "num_threads": 1,
        "data_sha256": {
            "data/split.json": "b" * 64,
        },
        "versions": {
            "python": "3.12.5",
            "torch": "2.14.1",
            "sklearn": "1.9.1",
            "numpy": "2.5.3",
            "mlflow": "3.17.0",
            "optuna": "5.0.0",
        },
    }


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """Временный каталог запуска runs/<run_id>/ внутри tmp_path.

    Pytest создаёт свежий tmp_path на каждый тест и удаляет его
    после прогона, поэтому тесты изолированы друг от друга и не
    трогают реальный runs/.
    """
    rd = tmp_path / "runs" / "test_run"
    rd.mkdir(parents=True, exist_ok=True)
    return rd


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Временный git-репозиторий с одним коммитом.

    Возвращает Path к каталогу репозитория. Внутри него установлены
    user.email и user.name, чтобы коммиты не падали на чистой машине.
    """
    repo = tmp_path / "repo"
    repo.mkdir()

    def run(*args: str) -> None:
        subprocess.run(args, cwd=repo, check=True, capture_output=True)

    run("git", "init", "-b", "main")
    run("git", "config", "user.email", "test@example.com")
    run("git", "config", "user.name", "Test User")

    (repo / "README.md").write_text("initial\n", encoding="utf-8")
    run("git", "add", "README.md")
    run("git", "commit", "-m", "initial commit")

    return repo
