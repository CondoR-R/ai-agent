"""Тесты для src/ai_agent/core/run.py."""

import json
import subprocess
from pathlib import Path

import pytest

from ai_agent.core.paths import (
    CONFIG_FILENAME,
    ERROR_FILENAME,
    METADATA_FILENAME,
    METRICS_FILENAME,
    RESULTS_FILENAME,
)
from ai_agent.core.run import run_experiment

REQUIRED = ("index", "y_true", "y_pred")


# --- помощники ---


def _prepare_ok(cfg):
    """prepare_fn-заглушка: ничего не делает."""
    return None


def _load_three(cfg):
    """load_records-заглушка: три записи."""
    return [0, 1, 2]


def _process_identity(record, context):
    """process_one-заглушка: предсказание = истина."""
    return {"index": record, "y_true": record % 10, "y_pred": record % 10}


def _accuracy(records):
    """compute_metrics_fn-заглушка: accuracy."""
    if not records:
        return {"acc": 0.0, "n": 0}
    correct = sum(1 for r in records if r["y_true"] == r["y_pred"])
    return {"acc": correct / len(records), "n": len(records)}


def _commit_file(repo: Path, relpath: str, content: str) -> Path:
    """Пишет файл в репозиторий и коммитит."""
    p = repo / relpath
    p.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", f"add {relpath}"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return p


def _valid_config_yaml(lr: float = 0.001) -> str:
    return (
        "experiment: lr0\n"
        "split_seed: 0\n"
        "seed: 0\n"
        "model: {n_layers: 1, hidden: 64, dropout: 0.0}\n"
        f"train: {{lr: {lr}, weight_decay: 0.0, batch_size: 64, epochs: 30}}\n"
        "runtime: {device: cpu, num_threads: 1, tracked_libs: [numpy]}\n"
    )


# --- фикстуры ---


@pytest.fixture
def config_in_repo(git_repo, monkeypatch):
    """Корректный конфиг, закоммиченный в git_repo. chdir → git_repo."""
    monkeypatch.chdir(git_repo)
    return _commit_file(git_repo, "config.yaml", _valid_config_yaml())


@pytest.fixture
def runs_in_tmp(tmp_path, monkeypatch):
    """Перенаправляет create_run_id_dir в tmp_path/runs."""
    base = tmp_path / "runs"

    def fake_create(run_id, runs_dir=None):
        base_dir = runs_dir if runs_dir is not None else base
        base_dir.mkdir(parents=True, exist_ok=True)
        d = base_dir / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    monkeypatch.setattr("ai_agent.core.run.create_run_id_dir", fake_create)
    return base


# --- успешный прогон ---


def test_successful_run(config_in_repo, runs_in_tmp):
    """Все файлы создаются, статус completed."""
    run_dir = run_experiment(
        config_path=config_in_repo,
        work="lr0",
        series="test",
        prepare_fn=_prepare_ok,
        load_records=_load_three,
        process_one=_process_identity,
        compute_metrics_fn=_accuracy,
        required_fields=REQUIRED,
        data_hashes={"fake": "0" * 64},
    )

    assert run_dir.exists()
    assert (run_dir / CONFIG_FILENAME).exists()
    assert (run_dir / METADATA_FILENAME).exists()
    assert (run_dir / RESULTS_FILENAME).exists()
    assert (run_dir / METRICS_FILENAME).exists()

    meta = json.loads((run_dir / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert meta["status"] == "completed"
    assert meta["finished_at"]
    assert meta["data_sha256"] == {"fake": "0" * 64}

    metrics = json.loads((run_dir / METRICS_FILENAME).read_text(encoding="utf-8"))
    assert metrics["n"] == 3
    assert metrics["acc"] == 1.0


def test_successful_run_writes_all_results(config_in_repo, runs_in_tmp):
    """results.jsonl содержит все три записи."""
    run_dir = run_experiment(
        config_path=config_in_repo,
        work="lr0",
        series="test",
        prepare_fn=_prepare_ok,
        load_records=_load_three,
        process_one=_process_identity,
        compute_metrics_fn=_accuracy,
        required_fields=REQUIRED,
        data_hashes={},
    )

    lines = (run_dir / RESULTS_FILENAME).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    for i, line in enumerate(lines):
        assert json.loads(line)["index"] == i


def test_cli_overrides_applied(config_in_repo, runs_in_tmp):
    """cli_overrides применяются и попадают в metadata.json."""
    run_dir = run_experiment(
        config_path=config_in_repo,
        work="lr0",
        series="test",
        prepare_fn=_prepare_ok,
        load_records=_load_three,
        process_one=_process_identity,
        compute_metrics_fn=_accuracy,
        required_fields=REQUIRED,
        data_hashes={},
        cli_overrides={"seed": 42},
    )
    meta = json.loads((run_dir / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert meta["seed"] == 42


# --- падение в процессе ---


def test_failure_in_process_one(config_in_repo, runs_in_tmp):
    """Падение на 2-й записи: первая сохранена, статус failed, error.txt."""

    def failing(record, context):
        if record == 1:
            raise RuntimeError("boom on record 1")
        return {"index": record, "y_true": record, "y_pred": record}

    with pytest.raises(RuntimeError, match="boom on record 1"):
        run_experiment(
            config_path=config_in_repo,
            work="lr0",
            series="test",
            prepare_fn=_prepare_ok,
            load_records=_load_three,
            process_one=failing,
            compute_metrics_fn=_accuracy,
            required_fields=REQUIRED,
            data_hashes={},
        )

    runs = list(runs_in_tmp.iterdir())
    assert len(runs) == 1
    run_dir = runs[0]

    meta = json.loads((run_dir / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert meta["status"] == "failed"
    assert "error_message" in meta
    assert "boom on record 1" in meta["error_message"]

    lines = (run_dir / RESULTS_FILENAME).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["index"] == 0

    err = (run_dir / ERROR_FILENAME).read_text(encoding="utf-8")
    assert "RuntimeError" in err
    assert "boom on record 1" in err


def test_failure_in_prepare(config_in_repo, runs_in_tmp):
    """Падение в prepare_fn: статус failed, results.jsonl отсутствует или пуст."""

    def failing_prepare(cfg):
        raise RuntimeError("prepare failed")

    with pytest.raises(RuntimeError, match="prepare failed"):
        run_experiment(
            config_path=config_in_repo,
            work="lr0",
            series="test",
            prepare_fn=failing_prepare,
            load_records=_load_three,
            process_one=_process_identity,
            compute_metrics_fn=_accuracy,
            required_fields=REQUIRED,
            data_hashes={},
        )

    runs = list(runs_in_tmp.iterdir())
    assert len(runs) == 1
    run_dir = runs[0]

    meta = json.loads((run_dir / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert meta["status"] == "failed"

    # results.jsonl не должно быть или он пуст
    results_path = run_dir / RESULTS_FILENAME
    assert not results_path.exists() or results_path.read_text(encoding="utf-8").strip() == ""


# --- проверки до создания каталога ---


def test_dirty_git_raises_before_dir(config_in_repo, runs_in_tmp, git_repo):
    """Грязный git → SystemExit, каталог не создан."""
    (git_repo / "dirty.txt").write_text("untracked", encoding="utf-8")

    with pytest.raises(SystemExit):
        run_experiment(
            config_path=config_in_repo,
            work="lr0",
            series="test",
            prepare_fn=_prepare_ok,
            load_records=_load_three,
            process_one=_process_identity,
            compute_metrics_fn=_accuracy,
            required_fields=REQUIRED,
            data_hashes={},
        )

    assert not runs_in_tmp.exists() or list(runs_in_tmp.iterdir()) == []


def test_invalid_config_raises_before_dir(git_repo, monkeypatch, runs_in_tmp):
    """lr=-1 → ValueError, каталог не создан."""
    monkeypatch.chdir(git_repo)
    cfg_path = _commit_file(git_repo, "config.yaml", _valid_config_yaml(lr=-1))

    with pytest.raises(ValueError, match="lr"):
        run_experiment(
            config_path=cfg_path,
            work="lr0",
            series="test",
            prepare_fn=_prepare_ok,
            load_records=_load_three,
            process_one=_process_identity,
            compute_metrics_fn=_accuracy,
            required_fields=REQUIRED,
            data_hashes={},
        )

    assert not runs_in_tmp.exists() or list(runs_in_tmp.iterdir()) == []


# --- воспроизводимость ---


def test_reproducibility(config_in_repo, runs_in_tmp, monkeypatch):
    """Два запуска с одним seed дают одинаковый results.jsonl."""
    # Патчим get_run_id, чтобы run_id гарантированно различались,
    # даже если оба запуска попадут в одну и ту же секунду.
    import itertools

    counter = itertools.count()

    def fake_get_run_id(work, series, seed):
        return f"{work}_{series}_{seed}_{next(counter)}"

    monkeypatch.setattr("ai_agent.core.run.get_run_id", fake_get_run_id)

    def run_once():
        return run_experiment(
            config_path=config_in_repo,
            work="lr0",
            series="test",
            prepare_fn=_prepare_ok,
            load_records=_load_three,
            process_one=_process_identity,
            compute_metrics_fn=_accuracy,
            required_fields=REQUIRED,
            data_hashes={},
        )

    run_dir_1 = run_once()
    run_dir_2 = run_once()

    assert run_dir_1 != run_dir_2

    lines_1 = (run_dir_1 / RESULTS_FILENAME).read_text(encoding="utf-8")
    lines_2 = (run_dir_2 / RESULTS_FILENAME).read_text(encoding="utf-8")
    assert lines_1 == lines_2

    meta_1 = json.loads((run_dir_1 / METADATA_FILENAME).read_text(encoding="utf-8"))
    meta_2 = json.loads((run_dir_2 / METADATA_FILENAME).read_text(encoding="utf-8"))
    # данные и seed совпадают, run_id/время — нет
    assert meta_1["seed"] == meta_2["seed"]
    assert meta_1["commit"] == meta_2["commit"]
    assert meta_1["data_sha256"] == meta_2["data_sha256"]
