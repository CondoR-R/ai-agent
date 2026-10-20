"""Тесты для src/ai_agent/core/results.py."""

import json
from pathlib import Path

import pytest

from ai_agent.core.paths import METRICS_FILENAME, RESULTS_FILENAME
from ai_agent.core.results import (
    append_result,
    compute_metrics,
    load_results,
    validate_result,
)

REQUIRED = ("index", "y_true", "y_pred")


def _results_path(run_dir: Path) -> Path:
    """Путь к results.jsonl внутри run_dir."""
    return run_dir / RESULTS_FILENAME


def _metrics_path(run_dir: Path) -> Path:
    """Путь к metrics.json внутри run_dir."""
    return run_dir / METRICS_FILENAME


def _read_lines(run_dir: Path) -> list[str]:
    """Все непустые строки results.jsonl."""
    text = _results_path(run_dir).read_text(encoding="utf-8")
    return [line for line in text.splitlines() if line.strip()]


def _make_record(index: int, y_true: int = 0, y_pred: int = 0) -> dict:
    """Создаёт запись ЛР0 с полями index, y_true, y_pred."""
    return {"index": index, "y_true": y_true, "y_pred": y_pred}


# --- validate_result ---


def test_validate_result_ok():
    """Корректная запись проходит без исключения."""
    validate_result(_make_record(0), REQUIRED)


def test_validate_result_not_a_dict():
    """Не словарь → TypeError."""
    with pytest.raises(TypeError):
        validate_result([1, 2, 3], REQUIRED)


@pytest.mark.parametrize("field", REQUIRED)
def test_validate_result_missing_field(field):
    """Пропущено любое из обязательных полей → ValueError."""
    record = _make_record(0)
    del record[field]
    with pytest.raises(ValueError, match=field):
        validate_result(record, REQUIRED)


# --- append_result ---


def test_append_result_creates_file(run_dir):
    """Первый вызов создаёт results.jsonl."""
    append_result(run_dir, _make_record(0), REQUIRED)
    assert _results_path(run_dir).exists()


def test_append_result_writes_one_line(run_dir):
    """Одна запись → одна строка в файле."""
    append_result(run_dir, _make_record(0), REQUIRED)
    assert len(_read_lines(run_dir)) == 1


def test_append_result_appends_multiple(run_dir):
    """Три записи → три строки."""
    for i in range(3):
        append_result(run_dir, _make_record(i), REQUIRED)
    assert len(_read_lines(run_dir)) == 3


def test_append_result_lines_are_valid_json(run_dir):
    """Каждая строка results.jsonl — валидный JSON."""
    for i in range(3):
        append_result(run_dir, _make_record(i), REQUIRED)
    for line in _read_lines(run_dir):
        json.loads(line)  # не должно бросить


def test_append_result_does_not_use_indent(run_dir):
    """Запись — одна строка JSONL, а не многострочный JSON."""
    append_result(run_dir, _make_record(0), REQUIRED)
    lines = _read_lines(run_dir)
    assert len(lines) == 1
    # в компактном JSON нет переводов строк внутри записи
    assert "\n" not in lines[0]


def test_append_result_rejects_invalid_record(run_dir):
    """Невалидная запись не попадает в файл."""
    bad = {"index": 0, "y_true": 0}  # нет y_pred
    with pytest.raises(ValueError):
        append_result(run_dir, bad, REQUIRED)
    # файл либо не существует, либо пуст
    assert not _results_path(run_dir).exists() or _read_lines(run_dir) == []


# --- load_results ---


def test_load_results_empty_file(run_dir):
    """Пустой файл → пустой список."""
    _results_path(run_dir).write_text("", encoding="utf-8")
    assert load_results(run_dir) == []


def test_load_results_roundtrip(run_dir):
    """Запись и чтение дают те же объекты."""
    records = [
        _make_record(i, y_true=i % 10, y_pred=(i + 1) % 10) for i in range(5)
    ]
    for r in records:
        append_result(run_dir, r, REQUIRED)
    loaded = load_results(run_dir)
    assert loaded == records


def test_load_results_ignores_blank_lines(run_dir):
    """Пустые строки внутри файла игнорируются."""
    _results_path(run_dir).write_text(
        '\n{"index": 0, "y_true": 1, "y_pred": 1}\n\n',
        encoding="utf-8",
    )
    loaded = load_results(run_dir)
    assert len(loaded) == 1
    assert loaded[0]["index"] == 0


def test_load_results_raises_on_broken_line(run_dir):
    """Битая строка → ValueError с номером строки."""
    _results_path(run_dir).write_text(
        '{"index": 0, "y_true": 1, "y_pred": 1}\n{broken\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="строки 2"):
        load_results(run_dir)


def test_load_results_error_message_contains_snippet(run_dir):
    """Сообщение об ошибке содержит фрагмент проблемной строки."""
    _results_path(run_dir).write_text(
        "{broken\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="broken"):
        load_results(run_dir)


# --- compute_metrics ---


def _accuracy(records: list[dict]) -> dict:
    """Простейшая metric_fn для тестов: accuracy по полям y_true/y_pred."""
    if not records:
        return {"accuracy": 0.0, "n": 0}
    correct = sum(1 for r in records if r["y_true"] == r["y_pred"])
    return {"accuracy": correct / len(records), "n": len(records)}


def test_compute_metrics_creates_file(run_dir):
    """metrics.json создаётся."""
    append_result(run_dir, _make_record(0, y_true=1, y_pred=1), REQUIRED)
    compute_metrics(run_dir, _accuracy)
    assert _metrics_path(run_dir).exists()


def test_compute_metrics_values(run_dir):
    """metrics.json содержит ожидаемые значения."""
    append_result(run_dir, _make_record(0, y_true=1, y_pred=1), REQUIRED)
    append_result(run_dir, _make_record(1, y_true=2, y_pred=3), REQUIRED)
    compute_metrics(run_dir, _accuracy)
    metrics = json.loads(_metrics_path(run_dir).read_text(encoding="utf-8"))
    assert metrics["accuracy"] == 0.5
    assert metrics["n"] == 2


def test_compute_metrics_is_multiline(run_dir):
    """metrics.json записан с отступами."""
    append_result(run_dir, _make_record(0), REQUIRED)
    compute_metrics(run_dir, _accuracy)
    text = _metrics_path(run_dir).read_text(encoding="utf-8")
    assert "\n" in text


def test_compute_metrics_reads_from_file(run_dir):
    """Метрики считаются из файла, а не из памяти.

    Дописываем запись в results.jsonl в обход append_result —
    метрики должны отразить изменения.
    """
    append_result(run_dir, _make_record(0, y_true=1, y_pred=1), REQUIRED)
    compute_metrics(run_dir, _accuracy)
    first = json.loads(_metrics_path(run_dir).read_text(encoding="utf-8"))
    assert first["n"] == 1

    # дописываем ещё одну запись напрямую в файл
    with open(_results_path(run_dir), "a", encoding="utf-8") as f:
        f.write(json.dumps(_make_record(1, y_true=2, y_pred=9)) + "\n")

    compute_metrics(run_dir, _accuracy)
    second = json.loads(_metrics_path(run_dir).read_text(encoding="utf-8"))
    assert second["n"] == 2
    assert second["accuracy"] == 0.5
