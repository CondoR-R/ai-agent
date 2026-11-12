"""Сборка отчёта ЛР1: подстановки, ручные блоки, таблицы."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "lr1_report", ROOT / "scripts" / "lr1" / "report.py"
)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def test_render_replaces_placeholders():
    text = report._render("a {{ x }} b {{ missing }}", {"x": "1"})
    assert text == "a 1 b {{ missing }}"


def test_merge_manual_keeps_previous_content():
    template = "<!-- manual:forecast -->new<!-- /manual:forecast -->"
    previous = "<!-- manual:forecast -->old<!-- /manual:forecast -->"
    merged = report._merge_manual(template, previous)
    assert "old" in merged and "new" not in merged


def test_table_escapes_nothing_and_keeps_columns():
    table = report._table(["a", "b"], [[1, 2], [3, 4]])
    assert table.splitlines()[0] == "| a | b |"
    assert table.splitlines()[2] == "| 1 | 2 |"


def test_exp2_section_uses_saved_metrics():
    exp2 = {
        "run_id": "run",
        "series": "exp2",
        "config": {"exp2": {"repeats": 1, "prompt_tokens": 8}},
        "metrics": {
            "by_output": {
                "16": {
                    "n_planned": 1,
                    "n_ok": 1,
                    "completion_tokens_median": 16,
                    "e2e_median_s": 0.5,
                    "ttft_median_s": 0.1,
                    "tpot_median_s": 0.03,
                    "decode_tok_s_median": 33.0,
                }
            },
            "e2e_fit": {
                "intercept_s": 0.07,
                "slope_s_per_token": 0.027,
                "intercept_minus_ttft_s": -0.03,
                "slope_minus_tpot_s": -0.003,
            },
            "decode_lower_bound_s": 0.027,
            "tpot_median_s": 0.03,
            "tpot_over_lower_bound": 1.1,
            "n_wrong_output_len": 0,
        },
    }
    sections = report._exp2_sections(exp2)
    assert "Таблица 3" in sections["exp2_table"]
    assert "Нижняя граница TPOT" in sections["exp2_summary"]


def test_exp3_section_handles_level_without_success():
    exp3 = {
        "run_id": "run",
        "series": "exp3",
        "metrics": {
            "by_clients": {
                "1": {
                    "n_planned": 1,
                    "n_ok": 0,
                    "n_error": 1,
                    "limits_ok": False,
                }
            },
            "n_planned": 1,
            "n_ok": 0,
            "n_error": 1,
        },
    }
    sections = report._exp3_sections(exp3)
    assert "Таблица 4" in sections["exp3_table"]
    assert "НЕТ" in sections["exp3_table"]
    assert "Ошибок вызова: 1 из 1" in sections["exp3_summary"]


def test_missing_data_sections_do_not_raise():
    assert report._exp1_sections({})["exp1_length_table"]
    assert report._exp2_sections({})["exp2_table"]
    assert report._exp3_sections({})["exp3_table"]
