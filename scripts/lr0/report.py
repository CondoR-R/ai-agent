"""Сборка Markdown-отчёта ЛР0 из сохранённых артефактов."""

import argparse
import re
from datetime import datetime
from pathlib import Path

from ai_agent import files, utils

REPORTS_DIR = files.PROJECT_ROOT / "reports" / "lr0"
DEFAULT_TEMPLATE = REPORTS_DIR / "report_template.md"
DEFAULT_OUTPUT = REPORTS_DIR / "report.md"

MANUAL_RE = re.compile(
    r"<!-- manual:([\w-]+) -->(.*?)<!-- /manual:\1 -->", re.S
)
PLACEHOLDER_RE = re.compile(r"\{\{\s*([\w-]+)\s*\}\}")


def _load_json(name: str) -> dict:
    """Читает JSON из reports/lr0 или возвращает пустой словарь."""
    try:
        return files.read_json(REPORTS_DIR / name)
    except FileNotFoundError:
        return {}


def _table(headers: list[str], rows: list[list]) -> str:
    """Собирает Markdown-таблицу."""
    if not rows:
        return "_нет данных_"
    lines = ["| " + " | ".join(map(str, headers)) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def _acc(value: float) -> str:
    return f"{value:.4f}"


def _pp(value: float) -> str:
    return f"{value * 100:+.2f}"


def _sec(value: float) -> str:
    return f"{value:.2f}"


def _num(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _get(cfg: dict, path: tuple[str, ...]):
    node = cfg
    for key in path:
        node = node[key]
    return node


def _report_meta() -> str:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    commit = utils.get_commit_hash()[:7]
    return (
        f"_Отчёт сформирован автоматически {stamp}, коммит `{commit}`. "
        "Числа берутся из `reports/lr0/*.json`; блоки `manual` предназначены "
        "для ручного текста и сохраняются при повторной сборке._"
    )


def _data_table() -> str:
    try:
        manifest = files.read_json(files.DATA_DIR / "manifest.json")["lr0"]
    except (FileNotFoundError, KeyError):
        return "_Нет данных манифеста._"

    rule = manifest["rule"]
    sizes = manifest["sizes"]
    rows = [
        ["Источник", f"{manifest['source']} {manifest['version']}"],
        [
            "Правило",
            f"test {rule['test_size'] * 100:.0f} %, val "
            f"{rule['val_size'] * 100:.0f} % остатка, "
            f"split_seed={rule['split_seed']}, stratify={rule['stratify']}",
        ],
        [
            "Размеры",
            f"train={sizes['train']}, val={sizes['val']}, "
            f"test={sizes['test']}",
        ],
        ["sha256 split.json", f"`{manifest['sha256'][:16]}…`"],
    ]
    return _table(["Поле", "Значение"], rows)


def _configs_table() -> str:
    try:
        base = files.read_yaml(
            files.PROJECT_ROOT / "configs/lr0/base.yaml"
        )
        best = files.read_yaml(
            files.PROJECT_ROOT / "configs/lr0/best.yaml"
        )
    except FileNotFoundError:
        return "_Конфигурации не найдены._"

    params = [
        ("seed", ("seed",)),
        ("n_layers", ("model", "n_layers")),
        ("hidden", ("model", "hidden")),
        ("dropout", ("model", "dropout")),
        ("lr", ("train", "lr")),
        ("weight_decay", ("train", "weight_decay")),
        ("batch_size", ("train", "batch_size")),
        ("epochs", ("train", "epochs")),
    ]
    rows = [
        [name, _num(_get(base, path)), _num(_get(best, path))]
        for name, path in params
    ]
    return _table(["Параметр", "base", "best"], rows)


def _exp1_sections(exp1: dict) -> dict:
    task1 = exp1.get("task_1", {})
    task2 = exp1.get("task_2", {})
    task3 = exp1.get("task_3", {})

    if task1.get("identical"):
        run_ids = ", ".join(
            f"`{run_id}`" for run_id in task1.get("run_ids", [])
        )
        repeatability = (
            "Повторяемость: **да**. Два запуска с `seed = 0` дали одинаковые "
            f"`history.csv` и `predictions.csv`: {run_ids}."
        )
    else:
        repeatability = "_Проверка повторяемости не зафиксирована._"

    seeds = task2.get("seeds", {})
    rows = []
    for seed in sorted(seeds, key=lambda item: int(item)):
        item = seeds[seed]
        rows.append(
            [
                seed,
                f"`{item.get('run_id', '—')}`",
                _acc(item["val_acc"]),
                _sec(item["train_time"]),
            ]
        )
    seeds_table = _table(
        ["seed", "run_id", "val_acc", "train_time, с"], rows
    )

    summary = task2.get("summary", {})
    if summary:
        spread = (summary["max_acc"] - summary["min_acc"]) * 100
        summary_text = (
            f"val_acc по seed 1–5: min = **{_acc(summary['min_acc'])}**, "
            f"median = **{_acc(summary['median_acc'])}**, "
            f"max = **{_acc(summary['max_acc'])}**; "
            f"разброс = **{spread:.2f} п.п.** "
            f"(одно изображение val = {100 / 360:.2f} п.п.)."
        )
    else:
        summary_text = "_Сводка по seed отсутствует._"

    counts = task3.get("counts", {})
    if counts:
        n_val = task3.get("n_val") or sum(counts.values())
        comparison_table = _table(
            ["", "второй верно", "второй ошибся"],
            [
                [
                    "первый верно",
                    counts.get("both_correct", 0),
                    counts.get("only_first_correct", 0),
                ],
                [
                    "первый ошибся",
                    counts.get("only_second_correct", 0),
                    counts.get("both_wrong", 0),
                ],
            ],
        )
        comparison = (
            f"Сравнение предсказаний двух seed на val (N = {n_val}):\n\n"
            f"{comparison_table}\n\n"
            "Первый — `seed 0`, второй — `seed 1`. "
            f"Точность первого = **{_acc(task3.get('first_acc', 0.0))}**, "
            f"второго = **{_acc(task3.get('second_acc', 0.0))}**, "
            f"Δ = **{_pp(task3.get('delta_acc', 0.0))} п.п.** "
            f"Второй исправил **{counts.get('only_second_correct', 0)}** "
            f"и испортил **{counts.get('only_first_correct', 0)}**."
        )
        run_ids = task3.get("run_ids", [])
        if run_ids:
            runs_text = ", ".join(f"`{run_id}`" for run_id in run_ids)
            comparison += f"\n\nЗапуски: {runs_text}."
    else:
        comparison = "_Сравнение предсказаний двух seed отсутствует._"

    examples = task3.get("examples", [])
    example_rows = []
    for item in examples[:10]:
        y_true = item["y_true"]
        first_pred = item["first_pred"]
        second_pred = item["second_pred"]
        if first_pred == y_true:
            who = "первый"
        elif second_pred == y_true:
            who = "второй"
        else:
            who = "оба ошиблись"
        example_rows.append(
            [item["index"], y_true, first_pred, second_pred, who]
        )
    examples_text = ""
    if example_rows:
        examples_text = "**Примеры расхождений.**\n\n" + _table(
            ["index", "y_true", "pred первого", "pred второго", "кто прав"],
            example_rows,
        )

    return {
        "exp1_repeatability": repeatability,
        "exp1_seeds_table": seeds_table,
        "exp1_summary": summary_text,
        "exp1_comparison": comparison,
        "exp1_examples": examples_text,
    }


def _exp2_sections(exp2: dict) -> dict:
    if not exp2:
        missing = "_Нет данных: запустите `experiment_2.py`._"
        return {
            "exp2_states_table": missing,
            "exp2_epochs": missing,
            "exp2_time": missing,
            "exp2_best_table": missing,
            "exp2_importance_table": missing,
            "exp2_plots": missing,
        }

    states = exp2.get("states", {})
    state_rows = [
        [name, count] for name, count in states.items() if count
    ]
    states_table = _table(["Состояние", "Trial"], state_rows)

    epochs = exp2.get("epochs", {})
    time_stats = exp2.get("time", {})
    executed = epochs.get("executed", 0)
    planned = epochs.get("planned_40", 0)
    planned_actual = epochs.get("planned_actual", 0)
    saved = epochs.get("pruner_saved", 0)
    pruned = states.get("PRUNED", 0)

    epochs_text = (
        f"Выполнено **{executed}** эпох из **{planned}** (40 × 30) — "
        f"**{epochs.get('executed_share_40', 0.0) * 100:.1f} %**. "
        f"Прунер остановил **{pruned}** trial и сэкономил **{saved}** эпох "
        f"(**{epochs.get('pruner_saved_share_40', 0.0) * 100:.1f} %** плана)."
    )
    if planned_actual and planned_actual != planned:
        epochs_text += (
            f"\n\nВ study фактически **{exp2.get('n_trials', 0)}** trial, "
            f"поэтому фактический план — **{planned_actual}** эпох; "
            f"выполнено {executed}."
        )

    avg = time_stats.get("avg_epoch_seconds", 0.0)
    if avg:
        time_text = (
            f"Среднее время эпохи ≈ **{avg:.3f} с**. Оценка экономии "
            f"прунера: **{time_stats.get('pruner_saved_seconds', 0.0):.1f} с** "
            f"({saved} эпох × среднее время эпохи). Фактическое время поиска ≈ "
            f"**{time_stats.get('search_seconds', 0.0):.1f} с**; оценка "
            "полного прогона без прунера ≈ "
            f"**{time_stats.get('full_estimate_seconds', 0.0):.1f} с**."
        )
    else:
        time_text = "_Время эпохи не определено._"

    best = exp2.get("best_trial")
    if best:
        rows = [
            ["trial", f"#{best['number']}"],
            ["val_acc", _acc(best["value"])],
        ]
        for name, value in best.get("params", {}).items():
            rows.append([name, _num(value)])
        best_table = _table(["Параметр", "Значение"], rows)
    else:
        best_table = "_Лучший trial не найден._"

    importances = exp2.get("importances", {})
    error = exp2.get("importance_error")
    if importances:
        rows = [
            [name, f"{value:.4f}", f"{value * 100:.1f} %"]
            for name, value in sorted(
                importances.items(), key=lambda item: item[1], reverse=True
            )
        ]
        importance_table = _table(
            ["Параметр", "Важность", "Доля"], rows
        )
        if error:
            importance_table += f"\n\n_Предупреждение: {error}_"
    else:
        importance_table = "_Важность не посчитана"
        if error:
            importance_table += f": {error}"
        importance_table += "._"

    plots = exp2.get("plots", {})
    plot_parts = []
    for key, label in (
        ("trials", "Номер trial — val accuracy"),
        ("importance", "Важность параметров"),
    ):
        name = plots.get(key)
        if name and (REPORTS_DIR / name).exists():
            plot_parts.append(f"![{label}]({name})")
    plot_text = (
        "\n\n".join(plot_parts)
        if plot_parts
        else "_Графики не построены._"
    )
    if exp2.get("plot_errors"):
        plot_text += (
            "\n\n_Ошибки графиков: "
            + "; ".join(exp2["plot_errors"])
            + "_"
        )

    return {
        "exp2_states_table": states_table,
        "exp2_epochs": epochs_text,
        "exp2_time": time_text,
        "exp2_best_table": best_table,
        "exp2_importance_table": importance_table,
        "exp2_plots": plot_text,
    }


def _exp3_sections(exp3: dict, exp1: dict, exp2: dict) -> dict:
    if not exp3:
        missing = "_Нет данных: запустите `experiment_3.py`._"
        return {
            "exp3_table": missing,
            "exp3_summary": missing,
            "exp3_three_numbers": missing,
        }

    seeds = exp3.get("seeds", {})
    rows = []
    for seed, item in sorted(seeds.items(), key=lambda pair: int(pair[0])):
        rows.append(
            [
                seed,
                _acc(item["base"]["val_acc"]),
                _acc(item["best"]["val_acc"]),
                _pp(item["delta_val_acc"]),
                _acc(item["base"]["test_acc"]),
                _acc(item["best"]["test_acc"]),
                _pp(item["delta_test_acc"]),
                item["test_counts"]["corrected"],
                item["test_counts"]["spoiled"],
            ]
        )
    table = _table(
        [
            "seed",
            "base val",
            "best val",
            "Δ val, п.п.",
            "base test",
            "best test",
            "Δ test, п.п.",
            "corrected",
            "spoiled",
        ],
        rows,
    )
    table = (
        f"Серия `{exp3.get('series', 'final')}`; run_id по каждому seed — "
        f"в `experiment_3.json`.\n\n{table}"
    )

    summary = exp3.get("summary", {})
    median_val = _pp(summary.get("median_delta_val_acc", 0.0))
    min_val = _pp(summary.get("min_delta_val_acc", 0.0))
    max_val = _pp(summary.get("max_delta_val_acc", 0.0))
    median_test = _pp(summary.get("median_delta_test_acc", 0.0))
    min_test = _pp(summary.get("min_delta_test_acc", 0.0))
    max_test = _pp(summary.get("max_delta_test_acc", 0.0))
    summary_text = (
        f"Δ val: медиана **{median_val} п.п.**, "
        f"диапазон **{min_val} … {max_val} п.п.**. "
        f"Δ test: медиана **{median_test}**, "
        f"диапазон **{min_test} … {max_test}**.\n\n"
        f"Исправлено **{summary.get('corrected', 0)}**, испорчено "
        f"**{summary.get('spoiled', 0)}** из **{summary.get('n_test', 0)}** "
        f"(5 × 360), net = **{summary.get('net', 0):+d}**; "
        f"одно изображение test = {100 / 360:.2f} п.п."
    )

    best_trial = exp2.get("best_trial") or {}
    best_values = [
        item["best"]["val_acc"] for item in seeds.values()
    ]
    best_tests = [
        item["best"]["test_acc"] for item in seeds.values()
    ]
    mean_val = sum(best_values) / len(best_values) if best_values else None
    mean_test = sum(best_tests) / len(best_tests) if best_tests else None
    best_trial_value = best_trial.get("value")

    rows = [
        [
            "Лучший trial на val",
            _acc(best_trial_value) if best_trial_value is not None else "—",
        ],
        [
            "Средняя val лучшей конфигурации (seed 101–105)",
            _acc(mean_val) if mean_val is not None else "—",
        ],
        [
            "Средняя test лучшей конфигурации",
            _acc(mean_test) if mean_test is not None else "—",
        ],
    ]
    three_numbers = _table(["Число", "Значение"], rows)

    if best_trial_value is not None and mean_val is not None:
        three_numbers += (
            f"\n\nПадение «лучший trial → новые seed» = "
            f"**{(mean_val - best_trial_value) * 100:+.2f} п.п.**"
        )
        if mean_test is not None:
            three_numbers += (
                f"; «новые seed val → test» = "
                f"**{(mean_test - mean_val) * 100:+.2f} п.п.**."
            )

    spread = exp1.get("task_2", {}).get("summary", {})
    min_delta_test = summary.get("min_delta_test_acc")
    if spread and min_delta_test is not None:
        spread_pp = (spread["max_acc"] - spread["min_acc"]) * 100
        three_numbers += (
            f"\n\nМинимальный Δ test = **{min_delta_test * 100:+.2f} п.п.**; "
            f"разброс base по seed = **{spread_pp:.2f} п.п.**"
        )

    return {
        "exp3_table": table,
        "exp3_summary": summary_text,
        "exp3_three_numbers": three_numbers,
    }


def _merge_manual(template_text: str, previous_text: str) -> str:
    """Переносит ручные блоки из предыдущего отчёта в новый шаблон."""
    if not previous_text:
        return template_text

    old_blocks = {
        match.group(1): match.group(2)
        for match in MANUAL_RE.finditer(previous_text)
    }

    def replace(match: re.Match) -> str:
        key = match.group(1)
        content = old_blocks.get(key, match.group(2))
        return (
            f"<!-- manual:{key} -->{content}<!-- /manual:{key} -->"
        )

    return MANUAL_RE.sub(replace, template_text)


def _render(template_text: str, values: dict) -> str:
    """Подставляет значения вместо {{ placeholder }}."""

    def replace(match: re.Match) -> str:
        return str(values.get(match.group(1), match.group(0)))

    return PLACEHOLDER_RE.sub(replace, template_text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    template_path = Path(args.template)
    output_path = Path(args.output)
    if not template_path.exists():
        raise SystemExit(f"Шаблон не найден: {template_path}")

    exp1 = _load_json("experiment_1.json")
    exp2 = _load_json("experiment_2.json")
    exp3 = _load_json("experiment_3.json")

    values = {
        "report_meta": _report_meta(),
        "data_table": _data_table(),
        "configs_table": _configs_table(),
    }
    values.update(_exp1_sections(exp1))
    values.update(_exp2_sections(exp2))
    values.update(_exp3_sections(exp3, exp1, exp2))

    template_text = template_path.read_text(encoding="utf-8")
    if output_path.exists():
        previous = output_path.read_text(encoding="utf-8")
        template_text = _merge_manual(template_text, previous)

    report_text = _render(template_text, values)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")

    leftovers = sorted(set(PLACEHOLDER_RE.findall(report_text)))
    print(f"[REPORT] отчёт: {output_path}", flush=True)
    if leftovers:
        print(f"[REPORT] не заполнено: {leftovers}", flush=True)


if __name__ == "__main__":
    main()
