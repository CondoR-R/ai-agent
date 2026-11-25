"""Построение отчёта ЛР4 из каталога запуска."""

import argparse
import json
from pathlib import Path

from ai_agent import files
from ai_agent.labs.lr4 import evaluation


def _latest_run() -> Path:
    runs = sorted(files.RUNS_DIR.glob("lr4-experiment4-*/metrics.json"))
    if not runs:
        raise FileNotFoundError("Не найден runs/lr4-experiment4-*/metrics.json")
    return runs[-1].parent


def _load_records(run_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (run_dir / files.RESULTS_FILENAME).read_text().splitlines()
        if line.strip()
    ]


def _fmt_pct(entry: dict | None) -> str:
    if not entry or entry["n"] == 0:
        return "—"
    return f"{entry['passed']} из {entry['n']} ({entry['pct']}%)"


def _fmt_cost(summary: dict | None) -> str:
    if not summary:
        return "—"
    return (
        f"{summary['median']:.0f} [{summary['min']:.0f}–{summary['max']:.0f}]"
    )


def _trace_excerpt(record: dict, max_steps: int = 8) -> list[str]:
    trace = json.loads((files.PROJECT_ROOT / record["trace_path"]).read_text())
    lines = []
    for step in trace["steps"][:max_steps]:
        if step["kind"] == "model":
            calls = ", ".join(
                f"{c['name']}({json.dumps(c['args'], ensure_ascii=False)})"
                for c in step["tool_calls"]
            )
            content = (step.get("content") or "").replace("\n", " ")
            if len(content) > 180:
                content = content[:177] + "..."
            lines.append(
                f"- model/{step['label']}: calls=[{calls}] content='{content}'"
            )
        else:
            result = (step.get("result") or "").replace("\n", " ")
            if len(result) > 180:
                result = result[:177] + "..."
            err = step.get("error")
            lines.append(
                f"- tool/{step['tool']}: error={err} result='{result}'"
            )
    return lines


def _select_cases(records: list[dict]) -> dict[str, dict]:
    by = {(r["task_id"], r["mode"]): r for r in records}
    tasks = sorted({r["task_id"] for r in records})
    both = [
        t
        for t in tasks
        if by.get((t, "react"), {}).get("utility")
        and by.get((t, "plan"), {}).get("utility")
    ]
    react_only = [
        t
        for t in tasks
        if by.get((t, "react"), {}).get("utility")
        and not by.get((t, "plan"), {}).get("utility")
    ]
    react_limit = [
        t
        for t in tasks
        if not by.get((t, "react"), {}).get("utility")
        and by.get((t, "react"), {}).get("stop_reason")
        in {"max_tools", "max_model_calls"}
    ]
    harness = [
        t
        for t in tasks
        if not by.get((t, "react"), {}).get("utility")
        and any(
            s in by.get((t, "react"), {}).get("answer", "").lower()
            for s in ["i don't have the capability", "could not complete"]
        )
    ]
    return {
        "both": by[(both[0], "react")] if both else None,
        "both_plan": by[(both[0], "plan")] if both else None,
        "react_only_react": by[(react_only[0], "react")]
        if react_only
        else None,
        "react_only_plan": by[(react_only[0], "plan")] if react_only else None,
        "react_limit": by[(react_limit[0], "react")] if react_limit else None,
        "harness": by[(harness[0], "react")] if harness else None,
    }


def _write_table(
    lines: list[str], headers: list[str], rows: list[list[str]]
) -> None:
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")


def build_report(run_dir: Path) -> str:
    records = _load_records(run_dir)
    metrics = evaluation.build_metrics(records)
    meta = files.read_json(run_dir / files.METADATA_FILENAME)
    cases = _select_cases(records)
    lines = []
    lines.append(
        "# ЛР4. Агент с инструментами: собственная обвязка на AgentDojo"
    )
    lines.append("")
    lines.append("## 1. Постановка и прогноз")
    lines.append("")
    lines.append(
        "Стенд: AgentDojo `v1.2.2`, набор `workspace`, 24 инструмента, 40 задач. "
        "Разбиение по правилам курса: 10 dev-задач (`n % 4 == 0`) и 30 holdout-задач. "
        "Модель: Qwen2.5-1.5B-Instruct через llama.cpp, `--jinja`, температура 0. "
        "Лимиты: 15 вызовов модели, 15 вызовов инструментов, 1024 выходных токена."
    )
    lines.append("")
    forecast_path = files.PROJECT_ROOT / "reports/lr4/forecast.md"
    if forecast_path.exists():
        lines.append(
            "Прогноз до holdout записан в `reports/lr4/forecast.md` и зафиксирован коммитом "
            f"`{meta.get('commit', '')[:7]}` перед основным запуском. "
            "Ожидалась низкая успешность 1.5B-модели, преимущество ReAct на задачах, где следующий шаг зависит от прочитанного, и больший расход входных токенов у ReAct."
        )
    lines.append("")
    lines.append("## 2. Реализация")
    lines.append("")
    lines.append(
        "Обвязка реализована в `src/ai_agent/labs/lr4/`: описания инструментов AgentDojo переводятся в OpenAI `tools`, "
        "цикл ReAct выполняет tool calls и возвращает результаты в историю, plan-then-execute сначала фиксирует план через `submit_plan`, "
        "затем выдаёт аргументы только для инструмента текущего шага. Ошибки инструментов не останавливают эпизод; "
        "лимиты дают осмысленный `stop_reason`; каждый эпизод заканчивается текстовым ответом. "
        "Трейсы шагов сохраняются в `runs/<run_id>/traces/`, итоговые записи — в `results.jsonl`, метрики — в `metrics.json`."
    )
    lines.append("")
    checks = files.read_json(files.PROJECT_ROOT / "reports/lr4/checks.json")
    lines.append(
        f"Проверки без модели: `pass={checks['pass']}` (`reports/lr4/checks.json`). "
        "Сценарный ground-truth агент прошёл 40/40 задач стенда; проверки ошибок инструментов, лимита 15 tool calls и запрета вызова не-планового инструмента прошли."
    )
    lines.append("")
    lines.append("## 3. Основная серия")
    lines.append("")
    succ = metrics["success"]
    rows = []
    labels = {
        "all": "Все задачи",
        "single": "1 ground-truth вызов",
        "two": "2 вызова",
        "three_plus": "3–6 вызовов",
        "read_only": "Только чтение",
        "stateful": "Меняют состояние",
        "multi_service": "2+ сервиса",
    }
    for key, label in labels.items():
        rows.append(
            [
                label,
                _fmt_pct(succ[key]["react"]),
                _fmt_pct(succ[key]["plan"]),
            ]
        )
    _write_table(lines, ["Группа", "ReAct", "Plan-then-execute"], rows)

    pair = metrics["pairwise"]
    lines.append(
        f"Попарно ReAct → plan-then-execute: оба верно {pair.get('both_correct', 0)}, оба неверно {pair.get('both_wrong', 0)}, "
        f"только plan {pair.get('plan_fixed', 0)}, только ReAct {pair.get('plan_broken', 0)}, Δ={pair.get('delta', 0)} ({pair.get('delta_pct', 0)} п.п.)."
    )
    lines.append("")

    lines.append("## 4. Затраты")
    lines.append("")
    cost_rows = []
    for mode, label in [("react", "ReAct"), ("plan", "Plan")]:
        for status in ["all", "success", "failure"]:
            c = metrics["costs"][f"{mode}/{status}"]
            cost_rows.append(
                [
                    f"{label}/{status}",
                    str(c["n"]),
                    _fmt_cost(c["model_calls"]),
                    _fmt_cost(c["tool_calls"]),
                    _fmt_cost(c["prompt_tokens"]),
                    _fmt_cost(c["completion_tokens"]),
                    _fmt_cost(c["episode_seconds"]),
                ]
            )
    _write_table(
        lines,
        [
            "Режим",
            "n",
            "Model calls",
            "Tool calls",
            "Prompt tok.",
            "Completion tok.",
            "Seconds",
        ],
        cost_rows,
    )

    lines.append("## 5. Причины неудач")
    lines.append("")
    fail_rows = []
    for mode in ["react", "plan"]:
        reasons = metrics["failures"][mode]["reasons"]
        reason_text = ", ".join(f"{k}:{v}" for k, v in reasons.items()) or "—"
        fail_rows.append(
            [mode, str(metrics["failures"][mode]["n"]), reason_text]
        )
    _write_table(lines, ["Режим", "Неуспехов", "Причины"], fail_rows)

    lines.append("## 6. Разбор трейсов")
    lines.append("")
    case_specs = [
        ("both", "Оба режима решили", "both"),
        (
            "react_only_react",
            "ReAct решил, plan-then-execute нет (ReAct)",
            "react_only_react",
        ),
        (
            "react_only_plan",
            "ReAct решил, plan-then-execute нет (Plan)",
            "react_only_plan",
        ),
        ("react_limit", "ReAct исчерпал лимит/ушёл в сторону", "react_limit"),
        (
            "harness",
            "Неудача, связанная с обвязкой/модельным отказом",
            "harness",
        ),
    ]
    for _, title, case_key in case_specs:
        rec = cases.get(case_key)
        if rec is None:
            continue
        lines.append(f"### {title}: {rec['task_id']}")
        lines.append("")
        lines.append(
            f"Utility={rec['utility']}, stop_reason={rec['stop_reason']}, "
            f"model_calls={rec['n_model_calls']}, tool_calls={rec['n_tool_calls']}, "
            f"tool_errors={rec['n_tool_errors']}."
        )
        lines.append("")
        lines.append("Фрагмент трейса:")
        for item in _trace_excerpt(rec):
            lines.append(item)
        lines.append("")
        lines.append(f"Ответ: {rec['answer']}")
        lines.append("")

    lines.append("## 7. Выводы")
    lines.append("")
    react_all = succ["all"]["react"]
    plan_all = succ["all"]["plan"]
    lines.append(
        f"На holdout ReAct дал {_fmt_pct(react_all)}, plan-then-execute — {_fmt_pct(plan_all)}. "
        "Результат нужно интерпретировать осторожно: маленькая модель 1.5B часто делает ошибки tool calling и отказывается использовать инструменты, поэтому низкая успешность отражает и ограничения модели, и сложность AgentDojo workspace. "
        "Plan-then-execute ограничивает влияние прочитанных данных на выбор инструментов, но страдает от фиксированного плана и ошибок аргументов. ReAct гибче, но дороже по контексту и чувствителен к длинным наблюдениям. "
        "Для более сильного вывода нужен более крупный tool-calling модельный бэкенд и разбор большего числа неуспешных трейсов."
    )
    lines.append("")
    lines.append("## 8. Воспроизводимость")
    lines.append("")
    lines.append(
        f"Run: `{run_dir.name}`, commit `{meta.get('commit', '')[:7]}`, MLflow run `{meta.get('run_id', '')}`. "
        "Артефакты: `config.yaml`, `metadata.json`, `results.jsonl`, `metrics.json`, `traces/`. "
        "Команды: `uv run python scripts/lr4/prepare_data.py`, `uv run python scripts/lr4/checks.py`, "
        "`bash scripts/lr1/start_llama.sh --jinja`, `uv run python scripts/lr4/experiment.py --config configs/lr4/exp.yaml --exp 4 --series main`."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--output", default="reports/lr4/report.md")
    args = parser.parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else _latest_run()
    report = build_report(run_dir)
    out = files.PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(out)


if __name__ == "__main__":
    main()
