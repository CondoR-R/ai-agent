"""Отчёт эксперимента 2: таблицы и графики из сохранённого запуска.

Запускается локально после того, как каталог запуска скачан с Colab
в runs/. Всё считается заново из results.jsonl (Правила, п. 4.2, 8):

    python scripts/lr1/report_exp2.py \
        --run runs/lr1-experiment2-...-<id>
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ai_agent import files
from ai_agent.labs.lr1 import exp2_analysis as analysis

REPORTS_DIR = files.PROJECT_ROOT / "reports" / "lr1"
FIGURES_DIR = REPORTS_DIR / "figures"


def _load(run_dir: Path) -> dict:
    """Результаты и сведения о запуске; итоги считаются заново."""
    cfg = files.read_yaml(run_dir / files.CONFIG_FILENAME)
    records = files.read_jsonl(run_dir / files.RESULTS_FILENAME)
    meta = files.read_json(run_dir / files.METADATA_FILENAME)
    if meta["status"] != "completed":
        raise ValueError(f"{run_dir.name}: статус {meta['status']}")
    return {
        "run_id": run_dir.name,
        "series": meta["series"],
        "cfg": cfg,
        "metrics": analysis.summarize_run(records, cfg),
        "server": {
            "launch_command": meta["server"]["launch_command"],
            "model_revision": meta["model"]["revision"],
            "gpu": meta["device"]["gpu"],
        },
    }


# ---------- таблицы ----------


def _ms(value: float | None) -> str:
    return "—" if value is None else f"{value * 1000:.0f}"


def _speed(value: float | None) -> str:
    return "—" if value is None else f"{value:.0f}"


def _number(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def output_tables(series: dict) -> str:
    metrics, ecfg = series["metrics"], series["cfg"]["exp2"]
    lines = [
        f"**Таблица 1.** Полное время ответа и скорость генерации по длине "
        f"ответа. Серия `{series['series']}`, run_id `{series['run_id']}`; "
        f"запросов на длину (знаменатель): {ecfg['repeats']}, запрос "
        f"{ecfg['prompt_tokens']} токенов, ignore_eos.",
        "",
        "| Длина ответа | Ответ по usage | Успешно | e2e медиана, мс "
        "| e2e мин–макс, мс | TTFT медиана, мс | TPOT медиана, мс "
        "| Decode, ток/с |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for length, e in metrics["by_output"].items():
        if "e2e_median_s" not in e:
            lines.append(
                f"| {length} | — | {e['n_ok']}/{e['n_planned']} "
                "| — | — | — | — | — |"
            )
            continue
        lines.append(
            f"| {length} | {e['completion_tokens_median']:.0f} "
            f"| {e['n_ok']}/{e['n_planned']} "
            f"| {_ms(e['e2e_median_s'])} "
            f"| {_ms(e['e2e_min_s'])}–{_ms(e['e2e_max_s'])} "
            f"| {_ms(e['ttft_median_s'])} "
            f"| {_ms(e['tpot_median_s'])} "
            f"| {_speed(e['decode_tok_s_median'])} |"
        )

    fit = metrics["e2e_fit"]
    lower = metrics["decode_lower_bound_s"]
    if fit:
        lines += [
            "",
            "**Таблица 2.** Прямая e2e = a + b × N по медианам "
            f"(N — токены ответа). Серия `{series['series']}`, "
            f"run_id `{series['run_id']}`.",
            "",
            "| Величина | Прямая | Опорное значение | Разница |",
            "|---|---|---|---|",
            f"| Свободный член a (TTFT) | {_ms(fit['intercept_s'])} мс "
            f"| {_ms(fit['ttft_ref_s'])} мс "
            f"| {_ms(fit['intercept_minus_ttft_s'])} мс |",
            f"| Наклон b (TPOT) | {_ms(fit['slope_s_per_token'])} мс/токен "
            f"| {_ms(fit['tpot_ref_s'])} мс "
            f"| {_ms(fit['slope_minus_tpot_s'])} мс |",
            f"| Нижняя граница TPOT (память весов) | {_ms(lower)} мс "
            f"| медиана TPOT {_ms(metrics['tpot_median_s'])} мс "
            f"| превышение ×{_number(metrics['tpot_over_lower_bound'], 2)} |",
        ]
    return "\n".join(lines)


# ---------- графики ----------


def plot_e2e(series: dict, path: Path) -> None:
    """Полное время ответа от длины ответа с прямой по медианам."""
    metrics = series["metrics"]
    points = [e for e in metrics["by_output"].values() if "e2e_median_s" in e]
    xs = np.array([e["completion_tokens_median"] for e in points])
    med = np.array([e["e2e_median_s"] for e in points])
    low = np.array([e["e2e_min_s"] for e in points])
    high = np.array([e["e2e_max_s"] for e in points])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.errorbar(
        xs,
        med,
        yerr=[med - low, high - med],
        marker="o",
        capsize=3,
        label=(
            "медиана и диапазон "
            f"(запросов на длину: {series['cfg']['exp2']['repeats']})"
        ),
    )
    fit = metrics["e2e_fit"]
    if fit:
        line = np.linspace(xs.min(), xs.max(), 50)
        ax.plot(
            line,
            fit["intercept_s"] + fit["slope_s_per_token"] * line,
            "--",
            color="gray",
            label="прямая по медианам",
        )
    ax.set_xlabel("Длина ответа, токенов")
    ax.set_ylabel("Полное время ответа, с")
    ax.set_title("Полное время ответа от длины ответа")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_tpot(series: dict, path: Path) -> None:
    """TPOT от длины ответа и нижняя граница по памяти весов."""
    metrics = series["metrics"]
    points = [
        e
        for e in metrics["by_output"].values()
        if e.get("tpot_median_s") is not None
    ]
    xs = np.array([e["completion_tokens_median"] for e in points])
    med = np.array([e["tpot_median_s"] for e in points])
    low = np.array([e["tpot_min_s"] for e in points])
    high = np.array([e["tpot_max_s"] for e in points])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.errorbar(
        xs,
        med,
        yerr=[med - low, high - med],
        marker="o",
        capsize=3,
        label="медиана и диапазон",
    )
    lower = metrics["decode_lower_bound_s"]
    if lower:
        ax.axhline(
            lower,
            linestyle="--",
            color="gray",
            label="нижняя граница: веса / память",
        )
    ax.set_xlabel("Длина ответа, токенов")
    ax.set_ylabel("TPOT, с")
    ax.set_title("Время следующего токена от длины ответа")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------- MLflow ----------


def log_to_mlflow(paths: list[Path], run_id: str) -> None:
    """Сохраняет рисунки и таблицы отчёта в MLflow (Правила, п. 4.4)."""
    import mlflow

    from ai_agent import tracking

    tracking.setup("lr1_report")
    with mlflow.start_run(run_name="exp2-report"):
        mlflow.set_tags(
            {"work": "lr1", "experiment": "2", "source_run_id": run_id}
        )
        for path in paths:
            mlflow.log_artifact(str(path))


# ---------- main ----------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    series = _load(args.run)

    tables = output_tables(series)
    tables_path = REPORTS_DIR / "experiment_2_tables.md"
    tables_path.write_text(tables + "\n", encoding="utf-8")

    json_path = REPORTS_DIR / "experiment_2.json"
    files.write_json(
        json_path,
        {
            "run_id": series["run_id"],
            "series": series["series"],
            "config": series["cfg"],
            "server": series["server"],
            "metrics": series["metrics"],
        },
    )

    outputs = [tables_path, json_path]
    for name, plot in (
        ("exp2_e2e_vs_output.png", plot_e2e),
        ("exp2_tpot_vs_output.png", plot_tpot),
    ):
        plot(series, FIGURES_DIR / name)
        outputs.append(FIGURES_DIR / name)

    print(tables)
    print(f"\n[OK] {json_path.relative_to(files.PROJECT_ROOT)}")
    for path in outputs:
        print(f"     {path.relative_to(files.PROJECT_ROOT)}")

    if not args.no_mlflow:
        log_to_mlflow(outputs, series["run_id"])
        print("[OK] файлы сохранены в MLflow (эксперимент lr1_report)")


if __name__ == "__main__":
    main()
