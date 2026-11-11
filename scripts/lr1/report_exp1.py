"""Отчёт эксперимента 1: таблицы и графики из сохранённых запусков.

Запускается локально после того, как каталоги запусков скачаны с Colab
в runs/. Всё считается заново из results.jsonl (Правила, п. 4.2, 8):

    python scripts/lr1/report_exp1.py \
        --length-run runs/lr1-experiment1-...-<id> \
        --cache-run  runs/lr1-experiment1-...-<id>
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ai_agent import files  # noqa: E402
from ai_agent.labs.lr1 import exp1_analysis as analysis  # noqa: E402

REPORTS_DIR = files.PROJECT_ROOT / "reports" / "lr1"
FIGURES_DIR = REPORTS_DIR / "figures"


def _load(run_dir: Path) -> dict:
    """Результаты и сведения о запуске; итоги считаются заново."""
    ecfg = files.read_yaml(run_dir / files.CONFIG_FILENAME)["exp1"]
    records = files.read_jsonl(run_dir / files.RESULTS_FILENAME)
    meta = files.read_json(run_dir / files.METADATA_FILENAME)
    if meta["status"] != "completed":
        raise ValueError(f"{run_dir.name}: статус {meta['status']}")
    return {
        "run_id": run_dir.name,
        "series": meta["series"],
        "ecfg": ecfg,
        "records": records,
        "metrics": analysis.summarize_run(ecfg["mode"], records, ecfg),
        "server": {
            "launch_command": meta["server"]["launch_command"],
            "prefix_caching": meta["prefix_caching"],
            "model_revision": meta["model"]["revision"],
            "gpu": meta["device"]["gpu"],
        },
    }


# ---------- таблицы ----------


def _ms(value: float | None) -> str:
    return "—" if value is None else f"{value * 1000:.0f}"


def length_tables(series: dict) -> str:
    metrics, ecfg = series["metrics"], series["ecfg"]
    lines = [
        f"**Таблица 1.** TTFT и скорость prefill по длине запроса. "
        f"Серия `{series['series']}`, run_id `{series['run_id']}`; "
        f"запросов на длину (знаменатель): {ecfg['repeats']}, ответ "
        f"{ecfg['output_tokens']} токенов, кеширование префикса выключено.",
        "",
        "| Длина (цель) | Запрос по usage | Успешно | TTFT медиана, мс "
        "| TTFT мин–макс, мс | Prefill медиана, ток/с | Prefill мин–макс |",
        "|---|---|---|---|---|---|---|",
    ]
    for length, e in metrics["by_length"].items():
        if "ttft_median_s" not in e:
            lines.append(f"| {length} | — | {e['n_ok']}/{e['n_planned']} "
                         "| — | — | — | — |")
            continue
        lines.append(
            f"| {length} | {e['prompt_tokens_median']:.0f} "
            f"| {e['n_ok']}/{e['n_planned']} | {_ms(e['ttft_median_s'])} "
            f"| {_ms(e['ttft_min_s'])}–{_ms(e['ttft_max_s'])} "
            f"| {e['prefill_tok_s_median']:.0f} "
            f"| {e['prefill_tok_s_min']:.0f}–{e['prefill_tok_s_max']:.0f} |"
        )

    lines += [
        "",
        "**Таблица 2.** Рост TTFT между соседними длинами (по медианам). "
        f"Серия `{series['series']}`, run_id `{series['run_id']}`.",
        "",
        "| Пара длин | Рост длины | Рост TTFT | Скорость на добавленных "
        "токенах, ток/с |",
        "|---|---|---|---|",
    ]
    for pair, s in metrics["scaling"].items():
        marginal = s["marginal_prefill_tok_s"]
        lines.append(
            f"| {pair} | ×{s['length_ratio']:.2f} | ×{s['ttft_ratio']:.2f} "
            f"| {'—' if marginal is None else f'{marginal:.0f}'} |"
        )

    fit = metrics["ttft_fit"]
    if fit:
        speed = fit["prefill_tok_s"]
        lines += [
            "",
            f"Прямая по медианам: TTFT = {int(round(fit['intercept_s'] * 1000))} мс"
            f" + {fit['slope_s_per_token'] * 1000:.4f} мс × токены"
            + (f" (скорость prefill {speed:.0f} ток/с)" if speed else "")
            + ".",
        ]
    return "\n".join(lines)


def cache_tables(series: dict) -> str:
    metrics, ecfg = series["metrics"], series["ecfg"]
    lines = [
        f"**Таблица 3.** Один и тот же запрос на {ecfg['prompt_tokens']} "
        f"токенов, повторённый {ecfg['repeats']} раза подряд. "
        f"Серия `{series['series']}`, run_id `{series['run_id']}`; "
        "кеширование префикса включено.",
        "",
        "| Запрос | Успешно | TTFT 1-й, мс | TTFT 2-й, мс | TTFT 3-й, мс "
        "| Ускорение 1/2 | Ускорение 1/3 | Токенов из кеша (1/2/3) "
        "| 1-й был холодным |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for number, e in metrics["prompts"].items():
        ttft = e["ttft_s"] + [None] * (3 - len(e["ttft_s"]))
        hits = e["hit_tokens"] + [None] * (3 - len(e["hit_tokens"]))
        cold = e["cold_start_ok"]
        lines.append(
            f"| {number} | {e['n_ok']}/{e['n_planned']} "
            f"| {_ms(ttft[0])} | {_ms(ttft[1])} | {_ms(ttft[2])} "
            f"| {_fmt(e.get('speedup_2'))} | {_fmt(e.get('speedup_3'))} "
            f"| {' / '.join('—' if h is None else f'{h:.0f}' for h in hits)} "
            f"| {'—' if cold is None else ('да' if cold else 'НЕТ')} |"
        )
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"×{value:.2f}"


# ---------- графики ----------


def plot_ttft(series: dict, path: Path) -> None:
    """TTFT от длины запроса: медиана и диапазон, прямая по медианам."""
    metrics = series["metrics"]
    points = [
        e for e in metrics["by_length"].values() if "ttft_median_s" in e
    ]
    xs = np.array([e["prompt_tokens_median"] for e in points])
    med = np.array([e["ttft_median_s"] for e in points])
    low = np.array([e["ttft_min_s"] for e in points])
    high = np.array([e["ttft_max_s"] for e in points])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.errorbar(
        xs, med, yerr=[med - low, high - med], marker="o", capsize=3,
        label=f"медиана и диапазон (запросов на длину: {series['ecfg']['repeats']})",
    )
    fit = metrics["ttft_fit"]
    if fit:
        line = np.linspace(xs.min(), xs.max(), 50)
        ax.plot(
            line, fit["intercept_s"] + fit["slope_s_per_token"] * line,
            "--", color="gray", label="прямая по медианам",
        )
    ax.set_xlabel("Длина запроса, токенов")
    ax.set_ylabel("TTFT, с")
    ax.set_title("Время до первого токена от длины запроса")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_prefill_speed(series: dict, path: Path) -> None:
    """Скорость prefill (токенов запроса в секунду TTFT) по длине."""
    points = [
        e for e in series["metrics"]["by_length"].values()
        if "ttft_median_s" in e
    ]
    xs = np.array([e["prompt_tokens_median"] for e in points])
    med = np.array([e["prefill_tok_s_median"] for e in points])
    low = np.array([e["prefill_tok_s_min"] for e in points])
    high = np.array([e["prefill_tok_s_max"] for e in points])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.errorbar(
        xs, med, yerr=[med - low, high - med], marker="o", capsize=3,
        label="медиана и диапазон",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Длина запроса, токенов (лог. шкала)")
    ax.set_ylabel("Скорость prefill, токенов/с")
    ax.set_title("Скорость prefill = токены запроса / TTFT")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cache(series: dict, path: Path) -> None:
    """TTFT повторных запросов с кешированием префикса."""
    prompts = series["metrics"]["prompts"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    width = 0.8 / max(len(prompts), 1)
    for index, (number, e) in enumerate(prompts.items()):
        ttft = [0 if v is None else v for v in e["ttft_s"]]
        positions = np.arange(len(ttft)) + index * width
        ax.bar(positions, ttft, width=width, label=f"запрос {number}")
    ax.set_xticks(np.arange(series["ecfg"]["repeats"]) + 0.4 - width / 2)
    ax.set_xticklabels(
        [f"{i}-й раз" for i in range(1, series["ecfg"]["repeats"] + 1)]
    )
    ax.set_ylabel("TTFT, с")
    ax.set_title(
        f"Запрос на {series['ecfg']['prompt_tokens']} токенов, "
        "кеширование префикса включено"
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------- MLflow ----------


def log_to_mlflow(paths: list[Path], run_ids: dict[str, str]) -> None:
    """Сохраняет рисунки и таблицы отчёта в MLflow (Правила, п. 4.4).

    Отдельный эксперимент lr1_report: артефакты Colab-запусков лежат по
    путям Colab, а отчёт собирается локально.
    """
    import mlflow

    from ai_agent import tracking

    tracking.setup("lr1_report")
    with mlflow.start_run(run_name="exp1-report"):
        mlflow.set_tags(
            {"work": "lr1", "experiment": "1",
             **{f"source_{k}": v for k, v in run_ids.items()}}
        )
        for path in paths:
            mlflow.log_artifact(str(path))


# ---------- main ----------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length-run", type=Path, default=None)
    parser.add_argument("--cache-run", type=Path, default=None)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    if args.length_run is None and args.cache_run is None:
        parser.error("нужен хотя бы один из --length-run, --cache-run")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {}
    sections: list[str] = []
    outputs: list[Path] = []
    run_ids: dict[str, str] = {}

    if args.length_run:
        series = _load(args.length_run)
        sections.append(length_tables(series))
        for name, plot in (
            ("exp1_ttft_vs_length.png", plot_ttft),
            ("exp1_prefill_speed.png", plot_prefill_speed),
        ):
            plot(series, FIGURES_DIR / name)
            outputs.append(FIGURES_DIR / name)
        report["ttft_vs_length"] = _report_part(series)
        run_ids["length"] = series["run_id"]

    if args.cache_run:
        series = _load(args.cache_run)
        sections.append(cache_tables(series))
        plot_cache(series, FIGURES_DIR / "exp1_prefix_cache.png")
        outputs.append(FIGURES_DIR / "exp1_prefix_cache.png")
        report["prefix_cache"] = _report_part(series)
        run_ids["cache"] = series["run_id"]

    tables_path = REPORTS_DIR / "experiment_1_tables.md"
    tables_path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    json_path = REPORTS_DIR / "experiment_1.json"
    files.write_json(json_path, report)
    outputs += [tables_path, json_path]

    print("\n\n".join(sections))
    print(f"\n[OK] {json_path.relative_to(files.PROJECT_ROOT)}")
    for path in outputs:
        print(f"     {path.relative_to(files.PROJECT_ROOT)}")

    if not args.no_mlflow:
        log_to_mlflow(outputs, run_ids)
        print("[OK] файлы сохранены в MLflow (эксперимент lr1_report)")


def _report_part(series: dict) -> dict:
    return {
        "run_id": series["run_id"],
        "series": series["series"],
        "config": series["ecfg"],
        "server": series["server"],
        "metrics": series["metrics"],
    }


if __name__ == "__main__":
    main()
