"""Отчёт эксперимента 3: таблица уровней и график «пропускная — задержка».

Запускается локально после выполнения эксперимента. Всё считается
заново из results.jsonl (Правила, п. 4.2, 8):

    python scripts/lr1/report_exp3.py \
        --run runs/lr1-experiment3-...-<id>
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ai_agent import files
from ai_agent.labs.lr1 import exp3_analysis as analysis

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
        "metrics": analysis.summarize_level_run(
            records, cfg, meta["kv_cache"]["kv_cache_tokens"]
        ),
        "server": {
            "launch_command": meta["server"]["launch_command"],
            "model_revision": meta["model"]["revision"],
            "gpu": meta["device"]["gpu"],
        },
    }


# ---------- таблицы ----------


def _ms(value: float | None) -> str:
    return "—" if value is None else f"{value * 1000:.0f}"


def _num(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def level_tables(series: dict) -> str:
    metrics, ecfg = series["metrics"], series["cfg"]["exp3"]
    limits = metrics["limits"]
    lines = [
        f"**Таблица 1.** Уровни нагрузки. Серия `{series['series']}`, "
        f"run_id `{series['run_id']}`; запрос {ecfg['prompt_tokens']} "
        f"токенов, ответ {ecfg['output_tokens']} токенов, ignore_eos; "
        f"пределы TTFT p90 ≤ {limits['ttft_p90_s'] * 1000:.0f} мс, "
        f"медиана TPOT ≤ {limits['tpot_median_s'] * 1000:.0f} мс.",
        "",
        "| Клиентов | Запросов | Успешно | Ошибок | Пропускная, ток/с "
        "| Запросов/с | TTFT медиана, мс | TTFT p90, мс | TPOT медиана, мс "
        "| e2e медиана, мс | Goodput, зап/с | Пределы | Очередь после "
        "| KV после |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for clients, e in metrics["by_clients"].items():
        lines.append(
            f"| {clients} | {e['n_planned']} "
            f"| {e['n_ok']}/{e['n_planned']} | {e['n_error']} "
            f"| {_num(e.get('throughput_tok_s'), 0)} "
            f"| {_num(e.get('requests_per_s'), 2)} "
            f"| {_ms(e.get('ttft_median_s'))} "
            f"| {_ms(e.get('ttft_p90_s'))} "
            f"| {_ms(e.get('tpot_median_s'))} "
            f"| {_ms(e.get('e2e_median_s'))} "
            f"| {_num(e.get('goodput_rps'), 2)} "
            f"| {'да' if e['limits_ok'] else 'НЕТ'} "
            f"| {_num(e.get('queue_waiting_after'), 1)} "
            f"| {_num(e.get('kv_cache_usage_after'), 2)} |"
        )

    max_load = metrics["max_clients_within_limits"]
    capacity = metrics["kv_requests_capacity"]
    kv_tokens = metrics["kv_cache_tokens"]
    max_load_text = "—" if max_load is None else f"{max_load} клиентов"
    kv_tokens_text = "—" if kv_tokens is None else f"{kv_tokens} токенов"
    capacity_text = "—" if capacity is None else str(capacity)
    lines += [
        "",
        "**Таблица 2.** Пределы и ёмкость KV-кеша.",
        "",
        "| Величина | Значение |",
        "|---|---|",
        f"| Наибольшая нагрузка в пределах | {max_load_text} |",
        f"| Ёмкость KV-кеша | {kv_tokens_text} |",
        f"| Запросов по {ecfg['prompt_tokens']} + {ecfg['output_tokens']} "
        f"токенов в кеше | {capacity_text} |",
        f"| Ошибок вызова | {metrics['n_error']} из {metrics['n_planned']} |",
    ]
    return "\n".join(lines)


# ---------- графики ----------


def plot_throughput_latency(series: dict, path: Path) -> None:
    """Пропускная способность от TTFT p90; подписи — число клиентов."""
    levels = [
        (int(clients), e)
        for clients, e in series["metrics"]["by_clients"].items()
        if e.get("throughput_tok_s") is not None
        and e.get("ttft_p90_s") is not None
    ]
    xs = np.array([e["ttft_p90_s"] * 1000 for _, e in levels])
    ys = np.array([e["throughput_tok_s"] for _, e in levels])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(xs, ys, marker="o")
    for clients, x, y in zip((c for c, _ in levels), xs, ys, strict=True):
        ax.annotate(
            f"{clients} кл.",
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    limit = series["metrics"]["limits"]["ttft_p90_s"] * 1000
    ax.axvline(
        limit,
        linestyle="--",
        color="gray",
        label=f"предел TTFT p90 = {limit:.0f} мс",
    )
    ax.set_xlabel("TTFT p90, мс")
    ax.set_ylabel("Пропускная способность, токенов/с")
    ax.set_title("Пропускная способность — задержка")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_throughput_clients(series: dict, path: Path) -> None:
    """Пропускная способность и goodput от числа клиентов."""
    levels = [
        (int(clients), e)
        for clients, e in series["metrics"]["by_clients"].items()
        if e.get("throughput_tok_s") is not None
    ]
    xs = np.array([clients for clients, _ in levels])
    throughput = np.array([e["throughput_tok_s"] for _, e in levels])
    goodput = np.array([e.get("goodput_rps") or 0.0 for _, e in levels])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(xs, throughput, marker="o", label="пропускная, ток/с")
    ax.plot(xs, goodput, marker="s", label="goodput, зап/с")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Одновременных клиентов")
    ax.set_ylabel("Значение")
    ax.set_title("Пропускная способность и goodput от нагрузки")
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
    with mlflow.start_run(run_name="exp3-report"):
        mlflow.set_tags(
            {"work": "lr1", "experiment": "3", "source_run_id": run_id}
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

    tables = level_tables(series)
    tables_path = REPORTS_DIR / "experiment_3_tables.md"
    tables_path.write_text(tables + "\n", encoding="utf-8")

    json_path = REPORTS_DIR / "experiment_3.json"
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
        ("exp3_throughput_latency.png", plot_throughput_latency),
        ("exp3_throughput_clients.png", plot_throughput_clients),
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
