"""Эксперимент 3: честное сравнение базовой и лучшей конфигураций."""

import math
from pathlib import Path
from statistics import median

from run import main as run

from ai_agent import files
from ai_agent.labs.lr0.analysis import REPORTS_DIR, compare_predictions

REPORT_PATH = REPORTS_DIR / "experiment_3.json"
SEEDS = range(101, 106)


def analyze(pairs: dict[int, tuple[Path, Path]]) -> dict:
    """Сравнивает base и best на val и test для каждого seed.

    params:
        pairs - словарь {seed: (каталог base, каталог best)}

    return: словарь с результатами по seed и сводкой

    raise:
        ValueError, если в metrics.json нет test_acc или расходятся
        суммарный и средний Δ на test
    """
    if not pairs:
        raise ValueError("Нет запусков для анализа")

    seeds: dict[int, dict] = {}
    delta_val = []
    delta_test = []
    corrected = 0
    spoiled = 0
    n_test = 0

    for seed in sorted(pairs):
        base_dir, best_dir = pairs[seed]
        base_metrics = files.read_json(base_dir / "metrics.json")
        best_metrics = files.read_json(best_dir / "metrics.json")

        for key in ("val_acc", "test_acc"):
            if key not in base_metrics or key not in best_metrics:
                raise ValueError(
                    f"seed={seed}: в metrics.json нет {key} — "
                    "запуск выполнен без --eval-test?"
                )

        d_val = best_metrics["val_acc"] - base_metrics["val_acc"]
        d_test = best_metrics["test_acc"] - base_metrics["test_acc"]

        comparison = compare_predictions(
            base_dir, best_dir, split="test", with_examples=False
        )
        counts = comparison["counts"]

        # only_second_correct: base ошибся, best верно (исправлено);
        # only_first_correct: base верно, best ошибся (испорчено).
        corrected += counts["only_second_correct"]
        spoiled += counts["only_first_correct"]
        n_test += comparison["n"]

        seeds[seed] = {
            "base_run_id": base_dir.name,
            "best_run_id": best_dir.name,
            "base": {
                "val_acc": base_metrics["val_acc"],
                "test_acc": base_metrics["test_acc"],
            },
            "best": {
                "val_acc": best_metrics["val_acc"],
                "test_acc": best_metrics["test_acc"],
            },
            "delta_val_acc": d_val,
            "delta_test_acc": d_test,
            "test_counts": {
                "both_correct": counts["both_correct"],
                "corrected": counts["only_second_correct"],
                "spoiled": counts["only_first_correct"],
                "both_wrong": counts["both_wrong"],
            },
        }

        delta_val.append(d_val)
        delta_test.append(d_test)

    delta_test_total = (corrected - spoiled) / n_test
    mean_delta_test = sum(delta_test) / len(delta_test)
    if not math.isclose(delta_test_total, mean_delta_test, abs_tol=1e-12):
        raise ValueError(
            "Суммарный Δ по изображениям не совпал со средним Δ по seed: "
            f"{delta_test_total} != {mean_delta_test}"
        )

    return {
        "seeds": seeds,
        "summary": {
            "n_seeds": len(seeds),
            "n_test": n_test,
            "corrected": corrected,
            "spoiled": spoiled,
            "net": corrected - spoiled,
            "delta_test_total": delta_test_total,
            "median_delta_val_acc": median(delta_val),
            "min_delta_val_acc": min(delta_val),
            "max_delta_val_acc": max(delta_val),
            "median_delta_test_acc": median(delta_test),
            "min_delta_test_acc": min(delta_test),
            "max_delta_test_acc": max(delta_test),
        },
    }


def save_report(statistic: dict) -> Path:
    """Записывает отчёт эксперимента 3 в reports/lr0/experiment_3.json."""
    files.write_json(REPORT_PATH, statistic)
    return REPORT_PATH


def main() -> None:
    # Параметры зашиты, чтобы не менять общий CLI под нужды эксперимента.
    series = "final"
    base_path = Path("configs/lr0/base.yaml")
    best_path = Path("configs/lr0/best.yaml")
    eval_test = True

    pairs: dict[int, tuple[Path, Path]] = {}
    for seed in SEEDS:
        cli_cfg = {"seed": seed}
        base_dir = run(base_path, cli_cfg, 3, series, eval_test)
        best_dir = run(best_path, cli_cfg, 3, series, eval_test)
        pairs[seed] = (base_dir, best_dir)

    statistic = analyze(pairs)
    statistic["series"] = series
    save_report(statistic)

    summary = statistic["summary"]
    print(
        f"Δ test: медиана {summary['median_delta_test_acc']:+.4f}, "
        f"исправлено {summary['corrected']}, "
        f"испорчено {summary['spoiled']} из {summary['n_test']}"
    )


if __name__ == "__main__":
    main()
