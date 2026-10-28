"""Эксперимент 1: проверка воспроизводимости базовой конфигурации"""

from pathlib import Path
from statistics import median

from compare_csv import compare_csv
from run import main as run

from ai_agent import cli, files


def analyze_task1(run_dirs: list[Path]) -> None:
    assert compare_csv(
        run_dirs[0] / "history.csv",
        run_dirs[1] / "history.csv",
        columns=["epoch", "train_loss", "val_loss", "val_acc"],
    )
    assert compare_csv(
        run_dirs[0] / "predictions.csv",
        run_dirs[1] / "predictions.csv",
        columns=["split", "index", "y_true", "y_pred"],
    )


def analyze_task2(run_dirs: list[Path]) -> None:
    report_path = files.PROJECT_ROOT / "reports/lr0/experiment_1.json"

    val_acc_list = []
    statistic = {
        "seeds": {},
        "summury": {},
    }
    for run_dir in run_dirs:
        seed = files.read_yaml(run_dir / 'config.yaml')["seed"]
        metrics = files.read_json(run_dir / 'metrics.json')
        val_acc, train_time = metrics["val_acc"], metrics["train_time"]
        statistic["seeds"][seed] = {
            "val_acc": val_acc,
            "train_time": train_time,
        }
        val_acc_list.append(val_acc)

    statistic["summury"] = {
        "min_acc": min(val_acc_list),
        "median_acc": median(val_acc_list),
        "max_acc": max(val_acc_list),
    }

    try:
        report = files.read_json(report_path)
    except:
        report = {}
    report["task_2"] = statistic
    files.write_json(
        report_path, report
    )

def analyze_task3(run_dirs: list[Path]) -> None:
    report_path = files.PROJECT_ROOT / "reports/lr0/experiment_1.json"

    def load_val(run_dir: Path) -> dict[int, dict]:
        rows = files.read_csv(
            run_dir / "predictions.csv",
            cast={"index": int, "y_true": int, "y_pred": int},
        )
        return {row["index"]: row for row in rows if row["split"] == "val"}

    first = load_val(run_dirs[0])
    second = load_val(run_dirs[1])

    if first.keys() != second.keys():
        raise ValueError("Разные наборы index в val — сравнивать нельзя")

    counts = {
        "both_correct": 0,
        "only_first_correct": 0,
        "only_second_correct": 0,
        "both_wrong": 0,
    }
    examples = []

    for index in sorted(first):
        row_1, row_2 = first[index], second[index]
        if row_1["y_true"] != row_2["y_true"]:
            raise ValueError(f"Разные y_true для index={index}")

        ok_1 = row_1["y_true"] == row_1["y_pred"]
        ok_2 = row_2["y_true"] == row_2["y_pred"]

        if ok_1 and ok_2:
            counts["both_correct"] += 1
        elif ok_1:
            counts["only_first_correct"] += 1
            examples.append({
                "index": index,
                "y_true": row_1["y_true"],
                "first_pred": row_1["y_pred"],
                "second_pred": row_2["y_pred"],
            })
        elif ok_2:
            counts["only_second_correct"] += 1
            examples.append({
                "index": index,
                "y_true": row_1["y_true"],
                "first_pred": row_1["y_pred"],
                "second_pred": row_2["y_pred"],
            })
        else:
            counts["both_wrong"] += 1

    n = len(first)
    result = {
        "run_ids": [run_dirs[0].name, run_dirs[1].name],
        "n_val": n,
        "counts": counts,
        "first_acc": (counts["both_correct"] + counts["only_first_correct"]) / n,
        "second_acc": (counts["both_correct"] + counts["only_second_correct"]) / n,
        "delta_acc": (
            counts["only_second_correct"] - counts["only_first_correct"]
        ) / n,
        "examples": examples,
    }

    try:
        report = files.read_json(report_path)
    except:
        report = {}
    report["task_3"] = result
    files.write_json(report_path, report)

def main():
    # обработка аргументов командной строки
    args = cli.parse_args()

    config_path = Path(args.config)
    cli_cfg = cli.get_cli_cfg(args)
    exp = 1
    eval_test = args.eval_test
    series = args.series

    # 1 Запуск базовой конфигурации дважды с seed 0.
    cli_cfg["seed"] = 0
    run_dir_1_1 = run(config_path, cli_cfg, exp, series, eval_test)
    run_dir_1_2 = run(config_path, cli_cfg, exp, series, eval_test)

    # 2 Серия base_seeds: базовая конфигурация с seed от 1 до 5.
    run_dirs_2 = []
    for seed in range(1, 6):
        cli_cfg["seed"] = seed
        run_dir_2 = run(config_path, cli_cfg, exp, "base_seeds", eval_test)
        run_dirs_2.append(run_dir_2)

    # 3 Сравнение предсказаний двух запусков с разными seed
    cli_cfg["seed"] = 0
    run_dir_3_1 = run(config_path, cli_cfg, exp, series, eval_test)
    cli_cfg["seed"] = 1
    run_dir_3_2 = run(config_path, cli_cfg, exp, series, eval_test)
    analyze_task1([run_dir_1_1, run_dir_1_2])
    analyze_task2(run_dirs_2)
    analyze_task3([run_dir_3_1, run_dir_3_2])


if __name__ == "__main__":
    main()
