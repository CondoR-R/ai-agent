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
    val_acc_list = []
    statistic = {
        "seeds": {},
        "summury": {},
    }
    for run_dir in run_dirs:
        seed = files.read_json(run_dir / 'metrics.json')["seed"]
        metrics = files.read_json(run_dir)
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

    files.write_json(
        files.PROJECT_ROOT / "reports/lr0/experiment_1.json", statistic
    )


def analyze(run_dirs: dict) -> None:
    analyze_task1(run_dirs["task_1"])
    analyze_task2(run_dirs["task_2"])


def main():
    # обработка аргументов командной строки
    args = cli.parse_args()

    config_path = Path(args.config)
    cli_cfg = cli.get_cli_cfg(args)
    exp = 1
    eval_test = args.eval_test
    series = args.series

    run_dirs = {}

    # 1 Запуск базовой конфигурации дважды с seed 0.
    cli_cfg["seed"] = 0
    run_dir_1_1 = run(config_path, cli_cfg, exp, series, eval_test)
    run_dir_1_2 = run(config_path, cli_cfg, exp, series, eval_test)
    run_dirs["task_1"] = [run_dir_1_1, run_dir_1_2]

    # 2 Серия base_seeds: базовая конфигурация с seed от 1 до 5.
    run_dirs["task_2"] = []
    for seed in range(1, 6):
        cli_cfg["seed"] = seed
        run_dir_2 = run(config_path, cli_cfg, exp, "base_seeds", eval_test)
        run_dirs["task_2"].append(run_dir_2)

    # 3 Сравнение предсказаний двух запусков с разными seed
    cli_cfg["seed"] = 0
    run_dir_3_1 = run(config_path, cli_cfg, exp, series, eval_test)
    cli_cfg["seed"] = 1
    run_dir_3_2 = run(config_path, cli_cfg, exp, series, eval_test)

    run_dirs["task_3"] = []
    run_dirs["task_3"].append(run_dir_3_1)
    run_dirs["task_3"].append(run_dir_3_2)

    analyze(run_dirs)


if __name__ == "__main__":
    main()
