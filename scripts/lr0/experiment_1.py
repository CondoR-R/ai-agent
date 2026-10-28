"""Эксперимент 1: проверка воспроизводимости базовой конфигурации"""

from pathlib import Path
from statistics import median

from compare_csv import compare_csv
from run import main as run

from ai_agent import cli, files
from ai_agent.labs.lr0.analysis import compare_predictions, update_report


def analyze_task1(run_dirs: list[Path]) -> None:
    """Проверяет, что два запуска с seed 0 дали одинаковые результаты."""
    assert compare_csv(
        run_dirs[0] / "history.csv",
        run_dirs[1] / "history.csv",
        columns=["epoch", "train_loss", "val_loss", "val_acc"],
    ), "history.csv двух запусков с seed 0 различаются"

    assert compare_csv(
        run_dirs[0] / "predictions.csv",
        run_dirs[1] / "predictions.csv",
        columns=["split", "index", "y_true", "y_pred"],
    ), "predictions.csv двух запусков с seed 0 различаются"

    update_report(
        "experiment_1.json",
        "task_1",
        {
            "run_ids": [run_dir.name for run_dir in run_dirs],
            "identical": True,
        },
    )


def analyze_task2(run_dirs: list[Path]) -> None:
    """Собирает val_acc и время обучения для серии base_seeds."""
    val_acc_list = []
    statistic = {"seeds": {}, "summary": {}}

    for run_dir in run_dirs:
        seed = files.read_yaml(run_dir / "config.yaml")["seed"]
        metrics = files.read_json(run_dir / "metrics.json")

        statistic["seeds"][seed] = {
            "run_id": run_dir.name,
            "val_acc": metrics["val_acc"],
            "train_time": metrics["train_time"],
        }
        val_acc_list.append(metrics["val_acc"])

    statistic["summary"] = {
        "min_acc": min(val_acc_list),
        "median_acc": median(val_acc_list),
        "max_acc": max(val_acc_list),
    }
    update_report("experiment_1.json", "task_2", statistic)


def analyze_task3(run_dirs: list[Path]) -> None:
    """Сравнивает предсказания двух запусков с разными seed на val."""
    seed_1 = files.read_yaml(run_dirs[0] / "config.yaml")["seed"]
    seed_2 = files.read_yaml(run_dirs[1] / "config.yaml")["seed"]
    if seed_1 == seed_2:
        raise ValueError("Для сравнения нужны запуски с разными seed")

    result = compare_predictions(run_dirs[0], run_dirs[1], split="val")
    result["run_ids"] = [run_dir.name for run_dir in run_dirs]
    result["n_val"] = result.pop("n")
    update_report("experiment_1.json", "task_3", result)


def main() -> None:
    """Выполняет запуски эксперимента 1 и сохраняет отчёт."""
    args = cli.parse_args()

    config_path = Path(args.config)
    cli_cfg = cli.get_cli_cfg(args)
    exp = 1
    series = args.series

    # Эксперимент 1 работает только с train/val: test оценивается в exp3.
    if args.eval_test:
        print("Эксперимент 1 не оценивает test: --eval-test игнорируется")
    eval_test = False

    # 1. Базовая конфигурация дважды с seed 0.
    print("[EXP1] 1/3: повторяемость — 2 запуска с seed=0", flush=True)
    cli_cfg["seed"] = 0
    run_dir_1_1 = run(config_path, cli_cfg, exp, series, eval_test)
    run_dir_1_2 = run(config_path, cli_cfg, exp, series, eval_test)

    # 2. Серия base_seeds: seed от 1 до 5.
    print("[EXP1] 2/3: серия base_seeds — seeds 1..5", flush=True)
    run_dirs_2 = []
    for seed in range(1, 6):
        cli_cfg["seed"] = seed
        run_dir_2 = run(config_path, cli_cfg, exp, "base_seeds", eval_test)
        run_dirs_2.append(run_dir_2)

    # 3. Сравниваем seed 0 из task_1 и seed 1 из task_2.
    print("[EXP1] 3/3: сравнение предсказаний seed 0 и seed 1", flush=True)
    analyze_task1([run_dir_1_1, run_dir_1_2])
    analyze_task2(run_dirs_2)
    analyze_task3([run_dir_1_1, run_dirs_2[0]])
    print("[EXP1] отчёт: reports/lr0/experiment_1.json", flush=True)


if __name__ == "__main__":
    main()
