from pathlib import Path

from compare_csv import compare_csv
from run import main as run

from ai_agent import cli, files


def main():
    args = cli.parse_args()

    config_path = Path(args.config)
    cli_cfg = cli.get_cli_cfg(args)
    exp = 1
    eval_test = args.eval_test
    series = args.series

    run_dirs = []

    # 1 Запуск базовой конфигурации дважды с seed 0.
    cli_cfg["seed"] = 0
    run_dir_1_1 = run(config_path, cli_cfg, exp, series, eval_test)
    run_dir_1_2 = run(config_path, cli_cfg, exp, series, eval_test)

    run_dirs.append(run_dir_1_1)
    run_dirs.append(run_dir_1_2)

    # Файлы history.csv и predictions.csv должны совпасть
    assert compare_csv(
        run_dir_1_1 / "history.csv",
        run_dir_1_2 / "history.csv",
        columns=["epoch", "train_loss", "val_loss", "val_acc"],
    )
    assert compare_csv(
        run_dir_1_1 / "predictions.csv",
        run_dir_1_2 / "predictions.csv",
        columns=["split", "index", "y_true", "y_pred"],
    )

    # 2 Серия base_seeds: базовая конфигурация с seed от 1 до 5.
    for seed in range(1, 6):
        cli_cfg["seed"] = seed
        run_dir_2 = run(config_path, cli_cfg, exp, "base_seeds", eval_test)
        run_dirs.append(run_dir_2)

    # 3 Сравнение предсказаний двух запусков с разными seed
    cli_cfg["seed"] = 0
    run_dir_3_1 = run(config_path, cli_cfg, exp, series, eval_test)
    cli_cfg["seed"] = 1
    run_dir_3_2 = run(config_path, cli_cfg, exp, series, eval_test)

    run_dirs.append(run_dir_3_1)
    run_dirs.append(run_dir_3_2)

    # Сохранение путей к запускам в JSON
    files.write_json(
        files.PROJECT_ROOT / "experiments" / "lr0.json",
        {"exp_1": [str(p) for p in run_dirs]},
    )


if __name__ == "__main__":
    main()
