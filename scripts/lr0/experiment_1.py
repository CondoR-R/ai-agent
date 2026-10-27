from pathlib import Path

from ai_agent import cli, files

from .run import main as run


def main():
    args = cli.parse_args()

    config_path = Path(args.config)
    cli_cfg = cli.get_cli_cfg(args)
    exp = int(args.exp)
    eval_test = args.eval_test
    series = args.series

    # 1 Запуск базовой конфигурации дважды с seed 0.
    cli_cfg["seed"] = 0
    run_dir_1_1 = run(config_path, cli_cfg, exp, series, eval_test)
    run_dir_1_2 = run(config_path, cli_cfg, exp, series, eval_test)

    # Файлы history.csv и predictions.csv должны совпасть полностью
    assert files.get_sha256(run_dir_1_1 / "history.csv") == files.get_sha256(
        run_dir_1_2 / "history.csv"
    ) and files.get_sha256(run_dir_1_1 / "predictions.csv") == files.get_sha256(
        run_dir_1_2 / "predictions.csv"
    )

    # 2 Серия base_seeds: базовая конфигурация с seed от 1 до 5.
    for seed in range(1, 6):
        cli_cfg["seed"] = seed
        run_dir_2 = run(config_path, cli_cfg, exp, "base_seeds", eval_test)


if __name__ == "__main__":
    main()
