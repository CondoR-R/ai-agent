"""Эксперимент 1: проверка воспроизводимости базовой конфигурации"""

from pathlib import Path

from compare_csv import compare_csv
from run import main as run

from ai_agent import cli, files


def main():
    # Чтобы не переписывать cli, вместо аргументов командной строки все параметры заданы переменными
    series = "final"
    base_path = Path("configs/lr0/base.yaml")
    best_path = Path("configs/lr0/best.yaml")
    seeds = [i for i in range(101, 106)]
    eval_test = True

    run_dirs = {
        'base': [],
        'best': [],
    }
    for seed in seeds:
        cli_cfg = {'seed': seed}
        run_dir_base = run(base_path, cli_cfg, 3, series, eval_test)
        run_dir_best = run(best_path, cli_cfg, 3, series, eval_test)

        run_dirs['base'].append(run_dir_base)
        run_dirs['best'].append(run_dir_best)

    try:
        exp_lr0 = files.read_json(files.PROJECT_ROOT / "experiments" / "lr0.json")
    except FileNotFoundError:
        exp_lr0 = {}

    exp_lr0["exp_3"] = run_dirs
    files.write_json(
        files.PROJECT_ROOT / "experiments" / "lr0.json",
        exp_lr0,
    )


if __name__ == "__main__":
    main()
