"""Поиск гиперпараметров ЛР0 (эксперимент 2)"""

import argparse
from pathlib import Path

from ai_agent.labs.lr0 import search


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lr0/base.yaml")
    parser.add_argument("--search", default="configs/lr0/search_space.yaml")
    parser.add_argument("--trials", type=int, default=None, help="общее число trial в study")
    parser.add_argument("--save-best", default=None, help="куда записать best.yaml")
    args = parser.parse_args()

    study = search.run_search(Path(args.config), Path(args.search), args.trials)
    print(f"Лучший trial #{study.best_trial.number}: val_acc={study.best_value:.4f}")
    print(study.best_trial.params)

    if args.save_best:
        search.save_best_config(study, Path(args.config), Path(args.search), Path(args.save_best))
        print(f"Параметры сохранены в {args.save_best}")


if __name__ == "__main__":
    main()
