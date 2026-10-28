"""Эксперимент 2: поиск гиперпараметров ЛР0"""

import argparse
from pathlib import Path

from optuna.trial import TrialState

from ai_agent import files
from ai_agent.labs.lr0 import search


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lr0/base.yaml")
    parser.add_argument("--search", default="configs/lr0/search_space.yaml")
    parser.add_argument(
        "--trials", type=int, default=40, help="общее число trial в study"
    )
    parser.add_argument(
        "--save-best", default=None, help="куда записать best.yaml"
    )
    args = parser.parse_args()

    limit = args.trials if args.trials is not None else "из YAML"
    print(
        f"[EXP2] поиск: конфиг={args.config}, "
        f"пространство={args.search}, лимит trial={limit}",
        flush=True,
    )

    study = search.run_search(Path(args.config), Path(args.search), args.trials)

    counts = {
        state.name: len(study.get_trials(states=(state,)))
        for state in TrialState
    }
    states_text = ", ".join(
        f"{name}={count}" for name, count in counts.items() if count
    )
    print(
        f"[EXP2] trial: {len(study.trials)} ({states_text})", flush=True
    )
    print(
        f"[EXP2] лучший trial #{study.best_trial.number}: "
        f"val_acc={study.best_value:.4f}",
        flush=True,
    )
    print(f"[EXP2] параметры: {study.best_trial.params}", flush=True)

    if args.save_best:
        search.save_best_config(
            study, Path(args.config), Path(args.search), Path(args.save_best)
        )
        print(f"[EXP2] best.yaml: {args.save_best}", flush=True)

    base_cfg = files.read_yaml(Path(args.config))
    search_cfg = files.read_yaml(Path(args.search))
    from ai_agent.labs.lr0 import search_report

    report = search_report.save_search_report(
        study,
        epochs_per_trial=base_cfg["train"]["epochs"],
        target_trials=search_cfg["n_trials"],
    )
    print(f"[EXP2] отчёт: {report['report_path']}", flush=True)
    if report["plots"]:
        print(
            "[EXP2] графики: " + ", ".join(report["plots"].values()),
            flush=True,
        )
    if report["plot_errors"]:
        print(
            f"[EXP2] ошибки графиков: {report['plot_errors']}", flush=True
        )


if __name__ == "__main__":
    main()
