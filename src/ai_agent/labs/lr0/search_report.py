"""Отчётные артефакты эксперимента 2 (поиск Optuna)."""

import os
from collections import Counter
from pathlib import Path
from statistics import median

from ai_agent import files

# matplotlib и fontconfig требуют записываемый кэш; держим его в проекте.
mp_cache = files.PROJECT_ROOT / ".cache"
mp_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(mp_cache))
os.environ.setdefault("MPLCONFIGDIR", str(mp_cache / "matplotlib"))

import matplotlib  # noqa: E402
import optuna  # noqa: E402
from optuna.trial import FrozenTrial, TrialState  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

REPORTS_DIR = files.PROJECT_ROOT / "reports" / "lr0"


def _trial_epochs(trial: FrozenTrial) -> int:
    """Сколько эпох trial реально сообщил прунеру."""
    return len(trial.intermediate_values)


def _state_counts(study: optuna.Study) -> dict[str, int]:
    """Число trial в каждом состоянии."""
    counter = Counter(trial.state.name for trial in study.trials)
    return {state.name: counter.get(state.name, 0) for state in TrialState}


def _avg_epoch_seconds(study: optuna.Study) -> float:
    """Медианное время эпохи по завершённым trial."""
    values = []
    for trial in study.trials:
        epochs = _trial_epochs(trial)
        if (
            trial.state == TrialState.COMPLETE
            and trial.duration is not None
            and epochs
        ):
            values.append(trial.duration.total_seconds() / epochs)
    if not values:
        for trial in study.trials:
            epochs = _trial_epochs(trial)
            if trial.duration is not None and epochs:
                values.append(trial.duration.total_seconds() / epochs)
    return median(values) if values else 0.0


def _search_seconds(study: optuna.Study) -> float:
    """Суммарная длительность trial в секундах."""
    return sum(
        trial.duration.total_seconds()
        for trial in study.trials
        if trial.duration is not None
    )


def _best_trial(study: optuna.Study) -> dict | None:
    """Лучший trial в виде словаря или None, если завершённых нет."""
    try:
        best = study.best_trial
    except ValueError:
        return None
    return {
        "number": best.number,
        "value": best.value,
        "params": dict(best.params),
    }


def _importances(study: optuna.Study) -> tuple[dict[str, float], str | None]:
    """Важность параметров Optuna и текст ошибки, если посчитать не удалось."""
    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception as exc:
        return {}, f"{type(exc).__name__}: {exc}"
    return dict(importances), None


def _plot_trials(study: optuna.Study, path: Path) -> bool:
    """Строит график номера trial и val accuracy с линией лучшего значения."""
    completed = [
        (trial.number, trial.value)
        for trial in study.trials
        if trial.state == TrialState.COMPLETE and trial.value is not None
    ]
    completed.sort()

    pruned = []
    for trial in study.trials:
        if trial.state != TrialState.PRUNED:
            continue
        if not trial.intermediate_values:
            continue
        last_step = max(trial.intermediate_values)
        pruned.append((trial.number, trial.intermediate_values[last_step]))

    if not completed and not pruned:
        return False

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if completed:
        numbers = [item[0] for item in completed]
        values = [item[1] for item in completed]
        ax.scatter(numbers, values, s=28, label="COMPLETE")

        best_numbers = []
        best_values = []
        running_best = float("-inf")
        for number, value in completed:
            running_best = max(running_best, value)
            best_numbers.append(number)
            best_values.append(running_best)
        ax.step(
            best_numbers,
            best_values,
            where="post",
            color="C1",
            label="лучший на данный момент",
        )

    if pruned:
        ax.scatter(
            [item[0] for item in pruned],
            [item[1] for item in pruned],
            s=24,
            marker="x",
            color="0.55",
            label="PRUNED (последняя val)",
        )

    ax.set_xlabel("Номер trial")
    ax.set_ylabel("val accuracy")
    ax.set_title("Поиск гиперпараметров: trial → val accuracy")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def _plot_importance(importances: dict[str, float], path: Path) -> bool:
    """Строит горизонтальную диаграмму важности параметров."""
    if not importances:
        return False

    items = sorted(importances.items(), key=lambda item: item[1])
    names = [item[0] for item in items]
    values = [item[1] for item in items]

    height = max(2.0, 0.6 * len(items) + 1.2)
    fig, ax = plt.subplots(figsize=(7, height))
    ax.barh(names, values, color="C0")
    ax.set_xlabel("Важность параметра")
    ax.set_title("Важность гиперпараметров (Optuna)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def save_search_report(
    study: optuna.Study,
    epochs_per_trial: int,
    target_trials: int,
    out_dir: Path = REPORTS_DIR,
) -> dict:
    """Сохраняет JSON и графики эксперимента 2 в reports/lr0.

    params:
        study - study после поиска
        epochs_per_trial - число эпох в одном trial (30)
        target_trials - плановое число trial (40)
        out_dir - каталог отчётов

    return: словарь записанного отчёта
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_trials = len(study.trials)
    states = _state_counts(study)
    executed_epochs = sum(_trial_epochs(trial) for trial in study.trials)
    planned_actual = n_trials * epochs_per_trial
    planned_40 = target_trials * epochs_per_trial
    pruner_saved = sum(
        epochs_per_trial - _trial_epochs(trial)
        for trial in study.trials
        if trial.state == TrialState.PRUNED
    )

    avg_epoch = _avg_epoch_seconds(study)
    search_seconds = _search_seconds(study)
    importances, importance_error = _importances(study)

    plots = {}
    plot_errors = []
    trials_plot = out_dir / "experiment_2_trials.png"
    try:
        if _plot_trials(study, trials_plot):
            plots["trials"] = trials_plot.name
    except Exception as exc:
        plot_errors.append(f"trial plot: {type(exc).__name__}: {exc}")

    importance_plot = out_dir / "experiment_2_importance.png"
    try:
        if _plot_importance(importances, importance_plot):
            plots["importance"] = importance_plot.name
    except Exception as exc:
        plot_errors.append(f"importance plot: {type(exc).__name__}: {exc}")

    report = {
        "study_name": study.study_name,
        "n_trials": n_trials,
        "target_trials": target_trials,
        "epochs_per_trial": epochs_per_trial,
        "states": states,
        "epochs": {
            "planned_40": planned_40,
            "planned_actual": planned_actual,
            "executed": executed_epochs,
            "executed_share_40": (
                executed_epochs / planned_40 if planned_40 else 0.0
            ),
            "pruner_saved": pruner_saved,
            "pruner_saved_share_40": (
                pruner_saved / planned_40 if planned_40 else 0.0
            ),
        },
        "time": {
            "search_seconds": search_seconds,
            "avg_epoch_seconds": avg_epoch,
            "pruner_saved_seconds": pruner_saved * avg_epoch,
            "full_estimate_seconds": planned_actual * avg_epoch,
        },
        "best_trial": _best_trial(study),
        "importances": importances,
        "importance_error": importance_error,
        "plots": plots,
        "plot_errors": plot_errors,
    }

    report_path = out_dir / "experiment_2.json"
    files.write_json(report_path, report)
    report["report_path"] = str(report_path)
    return report
