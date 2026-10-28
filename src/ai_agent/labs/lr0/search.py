"""Поиск гиперпараметров ЛР0 в Optuna с логированием в MLflow."""

import copy
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlflow
import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

from ai_agent import files, reproducibility, tracking, utils

from . import train
from .run import load_inputs, prepare

SEARCH_NAME_TAG = "search_name"
STORAGE_URI = f"sqlite:///{(files.PROJECT_ROOT / 'optuna.db').as_posix()}"


def set_by_path(cfg: dict, path: str, value: Any) -> None:
    """Записывает значение по пути вида 'train.lr'

    params:
        cfg - словарь конфигурации (изменяется)
        path - путь через точки
        value - значение
    """
    *parents, key = path.split(".")
    node = cfg
    for name in parents:
        node = node[name]
    node[key] = value


def suggest(trial: optuna.Trial, name: str, spec: dict) -> Any:
    """Предлагает значение одного параметра по описанию из search_space.yaml

    params:
        trial - текущий trial
        name - имя параметра
        spec - описание: type, low/high/log или choices

    raise:
        ValueError, если тип параметра неизвестен
    """
    kind = spec["type"]
    if kind == "float":
        return trial.suggest_float(
            name, spec["low"], spec["high"], log=spec.get("log", False)
        )
    if kind == "int":
        return trial.suggest_int(
            name, spec["low"], spec["high"], log=spec.get("log", False)
        )
    if kind == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"Неизвестный тип параметра {name}: {kind}")


def build_trial_cfg(
    base_cfg: dict, space: dict, params: dict[str, Any]
) -> dict:
    """Конфигурация trial: базовая плюс значения параметров поиска

    params:
        base_cfg - базовая конфигурация (не изменяется)
        space - секция space из search_space.yaml
        params - значения параметров {имя: значение}
    """
    cfg = copy.deepcopy(base_cfg)
    for name, spec in space.items():
        set_by_path(cfg, spec["target"], params[name])
    return cfg


def make_objective(
    base_cfg: dict, space: dict, inputs: tuple
) -> Callable[[optuna.Trial], float]:
    """Создаёт функцию цели: обучает сеть и возвращает val_acc после последней эпохи.

    Каждый trial — вложенная запись MLflow с параметрами, метриками
    по эпохам и состоянием в теге state (COMPLETE, PRUNED, FAIL).

    params:
        base_cfg - базовая конфигурация (seed и epochs одинаковы для всех trial)
        space - секция space из search_space.yaml
        inputs - кортеж (features, labels, split) из load_inputs
    """

    def objective(trial: optuna.Trial) -> float:
        params = {
            name: suggest(trial, name, spec) for name, spec in space.items()
        }
        cfg = build_trial_cfg(base_cfg, space, params)

        mlflow.start_run(run_name=f"trial-{trial.number}", nested=True)
        status = "FINISHED"
        try:
            logged = {
                k: v
                for k, v in cfg.items()
                if k not in tracking.EXCLUDED_PARAMS
            }
            mlflow.log_params(tracking.flatten(logged))
            mlflow.set_tags({"trial_number": trial.number, "series": "search"})

            reproducibility.set_seed(cfg["seed"], cfg["runtime"]["num_threads"])
            ctx = prepare(cfg, *inputs)

            val_acc = float("nan")
            for record in train.train(
                ctx.model,
                cfg["train"],
                ctx.train_loader,
                ctx.val_loader,
                ctx.device,
            ):
                tracking.log_epoch(record)
                val_acc = record["val_acc"]
                if math.isnan(val_acc):
                    break
                trial.report(val_acc, step=record["epoch"])
                if trial.should_prune():
                    mlflow.set_tag("state", "PRUNED")
                    raise optuna.TrialPruned()

            if math.isnan(val_acc):
                mlflow.set_tag("state", "FAIL")
                return val_acc  # Optuna сам пометит trial как FAIL
            mlflow.set_tag("state", "COMPLETE")
            return val_acc
        except optuna.TrialPruned:
            raise
        except BaseException as exc:
            mlflow.set_tags(
                {
                    "state": "FAIL",
                    "error_message": f"{type(exc).__name__}: {exc}",
                }
            )
            status = "FAILED"
            raise
        finally:
            mlflow.end_run(status=status)

    return objective


def create_study(search_cfg: dict) -> optuna.Study:
    """Создаёт study в optuna.db или открывает существующий (load_if_exists)

    params:
        search_cfg - содержимое search_space.yaml
    """
    return optuna.create_study(
        study_name=search_cfg["study_name"],
        storage=STORAGE_URI,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=search_cfg["sampler_seed"]),
        pruner=optuna.pruners.MedianPruner(**search_cfg["pruner"]),
    )


def run_search(
    base_config_path: Path, search_path: Path, n_trials: int | None = None
) -> optuna.Study:
    """Поиск гиперпараметров: до n_trials trial в сумме по study.

    Повторный вызов продолжает поиск: недостающие trial выполняются
    под той же родительской записью MLflow. Seed семплера при
    продолжении прежнюю последовательность не восстанавливает.

    params:
        base_config_path - базовая конфигурация (configs/lr0/base.yaml)
        search_path - описание поиска (configs/lr0/search_space.yaml)
        n_trials - общее число trial в study (по умолчанию из search_space.yaml)

    return: study после поиска
    """
    utils.check_git()

    base_cfg = files.read_yaml(base_config_path)
    search_cfg = files.read_yaml(search_path)
    target = n_trials or search_cfg["n_trials"]
    name = search_cfg["study_name"]

    experiment_id = tracking.setup(base_cfg["experiment"])
    existing = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.{SEARCH_NAME_TAG} = '{name}'",
        output_format="list",
        order_by=["start_time ASC"],
        max_results=1,
    )
    parent_id = existing[0].info.run_id if existing else None

    inputs = load_inputs()
    study = create_study(search_cfg)

    # if parent_id:
    #     parent = mlflow.start_run(run_id=parent_id)
    if not parent_id:
        # else:
        # parent = mlflow.start_run(run_name=name)
        mlflow.log_params(
            {
                "study_name": name,
                "sampler_seed": search_cfg["sampler_seed"],
                **{f"pruner.{k}": v for k, v in search_cfg["pruner"].items()},
            }
        )
        mlflow.log_artifact(str(search_path))
    try:
        mlflow.set_tags(
            {
                SEARCH_NAME_TAG: name,
                "series": "search",
                "commit": utils.get_commit_hash(),
                "target_trials": target,
            }
        )
        study.optimize(
            make_objective(base_cfg, search_cfg["space"], inputs),
            n_trials=target,
            callbacks=[MaxTrialsCallback(target, states=None)],
        )
        mlflow.set_tags(
            {
                f"n_{s.name.lower()}": len(study.get_trials(states=(s,)))
                for s in TrialState
            }
        )
        mlflow.end_run()
    except BaseException:
        mlflow.end_run(status="FAILED")
        raise
    return study


def save_best_config(
    study: optuna.Study,
    base_config_path: Path,
    search_path: Path,
    out_path: Path,
) -> dict:
    """Записывает параметры лучшего trial в конфигурацию (configs/lr0/best.yaml)

    params:
        study - study после поиска
        base_config_path - базовая конфигурация
        search_path - описание поиска
        out_path - куда писать best.yaml

    return: словарь итоговой конфигурации
    """
    base_cfg = files.read_yaml(base_config_path)
    space = files.read_yaml(search_path)["space"]
    cfg = build_trial_cfg(base_cfg, space, study.best_trial.params)
    files.write_yaml(out_path, cfg)
    return cfg
