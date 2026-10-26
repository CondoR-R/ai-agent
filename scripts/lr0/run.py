"""Скрипт экспериментов"""

import argparse
from pathlib import Path

from ai_agent import git_utils, paths, reproducibility
from ai_agent.labs.lr0 import (
    config,
    data,
    history,
    metadata,
    predictions,
    train,
)
from ai_agent.labs.lr0.model import build_model


def parse_args() -> argparse.Namespace:
    """Обработка аргументов командной строки"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed")
    parser.add_argument("--split-seed")
    parser.add_argument("--eval-test", action="store_true")

    return parser.parse_args()


def get_cli_cfg(args: argparse.Namespace) -> dict:
    cfg = {}
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.split_seed is not None:
        cfg["split_seed"] = int(args.split_seed)

    return cfg


def main(config_path: Path, cli_cfg: dict, eval_test: bool, experiment: int):
    # проверка чистоты дерева
    git_utils.check_git()

    # конфигурация
    file_cfg = config.load(config_path)
    cfg = config.merge(file_cfg, cli_cfg)

    # фиксация воспроизводимости
    reproducibility.set_seed(cfg["seed"], cfg["runtime"]["num_threads"])
    generator = reproducibility.make_generator(cfg["seed"])

    # создание директории <run_id>/
    run_id = paths.get_run_id(cfg["experiment"], experiment)
    run_dir = paths.create_run_dir(run_id)

    # сохранение конфигурации
    config.save(run_dir / paths.CONFIG_FILENAME, cfg)

    # начальное состояние metadata.json
    split_path = paths.DATA_DIR / "split.json"
    split_sha256 = data.file_sha256(split_path)
    m = metadata.get_metadata(
        run_id,
        "test",
        cfg["runtime"]["num_threads"],
        cfg["seed"],
        ["torch", "scikit-learn", "optuna", "mlflow"],
        {"split.json": split_sha256},
    )
    metadata.write(run_dir / paths.METADATA_FILENAME, m)

    # получение данных
    features, labels = data.load_data()
    split = data.load_split(split_path)

    train_features, train_labels = (
        features[split["train"]],
        labels[split["train"]],
    )
    val_features, val_labels = features[split["val"]], labels[split["val"]]
    test_features, test_labels = features[split["test"]], labels[split["test"]]

    device = cfg["runtime"]["device"]
    train_loader = data.get_loader(
        train_features,
        train_labels,
        cfg["train"]["batch_size"],
        generator,
        shuffle=True,
    )
    val_loader = data.get_loader(
        val_features,
        val_labels,
        1,
        generator,
    )
    test_loader = data.get_loader(
        test_features,
        test_labels,
        1,
        generator,
    )

    # обучение модели
    model = build_model(cfg["model"], device)
    for record in train.train(
        model, cfg["train"], train_loader, val_loader, device
    ):
        history.update(run_dir / "history.csv", record)

    # predictions
    preds = predictions.get_preds(
        model, val_loader, split["val"], "val", device
    )
    if eval_test:
        preds.extend(
            predictions.get_preds(
                model, test_loader, split["test"], "test", device
            )
        )
    predictions.write(run_dir / 'predictions.csv', preds)

if __name__ == "__main__":
    args = parse_args()
    config_path = Path(args.config)
    cli_cfg = get_cli_cfg(args)
    eval_test = args.eval_test
    main(config_path, cli_cfg, eval_test, 0)
