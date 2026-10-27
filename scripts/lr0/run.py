"""Скрипт одного запуска"""

from pathlib import Path

from ai_agent import cli, files, reproducibility, utils
from ai_agent.labs.lr0 import (
    data,
    metrics,
    train,
)
from ai_agent.labs.lr0.model import build_model


def main(config_path: Path, cli_cfg: dict, eval_test: bool, experiment: int):
    # проверка чистоты дерева
    utils.check_git()

    # конфигурация
    file_cfg = files.read_yaml(config_path)
    cfg = utils.merge_cfgs(file_cfg, cli_cfg)

    # фиксация воспроизводимости
    reproducibility.set_seed(cfg["seed"], cfg["runtime"]["num_threads"])
    generator = reproducibility.make_generator(cfg["seed"])

    # создание директории <run_id>/
    run_id = files.get_run_id(cfg["experiment"], experiment)
    run_dir = files.create_run_dir(run_id)

    # сохранение конфигурации
    files.write_yaml(run_dir / files.CONFIG_FILENAME, cfg)

    # начальное состояние metadata.json
    split_path = files.DATA_DIR / "split.json"
    split_sha256 = files.get_sha256(split_path)
    m = utils.get_metadata(
        run_id,
        "test",
        cfg["runtime"]["num_threads"],
        cfg["seed"],
        ["torch", "scikit-learn", "optuna", "mlflow"],
        {"split.json": split_sha256},
    )
    files.write_json(run_dir / files.METADATA_FILENAME, m)

    # получение данных
    features, labels = data.load_data()
    split = files.read_json(split_path)

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
    history_path = run_dir / "history.csv"
    for record in train.train(
        model, cfg["train"], train_loader, val_loader, device
    ):
        files.update_csv(history_path, record)

    # predictions
    preds = train.get_preds(model, val_loader, split["val"], "val", device)
    if eval_test:
        preds.extend(
            train.get_preds(model, test_loader, split["test"], "test", device)
        )
    predictions_path = run_dir / "predictions.csv"
    files.write_csv(predictions_path, preds)

    # metrics
    h = files.read_csv(
        history_path,
        cast={
            "val_loss": float,
            "time_epoch": float,
        },
    )
    val_loss = metrics.get_val_loss(h)
    train_time = metrics.get_train_time(h)
    p = files.read_csv(
        predictions_path,
        cast={
            "y_true": int,
            "y_pred": int,
        },
    )
    val_acc = metrics.get_acc(p, "val")

    metrics_dict = {
        "val_acc": val_acc,
        "val_loss": val_loss,
        "train_time": train_time,
    }
    if eval_test:
        metrics_dict["test_acc"] = metrics.get_acc(p, "test")

    files.write_json(run_dir / "metrics.json ", metrics_dict)

    # добавить обновление статуса в metadata
    # округление до 4 знаков после запятой при записи


if __name__ == "__main__":
    args = cli.parse_args()
    config_path = Path(args.config)
    cli_cfg = cli.get_cli_cfg(args)
    eval_test = args.eval_test
    main(config_path, cli_cfg, eval_test, 0)
