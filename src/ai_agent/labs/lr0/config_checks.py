"""Валидация конфига с учетом особенных полей lr0"""

from ai_agent.core import config as core_config

LR0_FIELDS = (
    "experiment",
    "split_seed",
    "seed",
    "model",
    "train",
    "runtime",
    "tracked_libs",
)


def validate_lr0(config: core_config.ConfigDict) -> None:
    """Проверка конфигурации ЛР0: обязательные поля и значения train.lr, train.epochs

    raise:
        ValueError, при отсутствии полей и некорректных значениях
    """
    core_config.validate(config, fields=LR0_FIELDS)

    train = config["train"]
    if not isinstance(train, dict):
        raise ValueError("Поле train должно быть словарём")

    for key in ("lr", "epochs"):
        if key not in train:
            raise ValueError(f"Отсутствует обязательное поле train.{key}")

    if train["lr"] <= 0:
        raise ValueError(f"train.lr должен быть > 0, получено {train['lr']}")
    if train["epochs"] < 1:
        raise ValueError(
            f"train.epochs должен быть >= 1, получено {train['epochs']}"
        )
