"""Обработка конфигурации"""

import copy
from pathlib import Path
from typing import Any

import yaml

ConfigDict = dict[str, Any]


def load_config(path: str | Path) -> ConfigDict:
    """Получение словаря конфигурации из yaml файла

    params:
        path - путь к файлу

    return: словарь из yaml файла
    """
    text = Path(path).read_text(encoding="utf-8")
    y = yaml.safe_load(text)
    if y is None:
        y = {}
    return y


def merge_configs(f_dict: ConfigDict, cli_dict: ConfigDict) -> ConfigDict:
    """Объединение словарей конфигурации из файла и CLI

    params:
        f_dict - словарь из YAML файла
        cli_dict - словарь из коммандной строки

    return: объединенный словарь
    """
    res_dict = copy.deepcopy(f_dict)
    for key, val in cli_dict.items():
        res_dict[key] = val
    return res_dict


def validate_config(
    config: ConfigDict,
    fields: tuple[str, ...] = (
        "experiment",
        "split_seed",
        "seed",
        "model",
        "train",
        "runtime",
    ),
) -> None:
    """Проверка присутвия обязательных полей и корректность значений

    params:
        config - словарь конфигурации
        fields - кортеж обязательных полей, default = (
            "experiment",
            "split_seed",
            "seed",
            "model",
            "train",
            "runtime"
        )

    raise:
        ValueError, при отсутствии обязательных полей и неправильных значениях
    """
    # проверка на присутствие обязательных полей
    for f in fields:
        if f not in config:
            raise ValueError(f"Отсутствует обязательное поле: {f}")

    # проверка на lr > 0
    if (
        "train" not in config
        or "lr" not in config["train"]
        or config["train"]["lr"] <= 0
    ):
        raise ValueError("значение lr в train должно быть больше 0")

    # проверка на epochs > 0
    if (
        "train" not in config
        or "epochs" not in config["train"]
        or config["train"]["epochs"] <= 0
    ):
        raise ValueError(
            f"epochs должен быть >= 1, получено {config['train']['epochs']}"
        )


def save_config(path: str | Path, config: ConfigDict) -> None:
    """Сохранение YAML файла по указанному пути

    params:
        path - путь сохранения
        config - словарь конфигурации
    """
    text = yaml.safe_dump(
        config,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    Path(path).write_text(text, encoding="utf-8")
