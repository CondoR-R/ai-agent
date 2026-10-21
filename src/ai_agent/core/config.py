"""Файл config.yaml в runs/<run_id>/"""

import copy
from pathlib import Path
from typing import Any

import yaml

ConfigDict = dict[str, Any]


def load(path: str | Path) -> ConfigDict:
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


def merge(f_dict: ConfigDict, cli_dict: ConfigDict) -> ConfigDict:
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


def validate(
    config: ConfigDict,
    fields: tuple[str, ...] = ("experiment", "seed"),
) -> None:
    """Проверка присутствия обязательных полей

    params:
        config - словарь конфигурации
        fields - кортеж обязательных полей, default = ("experiment", "seed")

    raise:
        ValueError, при отсутствии обязательных полей
    """
    for f in fields:
        if f not in config:
            raise ValueError(f"Отсутствует обязательное поле: {f}")


def save(path: str | Path, config: ConfigDict) -> None:
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
