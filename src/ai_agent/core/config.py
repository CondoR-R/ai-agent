import yaml
from pathlib import Path
from typing import Any

config_dict = dict[str, Any]

def load_config(path: str | Path) -> config_dict:
    """Получение словаря конфигурации из yaml файла
    
    params:
        path - путь ка файлу

    return: словарь из yaml файла
    """
    with open(path, 'r', encoding='utf-8') as f:
        y = yaml.safe_load(f)
    return y


def merge_configs(f_dict: config_dict, cli_dict: config_dict) -> config_dict:
    """Объединение словарей конфигурации из файла и CLI
    
    params:
        f_dict - словарь из YAML файла
        cli_dict - словарь из коммандной строки

    return: объединенный словарь
    """
    res_dict = f_dict.copy()
    for key, val in cli_dict.items():
        res_dict[key] = val
    return res_dict


def validate_config(
        config: config_dict,
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
    
    param:
        config - словарь конфигурации
        fields - кортеж обязательных полей, default = (
            "experiment",
            "split_seed",
            "seed",
            "model",
            "train",
            "runtime"
        )
    """
    # проверка на присутвие обязательных полей
    for f in fields:
        if f not in config:
            raise ValueError(f'Отсутствует обязательное поле: {f}')

    # проверка на lr > 0
    if 'experiment' in fields:
        n_lr = str(config['experiment']).removeprefix('lr')
        if not n_lr.isdigit() or int(n_lr) < 0 :
            raise ValueError('номер lr в experiment должен быть неотрицательным числом')


def save_config(path: str | Path, config: config_dict) -> None:
    """Сохранение YAML файла по указанному пути
    
    params:
        path - путь сохранения
        config - словарь конфигурации
    """
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f)
