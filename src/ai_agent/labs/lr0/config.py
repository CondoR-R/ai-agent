from pathlib import Path

import yaml

from ai_agent import paths


def load(path: Path) -> dict:
    """Чтение YAML файла

    params:
        path - путь к файлу
    return: словарь с конфигурацией"""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_cfg_path(filename: Path | str) -> Path:
    """Получение пути к файлу"""
    return paths.PROJECT_ROOT / "configs" / "lr0" / filename


def merge(file_cfg: dict, cli_cfg: dict) -> dict:
    """Объединение конфигурации файла и CLI

    params:
        file_cfg - словарь конфигурации из файла
        cli_cfg - словарь конфигурации из CLI

    return: словарь объединенной конфигурации
    """
    return {**file_cfg, **cli_cfg}


def save(path: Path, cfg: dict):
    """Сохранение файла конфигурации

    params:
        path - путь к файлу
        cfg - словрь конфигурации
    """
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
