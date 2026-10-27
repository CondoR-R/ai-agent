"""Утилиты проекта"""

import platform
import subprocess
from datetime import datetime
from importlib.metadata import version


def time_now(fmt: str = "%Y-%m-%d_%H-%M-%S") -> str:
    """Текущее время в формате ГОД-МЕСЯЦ-ДЕНЬ_ЧАС-МИНУТЫ-СЕКУНДЫ."""
    return datetime.now().strftime(fmt)


def check_git() -> None:
    """Проверка чистоты рабочего дерева"""
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True
    ).stdout.strip()
    if dirty:
        raise SystemExit(
            "Есть незакоммиченные изменения — итоговый запуск невозможен:\n"
            + dirty
        )


def get_commit_hash() -> str:
    """Получение хеша коммита"""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()


def merge_cfgs(file_cfg: dict, cli_cfg: dict) -> dict:
    """Объединение конфигурации файла и CLI

    params:
        file_cfg - словарь конфигурации из файла
        cli_cfg - словарь конфигурации из CLI

    return: словарь объединенной конфигурации
    """
    return {**file_cfg, **cli_cfg}


def get_versions(libs: tuple[str, ...]) -> dict[str, str]:
    """Получение версий Python и библиотек

    params:
        libs - список библиотек
    return: словарь с версиями
    """
    v = {
        "python": platform.python_version(),
    }
    for lib in libs:
        v[lib] = version(lib)
    return v
