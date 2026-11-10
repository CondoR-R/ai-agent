"""Утилиты проекта"""

import platform
import subprocess
from datetime import datetime
from importlib.metadata import version


def time_now(fmt: str = "%Y-%m-%d_%H-%M-%S") -> str:
    """Текущее время в формате ГОД-МЕСЯЦ-ДЕНЬ_ЧАС-МИНУТЫ-СЕКУНДЫ."""
    return datetime.now().strftime(fmt)


def _git(*args: str) -> str:
    """Запускает git и возвращает stdout.

    Ошибка git (нет репозитория, нет коммитов, нет самого git) —
    это SystemExit, а не пустая строка: иначе в metadata.json
    молча попал бы пустой коммит.
    """
    cmd = ["git", *args]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise SystemExit("git не найден в PATH") from None
    if result.returncode != 0:
        raise SystemExit(
            f"{' '.join(cmd)} завершился с кодом {result.returncode}: "
            f"{result.stderr.strip()}"
        )
    return result.stdout.strip()


def check_git() -> None:
    """Проверка чистоты рабочего дерева"""
    dirty = _git("status", "--porcelain")
    if dirty:
        raise SystemExit(
            "Есть незакоммиченные изменения — итоговый запуск невозможен:\n"
            + dirty
        )


def get_commit_hash() -> str:
    """Получение хеша коммита"""
    return _git("rev-parse", "HEAD")


def merge_cfgs(file_cfg: dict, cli_cfg: dict) -> dict:
    """Объединение конфигурации файла и CLI.

    Слияние глубокое: переопределение train.lr не затирает остальные
    поля секции train.

    params:
        file_cfg - словарь конфигурации из файла
        cli_cfg - словарь конфигурации из CLI

    return: словарь объединенной конфигурации
    """
    merged = dict(file_cfg)
    for key, value in cli_cfg.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_cfgs(merged[key], value)
        else:
            merged[key] = value
    return merged


def check_overrides(file_cfg: dict, cli_cfg: dict, prefix: str = "") -> None:
    """Проверяет, что CLI переопределяет только существующие поля.

    Без этой проверки опечатка (--set train.lrr=1) молча создала бы
    новое поле, а запуск прошёл бы с прежним lr.

    raise:
        ValueError с путём неизвестного поля
    """
    for key, value in cli_cfg.items():
        path = f"{prefix}{key}"
        if key not in file_cfg:
            raise ValueError(f"Неизвестный параметр конфигурации: {path}")
        if isinstance(value, dict) and isinstance(file_cfg[key], dict):
            check_overrides(file_cfg[key], value, f"{path}.")


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
