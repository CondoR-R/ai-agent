"""Утилиты для работы с git"""

import subprocess

from .paths import PROJECT_ROOT


def _run_git(*args: str) -> str:
    """Запускает git в корне проекта и возвращает stdout

    params:
        args - аргументы команды git

    return: stdout без пробелов по краям

    raise:
        RuntimeError, если git не найден или команда завершилась с ошибкой
    """
    try:
        proc = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
    except FileNotFoundError as err:
        raise RuntimeError("git не найден в PATH") from err

    if proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} завершился с ошибкой: {proc.stderr.strip()}"
        )
    return proc.stdout.strip()


def check_clean_status() -> None:
    """Проверка на отсутствие незакоммиченных изменений

    raise:
        SystemExit с перечнем файлов, если есть изменения
        RuntimeError, если git недоступен или это не репозиторий
    """
    dirty = _run_git("status", "--porcelain")
    if dirty:
        raise SystemExit(
            "Есть незакоммиченные изменения — итоговый запуск невозможен:\n"
            + dirty
        )


def get_commit_hash() -> str:
    """Получение полного хэша коммита

    return: хэш коммита
    """
    return _run_git("rev-parse", "HEAD")
