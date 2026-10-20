"""Утилиты для работы с git"""

import subprocess


def check_clean_status() -> None:
    """Проверка на отсутвие незакоммиченных изменений
    При отсутвии незакомиченных изменений ничего не происходит,
    если есть изменения - вызывается исключение

    raise: SystemExit с сообщением о незакоммиченных файлах
    """
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True
    ).stdout.strip()
    if dirty:
        raise SystemExit(
            "Есть незакоммиченные изменения — итоговый запуск невозможен:\n"
            + dirty
        )


def get_commit_hash() -> str:
    """Получение хэша коммита

    return: хэш коммита
    """
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    return commit
