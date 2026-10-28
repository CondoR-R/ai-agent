"""Единый формат сообщений в терминале."""

TOTAL_STEPS = 8


def step(number: int, title: str) -> None:
    """Печатает заголовок шага запуска."""
    print(f"[{number}/{TOTAL_STEPS}] {title}", flush=True)


def info(message: str) -> None:
    """Печатает подробность шага с отступом."""
    print(f"      {message}", flush=True)


def ok(message: str) -> None:
    """Печатает итог успешного запуска."""
    print(f"[OK] {message}", flush=True)


def error(message: str) -> None:
    """Печатает сообщение об ошибке."""
    print(f"[ERROR] {message}", flush=True)
