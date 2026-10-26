"""Работа со временем"""

from datetime import datetime


def now(fmt: str = "%Y-%m-%d_%H-%M-%S") -> str:
    """Текущее время в формате ГОД-МЕСЯЦ-ДЕНЬ_ЧАС-МИНУТЫ-СЕКУНДЫ."""
    return datetime.now().strftime(fmt)
