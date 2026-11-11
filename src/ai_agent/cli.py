"""Работа с интерфейсом командной строки"""

import argparse
from collections.abc import Callable, Sequence

import yaml


def parse_args(
    extra: Callable[[argparse.ArgumentParser], None] | None = None,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Обработка аргументов командной строки

    params:
        extra - функция, добавляющая аргументы конкретной лабораторной:
                extra(parser) вызывает parser.add_argument(...)
        argv - список аргументов (по умолчанию sys.argv[1:])

    return: разобранные аргументы
    """
    parser = argparse.ArgumentParser()
    # номер эксперимента
    parser.add_argument("--exp", default=0, type=int)

    # серия эксперимента
    parser.add_argument("--series", default="base")

    # путь к конфигу
    parser.add_argument("--config", required=True)

    # переопределение конфига
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="КЛЮЧ=ЗНАЧЕНИЕ",
        help="переопределить поле конфига, вложенные через точку: "
        "--set generation.max_tokens=64 (можно повторять)",
    )

    if extra is not None:
        extra(parser)

    return parser.parse_args(argv)


def _parse_override(item: str) -> tuple[list[str], object]:
    """Разбирает 'a.b=значение' в (['a', 'b'], значение).

    Значение читается как YAML: 64 -> int, 0.5 -> float, true -> bool.
    """
    key, sep, raw = item.partition("=")
    if not sep or not key:
        raise SystemExit(f"--set ожидает КЛЮЧ=ЗНАЧЕНИЕ, получено: {item!r}")
    return key.split("."), yaml.safe_load(raw)


def get_cli_cfg(args: argparse.Namespace, keys: tuple[str, ...] = ()) -> dict:
    """Собирает переопределения конфигурации из аргументов CLI

    params:
        args - разобранные аргументы
        keys - имена дополнительных аргументов лабораторной, значения
               которых копируются в конфиг, если заданы
               (например, ("split_seed",) для ЛР0)

    return: словарь переопределений; порядок приоритета:
            --set > --seed > keys
    """
    cfg: dict = {}

    for key in keys:
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value

    if args.seed is not None:
        cfg["seed"] = int(args.seed)

    for item in args.overrides:
        parts, value = _parse_override(item)
        node = cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise SystemExit(f"--set: {part!r} уже задан как значение")
        node[parts[-1]] = value

    return cfg
