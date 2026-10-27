"""Работа с интерфейсом командной строки"""

import argparse


def parse_args() -> argparse.Namespace:
    """Обработка аргументов командной строки"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed")
    parser.add_argument("--split-seed")
    parser.add_argument("--eval-test", action="store_true")

    return parser.parse_args()


def get_cli_cfg(args: argparse.Namespace) -> dict:
    cfg = {}
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.split_seed is not None:
        cfg["split_seed"] = int(args.split_seed)

    return cfg
