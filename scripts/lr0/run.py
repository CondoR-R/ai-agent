"""Скрипт экспериментов"""
import argparse
from pathlib import Path

from ai_agent.labs.lr0 import config
from ai_agent import paths, git_utils, reproducibility


def parse_args() -> argparse.Namespace:
    """Обработка аргументов командной строки"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--seed')
    parser.add_argument('--split-seed')
    parser.add_argument('--eval-test', action="store_true")

    return parser.parse_args()


def get_cli_cfg(args: argparse.Namespace) -> dict:
    cfg = {}
    if args.seed is not None:
        cfg['seed'] = int(args.seed)
    if args.split_seed is not None:
        cfg['split_seed'] = int(args.split_seed)

    return cfg


def main(config_path: Path, cli_cfg: dict, eval_test: bool, experiment: int):
    # проверка чистоты дерева
    git_utils.check_git()

    # конфигурация
    file_cfg = config.load(config_path)
    cfg = config.merge(file_cfg, cli_cfg)

    # фиксация воспроизводимости
    reproducibility.set_seed(cfg['seed'], cfg['runtime']['num_threads'])

    # создание директории <run_id>/
    run_id = paths.get_run_id(cfg['experiment'], experiment)
    run_dir = paths.create_run_dir(run_id)

    # сохранение конфигурации
    config.save(run_dir / paths.CONFIG_FILENAME, cfg)

    # получение данных
    # обучение модели
    # запись в файлы


if __name__ == "__main__":
    args = parse_args()
    config_path = Path(args.config)
    cli_cfg = get_cli_cfg(args)
    eval_test = args.eval_test
    main(config_path, cli_cfg, eval_test, 0)
