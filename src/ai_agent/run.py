"""Один запуск эксперимента"""

from collections.abc import Callable
from pathlib import Path

from . import files, reproducibility, utils


def run(
    config_path: Path,
    cli_cfg: dict,
    exp: int,
    sha256: dict[str, str],
    lab_fn: Callable[[dict, Path, bool], None],
    eval_test: bool,
):
    """Запуск эксперимента

    params:
        config_path - путь к файлу конфигурации
        cli_cfg - словарь с переопрелениями конфигурации из CLI
        exp - номер эксперимента
        sha256 - словарь с sha256 файлов
        lab_fn - функция запуска лабораторной
        eval_test - флаг тестовых данных
    """
    # проверка чистоты дерева
    utils.check_git()

    # конфигурация
    file_cfg = files.read_yaml(config_path)
    cfg = utils.merge_cfgs(file_cfg, cli_cfg)

    # создание директории <run_id>/
    run_id = files.get_run_id(cfg["experiment"], exp)
    run_dir = files.create_run_dir(run_id)

    # сохранение конфигурации
    files.write_yaml(run_dir / files.CONFIG_FILENAME, cfg)

    # фиксация воспроизводимости
    reproducibility.set_seed(cfg["seed"], cfg["runtime"]["num_threads"])

    # начальное состояние metadata.json
    m = utils.get_metadata(
        run_id,
        "test",
        cfg["runtime"]["num_threads"],
        cfg["seed"],
        ["torch", "scikit-learn", "optuna", "mlflow"],
        sha256,
    )
    files.write_json(run_dir / files.METADATA_FILENAME, m)

    try:
        lab_fn(cfg, run_dir, eval_test)
    except BaseException as ex:
        raise Exception from ex

    # добавить обновление статуса в metadata
    # округление до 4 знаков после запятой при записи
