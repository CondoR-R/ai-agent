"""Скрипт одного запуска"""

import argparse
from functools import partial
from pathlib import Path

from ai_agent import cli, files, run
from ai_agent.labs.lr0.run import lr0_run
from ai_agent.labs.lr0.summaries import config_summary, metrics_summary

# аргументы CLI, значения которых копируются в конфиг
CLI_KEYS = ("split_seed",)
# библиотеки ЛР0, версии которых пишутся в metadata.json
LIBS = ("torch", "scikit-learn", "optuna")


def add_args(parser: argparse.ArgumentParser) -> None:
    """Аргументы командной строки, специфичные для ЛР0."""
    parser.add_argument("--split-seed", type=int)
    # включать ли тестовые данные
    parser.add_argument("--eval-test", action="store_true")


def main(
    config_path: Path,
    cli_cfg: dict,
    exp: int,
    series: str,
    eval_test: bool,
) -> Path:
    """Запускает одну конфигурацию ЛР0 и возвращает каталог запуска."""
    split_path = files.DATA_DIR / "split.json"
    sha256 = {"split.json": files.get_sha256(split_path)}
    split_hash = sha256["split.json"][:12]
    overrides = cli_cfg if cli_cfg else "нет"

    print(f"[LR0] Конфиг: {config_path}", flush=True)
    print(
        f"      CLI: {overrides}; split.json sha256={split_hash}…",
        flush=True,
    )

    run_dir = run.run(
        config_path,
        cli_cfg,
        exp,
        series,
        sha256,
        partial(lr0_run, eval_test=eval_test),
        libs=LIBS,
        config_summary=config_summary,
        metrics_summary=metrics_summary,
    )

    return run_dir


if __name__ == "__main__":
    args = cli.parse_args(extra=add_args)

    config_path = Path(args.config)
    cli_cfg = cli.get_cli_cfg(args, keys=CLI_KEYS)
    exp = int(args.exp)
    eval_test = args.eval_test
    series = args.series
    main(config_path, cli_cfg, exp, series, eval_test)
