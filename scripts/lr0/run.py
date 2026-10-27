"""Скрипт одного запуска"""

from pathlib import Path

from ai_agent import cli, files, run
from ai_agent.labs.lr0.run import lr0_run


def main(config_path, cli_cfg, exp, series, eval_test):
    split_path = files.DATA_DIR / "split.json"
    sha256 = {"split.json": files.get_sha256(split_path)}

    run_dir = run.run(
        config_path, cli_cfg, exp, series, sha256, lr0_run, eval_test
    )

    return run_dir


if __name__ == "__main__":
    args = cli.parse_args()

    config_path = Path(args.config)
    cli_cfg = cli.get_cli_cfg(args)
    exp = int(args.exp)
    eval_test = args.eval_test
    series = args.series
    main(config_path, cli_cfg, exp, series, eval_test)
