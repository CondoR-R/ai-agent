"""Эксперимент 3 ЛР3: holdout после фиксации selection.json."""

from pathlib import Path

from ai_agent import cli, files, run
from ai_agent.labs.lr3 import provenance
from ai_agent.labs.lr3.holdout import holdout_lab


def main(config_path: Path, cli_cfg: dict, exp: int, series: str) -> Path:
    cfg = files.read_yaml(config_path)
    selection_path = files.PROJECT_ROOT / cfg["selection_path"]
    print(f"[EXP3] Конфиг: {config_path}, серия: {series}", flush=True)
    return run.run(
        config_path,
        cli_cfg,
        exp,
        series,
        provenance.input_sha256(cfg, selection_path=selection_path),
        holdout_lab,
        libs=("sentence-transformers", "transformers", "openai", "torch"),
    )


if __name__ == "__main__":
    args = cli.parse_args()
    main(
        Path(args.config),
        cli.get_cli_cfg(args),
        int(args.exp),
        args.series,
    )
