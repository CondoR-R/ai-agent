"""Эксперимент 1 ЛР3: Dense, BM25 и RRF."""

from pathlib import Path

from ai_agent import cli, files, run
from ai_agent.labs.lr3 import provenance
from ai_agent.labs.lr3.exp1 import exp1_lab


def main(config_path: Path, cli_cfg: dict, exp: int, series: str) -> Path:
    print(f"[EXP1] Конфиг: {config_path}, серия: {series}", flush=True)
    return run.run(
        config_path,
        cli_cfg,
        exp,
        series,
        provenance.input_sha256(files.read_yaml(config_path)),
        exp1_lab,
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
