"""Основной эксперимент ЛР4: AgentDojo workspace, 30 задач × 2 режима."""

from pathlib import Path

from ai_agent import cli, files, run
from ai_agent.labs.lr4 import provenance
from ai_agent.labs.lr4.runner import experiment_lab


def main(config_path: Path, cli_cfg: dict, exp: int, series: str) -> Path:
    print(f"[LR4] Конфиг: {config_path}, серия: {series}", flush=True)
    return run.run(
        config_path,
        cli_cfg,
        exp,
        series,
        provenance.input_sha256(files.read_yaml(config_path)),
        experiment_lab,
        libs=("agentdojo", "openai"),
    )


if __name__ == "__main__":
    args = cli.parse_args()
    main(Path(args.config), cli.get_cli_cfg(args), int(args.exp), args.series)
