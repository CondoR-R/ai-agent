"""Эксперимент 3 ЛР1: нагрузка, пропускная способность и goodput.

Запуск (сервер vLLM уже поднят, дерево git чистое):
    python scripts/lr1/experiment_3.py --config configs/lr1/exp3.yaml \
        --series load --exp 3
"""

from pathlib import Path

from ai_agent import cli, files, run
from ai_agent.labs.lr1 import data
from ai_agent.labs.lr1.exp3 import exp3_lab
from ai_agent.labs.lr1.exp3_analysis import config_summary, metrics_summary


def main(config_path: Path, cli_cfg: dict, exp: int, series: str) -> Path:
    """Запускает серию опыта 3 и возвращает каталог запуска."""
    sha256 = {
        files.ENDPOINT_PATH.name: files.get_sha256(files.ENDPOINT_PATH),
        data.TEXT_REL: files.get_sha256(data.TEXT_PATH),
    }
    print(f"[EXP3] Конфиг: {config_path}, серия: {series}", flush=True)
    return run.run(
        config_path,
        cli_cfg,
        exp,
        series,
        sha256,
        exp3_lab,
        config_summary=config_summary,
        metrics_summary=metrics_summary,
    )


if __name__ == "__main__":
    args = cli.parse_args()
    main(
        Path(args.config),
        cli.get_cli_cfg(args),
        int(args.exp),
        args.series,
    )
