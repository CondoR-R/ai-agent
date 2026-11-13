"""Эксперимент 2 ЛР1: длина ответа и скорость генерации.

Запуск (сервер llama.cpp уже поднят, дерево git чистое):
    python scripts/lr1/experiment_2.py --config configs/lr1/exp2.yaml \
        --series output_length --exp 2
"""

from pathlib import Path

from ai_agent import cli, files, run
from ai_agent.labs.lr1 import data
from ai_agent.labs.lr1.exp2 import exp2_lab
from ai_agent.labs.lr1.exp2_analysis import config_summary, metrics_summary


def main(config_path: Path, cli_cfg: dict, exp: int, series: str) -> Path:
    """Запускает серию опыта 2 и возвращает каталог запуска."""
    sha256 = {
        files.ENDPOINT_PATH.name: files.get_sha256(files.ENDPOINT_PATH),
        data.TEXT_REL: files.get_sha256(data.TEXT_PATH),
    }
    print(f"[EXP2] Конфиг: {config_path}, серия: {series}", flush=True)
    return run.run(
        config_path,
        cli_cfg,
        exp,
        series,
        sha256,
        exp2_lab,
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
