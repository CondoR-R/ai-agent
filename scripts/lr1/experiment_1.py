"""Эксперимент 1 ЛР1: одна серия за запуск (у серий разный режим сервера).

Серия 1 (сервер llama.cpp с --no-cache-prompt):
    python scripts/lr1/experiment_1.py --config configs/lr1/exp1_length.yaml \
        --series ttft_vs_length --exp 1
Серия 2 (сервер llama.cpp с кешем промпта по умолчанию):
    python scripts/lr1/experiment_1.py --config configs/lr1/exp1_cache.yaml \
        --series prefix_cache --exp 1
"""

from pathlib import Path

from ai_agent import cli, files, run
from ai_agent.labs.lr1 import data
from ai_agent.labs.lr1.exp1 import exp1_lab
from ai_agent.labs.lr1.exp1_analysis import config_summary, metrics_summary


def main(config_path: Path, cli_cfg: dict, exp: int, series: str) -> Path:
    """Запускает серию опыта 1 и возвращает каталог запуска."""
    sha256 = {
        files.ENDPOINT_PATH.name: files.get_sha256(files.ENDPOINT_PATH),
        data.TEXT_REL: files.get_sha256(data.TEXT_PATH),
    }
    print(f"[EXP1] Конфиг: {config_path}, серия: {series}", flush=True)
    return run.run(
        config_path,
        cli_cfg,
        exp,
        series,
        sha256,
        exp1_lab,
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
