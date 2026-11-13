"""Эксперимент 1 ЛР2: LLM-only, RAG k=3 и oracle.

Запуск (сервер llama.cpp поднят, дерево git чистое, индекс построен):
    python scripts/lr2/experiment_1.py --config configs/lr2/exp1.yaml \
        --series main --exp 1
"""

from pathlib import Path

from ai_agent import cli, files, run
from ai_agent.labs.lr2 import data, index
from ai_agent.labs.lr2.exp1 import exp1_lab


def main(config_path: Path, cli_cfg: dict, exp: int, series: str) -> Path:
    sha256 = {
        files.ENDPOINT_PATH.name: files.get_sha256(files.ENDPOINT_PATH),
        files.MANIFEST_PATH.name: files.get_sha256(files.MANIFEST_PATH),
        data.CORPUS_REL: files.get_sha256(data.CORPUS_PATH),
        data.CLAIMS_DEV_REL: files.get_sha256(data.CLAIMS_DEV_PATH),
        data.INDEX_REL: files.get_sha256(index.save_path()),
        data.DOC_IDS_REL: files.get_sha256(data.DOC_IDS_PATH),
    }
    print(f"[EXP1] Конфиг: {config_path}, серия: {series}", flush=True)
    return run.run(
        config_path,
        cli_cfg,
        exp,
        series,
        sha256,
        exp1_lab,
        libs=("sentence-transformers", "transformers", "openai"),
    )


if __name__ == "__main__":
    args = cli.parse_args()
    main(
        Path(args.config),
        cli.get_cli_cfg(args),
        int(args.exp),
        args.series,
    )
