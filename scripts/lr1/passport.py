"""Паспорт сервера и проверки реализации ЛР1 (п. 3.1 и 3.4).

Запуск (сервер llama.cpp уже поднят, дерево git чистое):
    python scripts/lr1/passport.py --config configs/lr1/passport.yaml \
        --series passport --exp 0
"""

import shutil
from pathlib import Path

from ai_agent import cli, files, run
from ai_agent.labs.lr1 import data
from ai_agent.labs.lr1.passport_lab import passport_lab


def main(config_path: Path, cli_cfg: dict, exp: int, series: str) -> Path:
    """Запускает паспорт и возвращает каталог запуска."""
    sha256 = {
        files.ENDPOINT_PATH.name: files.get_sha256(files.ENDPOINT_PATH),
        data.TEXT_REL: files.get_sha256(data.TEXT_PATH),
    }
    print(f"[LR1] Конфиг: {config_path}", flush=True)
    run_dir = run.run(config_path, cli_cfg, exp, series, sha256, passport_lab)
    reports_dir = files.PROJECT_ROOT / "reports" / "lr1"
    reports_dir.mkdir(parents=True, exist_ok=True)
    target = reports_dir / "passport.json"
    shutil.copyfile(run_dir / "passport.json", target)
    print(f"[LR1] Паспорт скопирован: {target}", flush=True)
    return run_dir


if __name__ == "__main__":
    args = cli.parse_args()
    main(
        Path(args.config),
        cli.get_cli_cfg(args),
        int(args.exp),
        args.series,
    )
