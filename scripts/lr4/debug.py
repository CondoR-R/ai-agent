"""Отладочный запуск ЛР4 без проверки чистого git-дерева."""

import argparse
import time
from pathlib import Path

from ai_agent import files, metadata
from ai_agent.labs.lr4 import provenance
from ai_agent.labs.lr4.runner import experiment_lab


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lr4/exp.yaml")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--modes", nargs="+", default=["react", "plan"])
    parser.add_argument("--series", default="dev")
    args = parser.parse_args()

    cfg = files.read_yaml(Path(args.config))
    cfg["run"]["split"] = args.split
    cfg["run"]["modes"] = args.modes
    run_id = f"lr4-debug-{args.split}-{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = files.RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    files.write_yaml(run_dir / files.CONFIG_FILENAME, cfg)
    metadata.init(
        run_dir,
        run_id,
        args.series,
        cfg,
        ("mlflow", "agentdojo", "openai"),
        provenance.input_sha256(cfg),
    )
    experiment_lab(cfg, run_dir)
    metadata.update(run_dir, "completed")
    print(f"[debug] run_dir={run_dir}")


if __name__ == "__main__":
    main()
