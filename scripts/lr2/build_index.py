"""Построение плотного индекса SciFact и паспорта в манифесте."""

import argparse
from pathlib import Path

from ai_agent import files
from ai_agent.labs.lr2 import data, index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = files.read_yaml(Path(args.config))
    data.prepare()
    entry = index.build(cfg)
    print(
        f"[INDEX] {entry['model']} rev={entry['revision']}, "
        f"{entry['dimension']}d, max={entry['max_seq_length']}"
    )
    print(
        f"[INDEX] обрезано аннотаций: {entry['clipped_fraction']:.1%}, "
        f"матрица={entry['matrix']['shape']}, "
        f"sha256={entry['matrix']['sha256'][:16]}…"
    )


if __name__ == "__main__":
    main()
