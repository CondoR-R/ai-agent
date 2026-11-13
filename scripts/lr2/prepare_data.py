"""Подготовка SciFact: архив, corpus/claims, метки и разбиение dev/holdout."""

import argparse
from pathlib import Path

from ai_agent.labs.lr2 import data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        default=None,
        help="готовый data.tar.gz вместо скачивания",
    )
    args = parser.parse_args()

    entry = data.prepare(Path(args.archive) if args.archive else None)
    claims = data.load_claims_dev()
    data.save_split(claims, entry["split"]["dev"], entry["split"]["holdout"])

    print(f"[DATA] corpus={entry['files'][data.CORPUS_REL]['n_records']}")
    print(
        f"[DATA] claims_dev={entry['files'][data.CLAIMS_DEV_REL]['n_records']}, "
        f"labels={entry['labels']['dev']}"
    )
    print(
        f"[DATA] dev={len(entry['split']['dev'])}, "
        f"holdout={len(entry['split']['holdout'])}"
    )
    print("[OK] SciFact готов, хеши совпадают с манифестом")


if __name__ == "__main__":
    main()
