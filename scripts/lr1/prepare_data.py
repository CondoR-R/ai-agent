"""Подготовка данных ЛР1: текст «Pride and Prejudice» и манифест.

Первый раз запускается там, где удобно (например, на Mac): скачивает
текст, записывает набор в data/manifest.json — манифест нужно закоммитить.
На Colab тот же скрипт скачивает текст заново и сверяет SHA-256 с
манифестом, ничего не меняя в репозитории.
"""

import argparse
from pathlib import Path

from ai_agent.labs.lr1 import data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-file",
        default=None,
        help="готовый файл Gutenberg вместо скачивания",
    )
    args = parser.parse_args()

    source = Path(args.source_file) if args.source_file else None
    entry = data.prepare_text(source)
    info = entry["files"][data.TEXT_REL]

    print(f"[DATA] {data.TEXT_REL}", flush=True)
    print(
        f"       символов={info['n_chars']}, sha256={info['sha256'][:16]}…",
        flush=True,
    )
    print("[OK] текст готов, хеш совпадает с манифестом", flush=True)


if __name__ == "__main__":
    main()
