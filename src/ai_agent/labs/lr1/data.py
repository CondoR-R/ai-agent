"""Данные ЛР1: текст «Pride and Prejudice» (Project Gutenberg, № 1342)"""

import hashlib
import re
import urllib.request
from pathlib import Path

from ai_agent import files, utils

GUTENBERG_ID = 1342
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"

TEXT_REL = "data/lr1/pride_and_prejudice.txt"
TEXT_PATH = files.PROJECT_ROOT / TEXT_REL
MANIFEST_KEY = "gutenberg_1342"

_START = re.compile(
    r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*\*\*",
    re.IGNORECASE,
)
_END = re.compile(
    r"\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", re.IGNORECASE
)


def strip_gutenberg(raw: str) -> str:
    """Оставляет только текст книги между маркерами Gutenberg.

    Заголовок и лицензия проекта в запросы попадать не должны.

    raise:
        ValueError, если маркеры не найдены (формат файла изменился)
    """
    raw = raw.replace("\r\n", "\n")
    start = _START.search(raw)
    end = _END.search(raw)
    if start is None or end is None or end.start() <= start.end():
        raise ValueError(
            "Не найдены маркеры START/END Project Gutenberg: "
            "формат файла изменился"
        )
    return raw[start.end() : end.start()].strip() + "\n"


def download(url: str = GUTENBERG_URL) -> str:
    """Скачивает текст (без сторонних библиотек)."""
    request = urllib.request.Request(
        url, headers={"User-Agent": "ai-agent-lab/1.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8-sig")


def read_manifest() -> dict:
    """Читает data/manifest.json (пустой словарь, если его ещё нет)."""
    if not files.MANIFEST_PATH.exists():
        return {}
    manifest = files.read_json(files.MANIFEST_PATH)
    if not isinstance(manifest, dict):
        raise ValueError(
            "Ожидался словарь {имя набора: описание} в manifest.json"
        )
    return manifest


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare_text(source_file: Path | None = None) -> dict:
    """Готовит текст и записывает набор в манифест данных.

    Если файл и запись в манифесте уже есть, только сверяет хеш: манифест
    не меняется, дерево git остаётся чистым (важно на Colab, где текст
    скачивается заново).

    params:
        source_file - готовый «сырой» файл Gutenberg вместо скачивания

    return: запись манифеста

    raise:
        ValueError, если хеш полученного текста не совпал с манифестом
    """
    manifest = read_manifest()
    entry = manifest.get(MANIFEST_KEY)

    if TEXT_PATH.exists():
        body_bytes = TEXT_PATH.read_bytes()
        raw_chars = None
    else:
        if source_file is not None:
            raw = Path(source_file).read_text(encoding="utf-8-sig")
        else:
            raw = download()
        body = strip_gutenberg(raw)
        body_bytes = body.encode("utf-8")
        raw_chars = len(raw)
        TEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
        TEXT_PATH.write_bytes(body_bytes)

    sha = _sha256_bytes(body_bytes)
    n_chars = len(body_bytes.decode("utf-8"))

    if entry is not None:
        expected = entry["files"][TEXT_REL]["sha256"]
        if sha != expected:
            raise ValueError(
                f"SHA-256 текста не совпал с манифестом: {sha} != "
                f"{expected}. Gutenberg мог обновить файл: возьмите "
                "текст, с которым подготовлен манифест (--source-file)"
            )
        return entry

    entry = {
        "source": (
            "Project Gutenberg, ebook #1342: Jane Austen, Pride and Prejudice"
        ),
        "url": GUTENBERG_URL,
        "version": (
            f"получено {utils.time_now('%Y-%m-%d')}; у файла нет версии, "
            "версией служит SHA-256 очищенного текста"
        ),
        "license": "общественное достояние",
        "selection": {
            "rule": (
                "текст между маркерами '*** START OF' и '*** END OF' "
                "Project Gutenberg, переводы строк приведены к LF"
            ),
            "chars_before": raw_chars,
            "chars_after": n_chars,
            "excluded": "заголовок и лицензия Project Gutenberg",
        },
        "files": {
            TEXT_REL: {"sha256": sha, "n_records": 1, "n_chars": n_chars}
        },
    }
    manifest[MANIFEST_KEY] = entry
    files.write_json(files.MANIFEST_PATH, manifest)
    return entry


def load_verified_text(path: Path = TEXT_PATH) -> str:
    """Читает текст, сверяя хеш с манифестом по тем же прочитанным байтам.

    Байты читаются один раз: хешируется ровно то, что идёт в запросы.

    raise:
        ValueError, если набора нет в манифесте или хеш не совпал
    """
    entry = read_manifest().get(MANIFEST_KEY)
    if entry is None:
        raise ValueError(
            f"В манифесте нет {MANIFEST_KEY}: запустите prepare_data.py"
        )
    data = path.read_bytes()
    expected = entry["files"][TEXT_REL]["sha256"]
    actual = _sha256_bytes(data)
    if actual != expected:
        raise ValueError(
            f"SHA-256 {path.name} не совпал с манифестом: "
            f"{actual} != {expected}"
        )
    return data.decode("utf-8")
