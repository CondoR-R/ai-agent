"""Данные SciFact: релиз, метки, разбиение dev/holdout и манифест."""

import hashlib
import json
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

from ai_agent import files, utils

SCIFACT_URL = (
    "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
)
ARCHIVE_SHA256 = (
    "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"
)
MANIFEST_KEY = "scifact"

ARCHIVE_REL = "data/raw/scifact-data.tar.gz"
CORPUS_REL = "data/lr2/corpus.jsonl"
CLAIMS_TRAIN_REL = "data/lr2/claims_train.jsonl"
CLAIMS_DEV_REL = "data/lr2/claims_dev.jsonl"
INDEX_REL = "data/lr2/index/corpus_embeddings.npy"
DOC_IDS_REL = "data/lr2/index/doc_ids.json"

LABELS = ("SUPPORT", "CONTRADICT", "NOINFO")
PER_CLASS_DEV = 30

ARCHIVE_PATH = files.PROJECT_ROOT / ARCHIVE_REL
CORPUS_PATH = files.PROJECT_ROOT / CORPUS_REL
CLAIMS_TRAIN_PATH = files.PROJECT_ROOT / CLAIMS_TRAIN_REL
CLAIMS_DEV_PATH = files.PROJECT_ROOT / CLAIMS_DEV_REL
INDEX_PATH = files.PROJECT_ROOT / INDEX_REL
DOC_IDS_PATH = files.PROJECT_ROOT / DOC_IDS_REL

_EXTRACT = {
    "data/corpus.jsonl": CORPUS_PATH,
    "data/claims_train.jsonl": CLAIMS_TRAIN_PATH,
    "data/claims_dev.jsonl": CLAIMS_DEV_PATH,
}


def read_manifest() -> dict:
    """Читает data/manifest.json; пустой словарь, если файла ещё нет."""
    if not files.MANIFEST_PATH.exists():
        return {}
    manifest = files.read_json(files.MANIFEST_PATH)
    if not isinstance(manifest, dict):
        raise ValueError("Ожидался словарь {имя набора: описание}")
    return manifest


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def claim_label(claim: dict) -> str | None:
    """Метка утверждения; None — если в разметке обе метки сразу."""
    labels = {
        record["label"]
        for records in claim["evidence"].values()
        for record in records
    }
    if not claim["evidence"]:
        return "NOINFO"
    return labels.pop() if len(labels) == 1 else None


def split_dev(claims: list[dict], per_class: int = PER_CLASS_DEV) -> tuple:
    """Делит claims_dev на dev и holdout по SHA-256 строки scifact:<id>."""
    by_class: dict[str, list[int]] = {}
    for claim in claims:
        label = claim_label(claim)
        if label is not None:
            by_class.setdefault(label, []).append(claim["id"])

    dev: list[int] = []
    holdout: list[int] = []
    for label in LABELS:
        ids = sorted(
            by_class[label],
            key=lambda claim_id: hashlib.sha256(
                f"scifact:{claim_id}".encode()
            ).hexdigest(),
        )
        dev.extend(ids[:per_class])
        holdout.extend(ids[per_class:])
    return dev, holdout


def label_counts(claims: list[dict]) -> dict[str, int]:
    """Сколько утверждений каждого класса; None-метки не входят."""
    counts = Counter(claim_label(claim) for claim in claims)
    counts.pop(None, None)
    return {label: counts[label] for label in LABELS}


def _verify_archive(path: Path) -> None:
    actual = files.get_sha256(path)
    if actual != ARCHIVE_SHA256:
        raise ValueError(
            f"SHA-256 архива SciFact не совпал: {actual} != {ARCHIVE_SHA256}"
        )


def _download_archive(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        SCIFACT_URL, headers={"User-Agent": "ai-agent-lab/1.0"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        path.write_bytes(response.read())


def _extract_files(archive_path: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as archive:
        for member_name, target in _EXTRACT.items():
            member = archive.getmember(member_name)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"В архиве нет файла {member_name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read())


def _file_entry(path: Path, rel: str, n_records: int | None = None) -> dict:
    data = path.read_bytes()
    entry = {"sha256": _sha256_bytes(data), "n_bytes": len(data)}
    if n_records is not None:
        entry["n_records"] = n_records
    return entry


def prepare(archive_path: Path | None = None) -> dict:
    """Готовит corpus/claims и записывает набор в data/manifest.json.

    Если архив и файлы уже есть, только проверяет SHA-256 и возвращает
    прежнюю запись: манифест не меняется, рабочее дерево остаётся чистым.
    """
    manifest = read_manifest()
    entry = manifest.get(MANIFEST_KEY)
    archive = archive_path or ARCHIVE_PATH

    if not archive.exists():
        if archive_path is not None:
            raise FileNotFoundError(f"Нет архива {archive}")
        _download_archive(archive)
    _verify_archive(archive)

    all_present = all(path.exists() for path in _EXTRACT.values())
    if not all_present:
        _extract_files(archive)

    corpus = _read_jsonl(CORPUS_PATH)
    claims_train = _read_jsonl(CLAIMS_TRAIN_PATH)
    claims_dev = _read_jsonl(CLAIMS_DEV_PATH)
    dev_ids, holdout_ids = split_dev(claims_dev)

    hashes = {
        CORPUS_REL: _file_entry(CORPUS_PATH, CORPUS_REL, len(corpus)),
        CLAIMS_TRAIN_REL: _file_entry(
            CLAIMS_TRAIN_PATH, CLAIMS_TRAIN_REL, len(claims_train)
        ),
        CLAIMS_DEV_REL: _file_entry(
            CLAIMS_DEV_PATH, CLAIMS_DEV_REL, len(claims_dev)
        ),
    }

    if entry is not None:
        for rel, expected in hashes.items():
            actual = entry["files"][rel]["sha256"]
            if actual != expected["sha256"]:
                raise ValueError(
                    f"SHA-256 {rel} не совпал с манифестом: "
                    f"{expected['sha256']} != {actual}"
                )
        return entry

    entry = {
        "source": "SciFact: Scientific Fact-Checking Dataset",
        "url": SCIFACT_URL,
        "version": "release/latest; архив от 2020-12-18",
        "license": (
            "SciFact release, Allen Institute for AI; условия использования "
            "см. в репозитории allenai/scifact"
        ),
        "archive": {
            "path": ARCHIVE_REL,
            "sha256": ARCHIVE_SHA256,
            "n_bytes": archive.stat().st_size,
        },
        "files": hashes,
        "labels": {
            "dev": label_counts(claims_dev),
            "train": label_counts(claims_train),
        },
        "split": {
            "rule": (
                "claims_dev.jsonl делится до опытов: в каждом классе "
                "утверждения упорядочены по SHA-256 строки scifact:<id>; "
                "первые 30 — dev, остальные — holdout"
            ),
            "per_class": PER_CLASS_DEV,
            "dev": dev_ids,
            "holdout": holdout_ids,
        },
        "prepared_at": utils.time_now("%Y-%m-%d"),
    }
    manifest[MANIFEST_KEY] = entry
    files.write_json(files.MANIFEST_PATH, manifest)
    return entry


def load_entry() -> dict:
    """Запись SciFact из манифеста."""
    entry = read_manifest().get(MANIFEST_KEY)
    if entry is None:
        raise ValueError(
            f"В манифесте нет {MANIFEST_KEY}: запустите scripts/lr2/prepare_data.py"
        )
    return entry


def load_verified_jsonl(rel: str) -> list[dict]:
    """Читает JSONL, сверяя SHA-256 прочитанных байтов с манифестом."""
    path = files.PROJECT_ROOT / rel
    data = path.read_bytes()
    expected = load_entry()["files"][rel]["sha256"]
    actual = _sha256_bytes(data)
    if actual != expected:
        raise ValueError(
            f"SHA-256 {rel} не совпал с манифестом: {actual} != {expected}"
        )
    return [
        json.loads(line) for line in data.decode("utf-8").splitlines() if line
    ]


def load_corpus() -> list[dict]:
    """Корпус в порядке файла: 5183 аннотации."""
    return load_verified_jsonl(CORPUS_REL)


def load_claims_train() -> list[dict]:
    """Утверждения train: только для проверки форматов и прогрева."""
    return load_verified_jsonl(CLAIMS_TRAIN_REL)


def load_claims_dev() -> list[dict]:
    """Все 300 утверждений dev: используется только для разбиения."""
    return load_verified_jsonl(CLAIMS_DEV_REL)


def dev_claims(claims: list[dict] | None = None) -> list[dict]:
    """90 утверждений dev в порядке манифеста (30 на класс)."""
    claims = claims or load_claims_dev()
    by_id = {claim["id"]: claim for claim in claims}
    return [by_id[claim_id] for claim_id in load_entry()["split"]["dev"]]


def holdout_claims(claims: list[dict] | None = None) -> list[dict]:
    """210 утверждений holdout: в ЛР2 не используются."""
    claims = claims or load_claims_dev()
    by_id = {claim["id"]: claim for claim in claims}
    return [by_id[claim_id] for claim_id in load_entry()["split"]["holdout"]]


def document_text(document: dict) -> str:
    """Текст документа для кодировщика и контекста."""
    title = str(document["title"]).strip()
    abstract = " ".join(str(s) for s in document["abstract"])
    return f"{title}\n{abstract}".strip()


def corpus_by_id(corpus: list[dict] | None = None) -> dict[str, dict]:
    """Словарь документов по строковому doc_id."""
    corpus = corpus or load_corpus()
    return {str(document["doc_id"]): document for document in corpus}


def save_split(claims: list[dict], dev: list[int], holdout: list[int]) -> None:
    """Сохраняет оба списка отдельным файлом для чтения отчётными скриптами."""
    by_id = {claim["id"]: claim for claim in claims}
    payload = {
        "dev": [by_id[claim_id] for claim_id in dev],
        "holdout": [by_id[claim_id] for claim_id in holdout],
    }
    files.write_json(files.DATA_DIR / "lr2" / "split.json", payload)
