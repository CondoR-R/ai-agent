"""Плотный индекс SciFact: матрица эмбеддингов и поиск по косинусу."""

import hashlib
from pathlib import Path

import numpy as np

from ai_agent import files, utils

from . import data

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _sha256_array(matrix: np.ndarray) -> str:
    return hashlib.sha256(matrix.tobytes()).hexdigest()


def _load_model(encoder_cfg: dict):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        encoder_cfg["model"],
        device=encoder_cfg.get("device", "cpu"),
        revision=encoder_cfg.get("revision"),
    )
    if encoder_cfg.get("max_seq_length") is not None:
        model.max_seq_length = int(encoder_cfg["max_seq_length"])
    return model


def _model_revision(model_name: str, revision: str | None) -> str | None:
    if revision is not None:
        return revision
    from huggingface_hub import HfApi

    return HfApi().model_info(model_name).sha


def _clipped_fraction(model, texts: list[str]) -> float:
    max_length = int(model.max_seq_length)
    lengths = []
    for text in texts:
        ids = model.tokenizer(text, add_special_tokens=True, truncation=False)[
            "input_ids"
        ]
        lengths.append(len(ids))
    return sum(length > max_length for length in lengths) / len(texts)


def encode_documents(
    model, texts: list[str], batch_size: int = 32
) -> np.ndarray:
    """Нормализованные эмбеддинги документов, float32."""
    matrix = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
        device="cpu",
    )
    return np.asarray(matrix, dtype=np.float32)


def encode_queries(
    model, texts: list[str], prefix: str, batch_size: int = 32
) -> np.ndarray:
    """Нормализованные эмбеддинги запросов с префиксом кодировщика."""
    matrix = model.encode(
        [f"{prefix}{text}" for text in texts],
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
        device="cpu",
    )
    return np.asarray(matrix, dtype=np.float32)


def search_matrix(
    query_matrix: np.ndarray,
    document_matrix: np.ndarray,
    doc_ids: list[str],
    top_k: int = 5,
) -> list[list[dict]]:
    """Топ-k документов по косинусной близости; строки уже нормализованы."""
    scores = query_matrix @ document_matrix.T
    order = np.argsort(-scores, axis=1)[:, :top_k]
    result = []
    for row, indices in zip(scores, order, strict=True):
        result.append(
            [
                {
                    "doc_id": doc_ids[int(index)],
                    "score": float(row[int(index)]),
                    "rank": rank,
                }
                for rank, index in enumerate(indices, start=1)
            ]
        )
    return result


def build(cfg: dict) -> dict:
    """Кодирует корпус, сохраняет матрицу и паспорт в манифест."""
    encoder_cfg = cfg["encoder"]
    corpus = data.load_corpus()
    texts = [data.document_text(document) for document in corpus]
    doc_ids = [str(document["doc_id"]) for document in corpus]

    model = _load_model(encoder_cfg)
    matrix = encode_documents(model, texts, int(encoder_cfg["batch_size"]))
    if matrix.shape != (len(corpus), model.get_embedding_dimension()):
        raise ValueError(f"Неожиданная форма матрицы: {matrix.shape}")

    data.INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(data.INDEX_PATH, matrix)
    files.write_json(data.DOC_IDS_PATH, doc_ids)

    index_entry = {
        "model": encoder_cfg["model"],
        "revision": _model_revision(
            encoder_cfg["model"], encoder_cfg.get("revision")
        ),
        "dimension": int(matrix.shape[1]),
        "max_seq_length": int(model.max_seq_length),
        "normalize": True,
        "query_prefix": encoder_cfg["query_prefix"],
        "document_prefix": encoder_cfg.get("document_prefix", ""),
        "clipped_fraction": _clipped_fraction(model, texts),
        "matrix": {
            "path": data.INDEX_REL,
            "sha256": files.get_sha256(data.INDEX_PATH),
            "n_bytes": data.INDEX_PATH.stat().st_size,
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
            "raw_values_sha256": _sha256_array(matrix),
        },
        "doc_ids": {
            "path": data.DOC_IDS_REL,
            "sha256": files.get_sha256(data.DOC_IDS_PATH),
            "n_docs": len(doc_ids),
        },
        "built_at": utils.time_now("%Y-%m-%d"),
    }

    manifest = data.read_manifest()
    manifest[data.MANIFEST_KEY]["index"] = index_entry
    files.write_json(files.MANIFEST_PATH, manifest)
    return index_entry


def load(cfg: dict | None = None) -> dict:
    """Загружает матрицу и проверяет её хеш по манифесту."""
    entry = data.load_entry()
    index_entry = entry.get("index")
    if index_entry is None:
        raise ValueError(
            "В манифесте нет паспорта индекса: запустите "
            "scripts/lr2/build_index.py"
        )

    expected = index_entry["matrix"]["sha256"]
    actual = files.get_sha256(data.INDEX_PATH)
    if actual != expected:
        raise ValueError(
            f"SHA-256 матрицы не совпал с манифестом: {actual} != {expected}"
        )
    raw_ids = files.read_json(data.DOC_IDS_PATH)
    expected_ids = index_entry["doc_ids"]["sha256"]
    actual_ids = files.get_sha256(data.DOC_IDS_PATH)
    if actual_ids != expected_ids:
        raise ValueError(
            f"SHA-256 doc_ids не совпал с манифестом: "
            f"{actual_ids} != {expected_ids}"
        )

    matrix = np.load(data.INDEX_PATH)
    doc_ids = [str(doc_id) for doc_id in raw_ids]
    if matrix.shape != tuple(index_entry["matrix"]["shape"]):
        raise ValueError(f"Форма матрицы не совпала: {matrix.shape}")
    if len(doc_ids) != matrix.shape[0]:
        raise ValueError("Число doc_id не совпало с числом строк матрицы")
    return {
        "matrix": np.asarray(matrix, dtype=np.float32),
        "doc_ids": doc_ids,
        "passport": index_entry,
        "encoder_model": index_entry["model"],
        "query_prefix": index_entry["query_prefix"],
    }


def load_model_for_search(index_record: dict):
    """Модель кодировщика для запросов; ревизия берётся из паспорта."""
    return _load_model(
        {
            "model": index_record["encoder_model"],
            "revision": index_record["passport"]["revision"],
            "device": "cpu",
            "max_seq_length": index_record["passport"]["max_seq_length"],
        }
    )


def search_claims(
    claims: list[dict],
    index_record: dict,
    *,
    top_k: int = 5,
    batch_size: int = 32,
) -> list[list[dict]]:
    """Первые top_k документов для каждого утверждения."""
    model = load_model_for_search(index_record)
    query_matrix = encode_queries(
        model,
        [claim["claim"] for claim in claims],
        index_record["query_prefix"],
        batch_size,
    )
    return search_matrix(
        query_matrix,
        index_record["matrix"],
        index_record["doc_ids"],
        top_k=top_k,
    )


def search_claim(
    claim: dict,
    index_record: dict,
    model,
    *,
    top_k: int = 5,
) -> list[dict]:
    """Поиск для одного утверждения без повторной загрузки модели."""
    query_matrix = encode_queries(
        model, [claim["claim"]], index_record["query_prefix"]
    )
    return search_matrix(
        query_matrix,
        index_record["matrix"],
        index_record["doc_ids"],
        top_k=top_k,
    )[0]


def save_path() -> Path:
    """Путь к сохранённой матрице (для отчёта и проверок)."""
    return data.INDEX_PATH
