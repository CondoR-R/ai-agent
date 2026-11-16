"""Reciprocal Rank Fusion для выдач Dense и BM25."""

from collections.abc import Sequence

from .ranking import doc_key, ids


def reciprocal_rank_fusion(
    dense: Sequence[object],
    bm25: Sequence[object],
    *,
    depth: int = 20,
    c: float = 60.0,
    top_k: int = 20,
) -> list[dict]:
    """Складывает ранги двух списков и сохраняет вклады методов."""
    scores: dict[str, float] = {}
    contributions: dict[str, dict[str, dict[str, float] | None]] = {}

    for method, ranking in (("dense", dense), ("bm25", bm25)):
        for rank, doc_id in enumerate(ids(ranking, depth), start=1):
            value = 1.0 / (c + rank)
            scores[doc_id] = scores.get(doc_id, 0.0) + value
            contributions.setdefault(
                doc_id,
                {"dense": None, "bm25": None},
            )[method] = {"rank": rank, "contribution": value}

    ordered = sorted(
        scores,
        key=lambda doc_id: (-scores[doc_id], doc_key(doc_id)),
    )[:top_k]
    result = []
    for rank, doc_id in enumerate(ordered, start=1):
        parts = contributions[doc_id]
        dense_part = parts["dense"]
        bm25_part = parts["bm25"]
        result.append(
            {
                "doc_id": doc_id,
                "score": float(scores[doc_id]),
                "rank": rank,
                "dense_rank": (int(dense_part["rank"]) if dense_part else None),
                "bm25_rank": int(bm25_part["rank"]) if bm25_part else None,
                "dense_contribution": (
                    float(dense_part["contribution"]) if dense_part else 0.0
                ),
                "bm25_contribution": (
                    float(bm25_part["contribution"]) if bm25_part else 0.0
                ),
            }
        )
    return result
