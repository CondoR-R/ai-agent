"""Плотный поиск ЛР3: top-20 с детерминированными ничьими."""

from ai_agent.labs.lr2 import index

from .ranking import doc_key


def dense_search(
    index_record: dict,
    model,
    text: str,
    *,
    top_k: int = 20,
) -> list[dict]:
    """Кодирует запрос и возвращает top_k документов по косинусу."""
    query = index.encode_queries(
        model,
        [text],
        index_record["query_prefix"],
    )[0]
    scores = query @ index_record["matrix"].T
    doc_ids = index_record["doc_ids"]
    order = sorted(
        range(len(doc_ids)),
        key=lambda position: (
            -float(scores[position]),
            doc_key(doc_ids[position]),
        ),
    )[:top_k]
    return [
        {
            "doc_id": str(doc_ids[position]),
            "score": float(scores[position]),
            "rank": rank,
        }
        for rank, position in enumerate(order, start=1)
    ]
