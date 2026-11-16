"""Единые правила идентификаторов и ранжирования ЛР3."""

from collections.abc import Sequence


def doc_key(doc_id: object) -> tuple[int, str | int]:
    """Ключ сортировки: числовые doc_id идут по значению."""
    text = str(doc_id)
    if text.isdigit():
        return (0, int(text))
    return (1, text)


def item_id(item: object) -> str:
    """doc_id из элемента выдачи или из строки."""
    if isinstance(item, dict):
        return str(item["doc_id"])
    return str(item)


def ids(ranking: Sequence[object], depth: int | None = None) -> list[str]:
    """doc_id выдачи по порядку, необязательно только первые depth."""
    items = ranking if depth is None else ranking[:depth]
    return [item_id(item) for item in items]


def ranked_list(
    scores: dict[str, float],
    *,
    top_k: int = 20,
) -> list[dict]:
    """Сортировка по убыванию оценки, при равенстве — меньший doc_id."""
    ordered = sorted(
        (doc_id for doc_id, score in scores.items() if score > 0),
        key=lambda doc_id: (-scores[doc_id], doc_key(doc_id)),
    )[:top_k]
    return [
        {"doc_id": doc_id, "score": float(scores[doc_id]), "rank": rank}
        for rank, doc_id in enumerate(ordered, start=1)
    ]
