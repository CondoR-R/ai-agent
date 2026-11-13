"""Метрики поиска ЛР2: Hit@k, Recall@k, RR@k."""

from collections.abc import Iterable


def relevant_documents(gold: dict, claim_id: int) -> set[str]:
    """Множество документов-свидетелей утверждения."""
    evidence = gold[claim_id]["evidence"]
    return {str(doc_id) for doc_id in evidence}


def _ranked_ids(ranking: list[dict], k: int) -> list[str]:
    return [str(item["doc_id"]) for item in ranking[:k]]


def retrieval_scores(
    claim_ids: Iterable[int],
    rankings: dict[int, list[dict]],
    gold: dict,
    k: int,
) -> dict:
    """Средние Hit@k, Recall@k и MRR@k по множеству Q+."""
    hits = 0.0
    recalls = 0.0
    reciprocal_ranks = 0.0
    n = 0
    for claim_id in claim_ids:
        relevant = relevant_documents(gold, claim_id)
        if not relevant:
            continue
        n += 1
        found = set(_ranked_ids(rankings[claim_id], k))
        intersection = relevant & found
        hits += bool(intersection)
        recalls += len(intersection) / len(relevant)
        first_rank = None
        for rank, item in enumerate(rankings[claim_id], start=1):
            if str(item["doc_id"]) in relevant:
                first_rank = rank
                break
        if first_rank is not None and first_rank <= k:
            reciprocal_ranks += 1 / first_rank
    return {
        "n": n,
        "hit": hits / n if n else 0.0,
        "recall": recalls / n if n else 0.0,
        "mrr": reciprocal_ranks / n if n else 0.0,
    }


def metrics_by_k(
    rankings: dict[int, list[dict]],
    gold: dict,
    k_values: tuple[int, ...] = (1, 3, 5),
) -> dict:
    """Метрики для каждого k; Q+ — только SUPPORT и CONTRADICT."""
    q_plus = [
        claim_id
        for claim_id, record in gold.items()
        if record["label"] in {"SUPPORT", "CONTRADICT"}
    ]
    return {
        "n_q_plus": len(q_plus),
        "by_k": {
            str(k): retrieval_scores(q_plus, rankings, gold, k)
            for k in k_values
        },
    }
