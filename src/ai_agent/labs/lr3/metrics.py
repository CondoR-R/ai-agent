"""Метрики поиска, дополнительности, времени и ответов ЛР3."""

from collections.abc import Iterable, Sequence
from statistics import mean, median

from ai_agent.labs.lr2 import evaluation, retrieval

from .ranking import ids

POSITIVE = {"SUPPORT", "CONTRADICT"}
GROUPS = ("both", "dense_only", "bm25_only", "neither")


def q_plus_ids(gold: dict) -> list[int]:
    """Утверждения с эталонными документами."""
    return [
        claim_id
        for claim_id, record in gold.items()
        if record["label"] in POSITIVE
    ]


def search_metrics(
    rankings: dict[str, dict[int, list[dict]]],
    gold: dict,
    *,
    k_values: tuple[int, ...] = (3, 20),
) -> dict:
    """Hit/Recall/MRR для каждого варианта по Q+."""
    claim_ids = q_plus_ids(gold)
    return {
        "n_q_plus": len(claim_ids),
        "variants": {
            variant: {
                f"k{k}": retrieval.retrieval_scores(
                    claim_ids,
                    by_claim,
                    gold,
                    k,
                )
                for k in k_values
            }
            for variant, by_claim in rankings.items()
        },
    }


def _has_relevant(
    rankings: dict[int, list[dict]],
    gold: dict,
    claim_id: int,
    depth: int,
) -> bool:
    relevant = retrieval.relevant_documents(gold, claim_id)
    return bool(relevant & set(ids(rankings[claim_id], depth)))


def complementarity(
    claim_ids: Iterable[int],
    dense: dict[int, list[dict]],
    bm25: dict[int, list[dict]],
    rrf: dict[int, list[dict]],
    gold: dict,
    *,
    depths: tuple[int, ...] = (3, 20),
    rrf_depth: int = 3,
) -> dict:
    """Таблица «оба / только Dense / только BM25 / ни один» и роль RRF."""
    claim_ids = list(claim_ids)
    result = {}
    for depth in depths:
        by_group: dict[str, list[int]] = {group: [] for group in GROUPS}
        for claim_id in claim_ids:
            dense_hit = _has_relevant(dense, gold, claim_id, depth)
            bm25_hit = _has_relevant(bm25, gold, claim_id, depth)
            if dense_hit and bm25_hit:
                group = "both"
            elif dense_hit:
                group = "dense_only"
            elif bm25_hit:
                group = "bm25_only"
            else:
                group = "neither"
            by_group[group].append(claim_id)

        groups = {}
        for group, members in by_group.items():
            found = sum(
                _has_relevant(rrf, gold, claim_id, rrf_depth)
                for claim_id in members
            )
            groups[group] = {
                "n": len(members),
                "rrf_top3": found,
                "rrf_top3_fraction": found / len(members) if members else None,
                "claim_ids": members,
            }
        result[str(depth)] = groups
    return {
        "n_q_plus": len(claim_ids),
        "rrf_depth": rrf_depth,
        "by_depth": result,
    }


def by_class_search(
    rankings: dict[str, dict[int, list[dict]]],
    gold: dict,
    *,
    k: int = 3,
) -> dict:
    """Hit/Recall/MRR по классам SUPPORT и CONTRADICT."""
    result = {}
    for label in sorted(POSITIVE):
        claim_ids = [
            claim_id
            for claim_id, record in gold.items()
            if record["label"] == label
        ]
        result[label] = {
            "n": len(claim_ids),
            "variants": {
                variant: retrieval.retrieval_scores(
                    claim_ids,
                    by_claim,
                    gold,
                    k,
                )
                for variant, by_claim in rankings.items()
            },
        }
    return result


def timing_summary(values: Sequence[float]) -> dict:
    """Медиана и диапазон времени в секундах."""
    if not values:
        return {"n": 0, "median_s": None, "min_s": None, "max_s": None}
    return {
        "n": len(values),
        "median_s": float(median(values)),
        "min_s": float(min(values)),
        "max_s": float(max(values)),
        "mean_s": float(mean(values)),
    }


def answer_metrics(
    records: Sequence[dict],
    gold: dict,
) -> dict:
    """Accuracy, ссылки и обоснование для одного варианта."""
    return {
        "answers": evaluation.answer_metrics(records, gold, "rag"),
        "context": evaluation.context_metrics(records, gold),
    }


def selection_values(metrics: dict) -> dict:
    """Три метрики правила выбора для одного варианта."""
    return {
        "mrr@3": metrics["k3"]["mrr"],
        "recall@3": metrics["k3"]["recall"],
        "hit@3": metrics["k3"]["hit"],
        "recall@20": metrics["k20"]["recall"],
    }
