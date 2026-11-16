"""Проверки ЛР3 без сервера модели."""

import numpy as np
import pytest

from ai_agent.labs.lr2 import data, evaluation
from ai_agent.labs.lr3 import (
    bm25,
    dense,
    fusion,
    metrics,
    rerank,
    selection,
)

BM25_DOCUMENTS = {
    "1": "alpha beta beta gamma",
    "2": "alpha gamma",
    "3": "delta epsilon",
}


def test_bm25_manual_example_and_no_match():
    model = bm25.BM25Index(BM25_DOCUMENTS, k1=1.2, b=0.75)
    ranked = model.search("alpha beta beta zeta", top_k=20)
    assert [item["doc_id"] for item in ranked] == ["1", "2"]
    assert ranked[0]["score"] == pytest.approx(1.5725612026838962)
    assert ranked[1]["score"] == pytest.approx(0.523548346501579)
    assert model.search("zeta omega", top_k=20) == []


def test_bm25_ties_use_smaller_doc_id():
    model = bm25.BM25Index(
        {"10": "same", "2": "same"},
        k1=1.2,
        b=0.75,
    )
    ranked = model.search("same", top_k=20)
    assert [item["doc_id"] for item in ranked] == ["2", "10"]


def test_rrf_toy_example():
    ranked = fusion.reciprocal_rank_fusion(
        ["1", "2"],
        ["2", "3"],
        depth=20,
        c=60.0,
        top_k=20,
    )
    assert [item["doc_id"] for item in ranked] == ["2", "1", "3"]
    assert ranked[0]["dense_rank"] == 2
    assert ranked[0]["bm25_rank"] == 1
    assert ranked[0]["dense_contribution"] == pytest.approx(1 / 62)
    assert ranked[0]["bm25_contribution"] == pytest.approx(1 / 61)


def test_rerank_keeps_set_and_uses_tiebreak():
    corpus = {
        "10": {"doc_id": "10", "title": "t", "abstract": ["a"]},
        "2": {"doc_id": "2", "title": "t", "abstract": ["b"]},
        "3": {"doc_id": "3", "title": "t", "abstract": ["c"]},
    }
    ranking = [
        {"doc_id": "10", "score": 0.3, "rank": 1},
        {"doc_id": "2", "score": 0.2, "rank": 2},
        {"doc_id": "3", "score": 0.1, "rank": 3},
    ]
    scores = {"10": 0.5, "2": 0.5, "3": 0.9}
    result = rerank.rerank_with_scores(
        "claim",
        ranking,
        corpus,
        lambda claim, documents: [scores[item] for item in ("10", "2", "3")],
    )
    assert {item["doc_id"] for item in result} == {"10", "2", "3"}
    assert [item["doc_id"] for item in result] == ["3", "2", "10"]
    assert result[0]["rrf_score"] == pytest.approx(0.1)
    assert result[0]["rerank_score"] == pytest.approx(0.9)


def test_dense_ties_use_smaller_doc_id(monkeypatch):
    monkeypatch.setattr(
        "ai_agent.labs.lr2.index.encode_queries",
        lambda model, texts, prefix: np.array([[1.0, 0.0]]),
    )
    record = {
        "query_prefix": "",
        "doc_ids": ["10", "2", "3"],
        "matrix": np.array(
            [
                [0.5, 0.0],
                [0.5, 0.0],
                [0.9, 0.0],
            ],
            dtype=np.float32,
        ),
    }
    result = dense.dense_search(record, object(), "claim", top_k=3)
    assert [item["doc_id"] for item in result] == ["3", "2", "10"]


def test_complementarity_groups_and_rrf_top3():
    gold = {
        claim_id: {
            "label": "SUPPORT",
            "evidence": {"x": [{"sentences": [0], "label": "SUPPORT"}]},
        }
        for claim_id in (1, 2, 3, 4)
    }
    dense = {
        1: [{"doc_id": "x"}],
        2: [{"doc_id": "x"}],
        3: [{"doc_id": "y"}],
        4: [{"doc_id": "y"}],
    }
    bm25 = {
        1: [{"doc_id": "x"}],
        2: [{"doc_id": "y"}],
        3: [{"doc_id": "z"}],
        4: [{"doc_id": "x"}],
    }
    rrf = {
        1: [{"doc_id": "x"}],
        2: [{"doc_id": "x"}],
        3: [{"doc_id": "y"}],
        4: [{"doc_id": "x"}],
    }
    result = metrics.complementarity(
        [1, 2, 3, 4],
        dense,
        bm25,
        rrf,
        gold,
        depths=(3,),
    )
    groups = result["by_depth"]["3"]
    assert groups["both"]["n"] == 1
    assert groups["dense_only"]["n"] == 1
    assert groups["bm25_only"]["n"] == 1
    assert groups["neither"]["n"] == 1
    assert groups["both"]["rrf_top3_fraction"] == 1.0
    assert groups["dense_only"]["rrf_top3_fraction"] == 1.0
    assert groups["bm25_only"]["rrf_top3_fraction"] == 1.0
    assert groups["neither"]["rrf_top3_fraction"] == 0.0


def test_selection_rule_ties_use_recall_then_simplicity():
    values = {
        "bm25": {"mrr@3": 0.5, "recall@3": 0.5, "hit@3": 1, "recall@20": 1},
        "rrf": {"mrr@3": 0.5, "recall@3": 0.5, "hit@3": 1, "recall@20": 1},
        "rrf_ce": {"mrr@3": 0.5, "recall@3": 0.6, "hit@3": 1, "recall@20": 1},
    }
    selected, record = selection.choose_variant(values)
    assert selected == "rrf_ce"
    assert record["selected"] == "rrf_ce"

    values["rrf_ce"]["recall@3"] = 0.5
    selected, _ = selection.choose_variant(values)
    assert selected == "bm25"


def test_holdout_q_plus_count():
    claims = data.holdout_claims()
    gold = evaluation.gold_by_id(claims)
    assert len(claims) == 210
    assert len(metrics.q_plus_ids(gold)) == 128
