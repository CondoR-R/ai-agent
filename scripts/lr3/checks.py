"""Проверки ЛР3 перед официальными опытами."""

import math

from ai_agent import files
from ai_agent.labs.lr2 import data, evaluation, index, retrieval
from ai_agent.labs.lr3 import (
    bm25,
    dense,
    fusion,
    metrics,
    reference,
    rerank,
)
from ai_agent.labs.lr3.ranking import ids

BM25_DOCUMENTS = {
    "1": "alpha beta beta gamma",
    "2": "alpha gamma",
    "3": "delta epsilon",
}
BM25_EXPECTED = {
    "1": 1.5725612026838962,
    "2": 0.523548346501579,
}


def _bm25_check() -> dict:
    model = bm25.BM25Index(BM25_DOCUMENTS, k1=1.2, b=0.75)
    ranked = model.search("alpha beta beta zeta", top_k=20)
    no_match = model.search("zeta omega", top_k=20)
    got = {item["doc_id"]: item["score"] for item in ranked}
    close = {
        doc_id: math.isclose(
            got.get(doc_id, float("nan")),
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for doc_id, expected in BM25_EXPECTED.items()
    }
    return {
        "documents": BM25_DOCUMENTS,
        "query": "alpha beta beta zeta",
        "ranked": ranked,
        "expected_scores": BM25_EXPECTED,
        "scores_close": close,
        "order": ids(ranked),
        "no_match_query": "zeta omega",
        "no_match": no_match,
        "pass": (
            ids(ranked) == ["1", "2"]
            and set(got) == {"1", "2"}
            and all(close.values())
            and no_match == []
        ),
    }


def _rrf_check() -> dict:
    result = fusion.reciprocal_rank_fusion(
        ["1", "2"],
        ["2", "3"],
        depth=20,
        c=60.0,
        top_k=20,
    )
    return {
        "dense": ["1", "2"],
        "bm25": ["2", "3"],
        "ranked": result,
        "pass": ids(result) == ["2", "1", "3"],
    }


def _dense_reference_check(cfg: dict) -> dict:
    record = index.load()
    model = index.load_model_for_search(record)
    claims = data.dev_claims()
    retrieval_records = []
    for claim in claims:
        retrieval_records.append(
            {
                "claim_id": claim["id"],
                "dense": dense.dense_search(
                    record,
                    model,
                    claim["claim"],
                    top_k=20,
                ),
            }
        )
    result = reference.dense_reference_check(
        retrieval_records,
        reference.load_lr2_dense_records(cfg),
    )
    result["pass"] = result["all_match"]
    return result


def _rerank_check(cfg: dict) -> dict:
    record = index.load()
    query_model = index.load_model_for_search(record)
    corpus = data.corpus_by_id()
    bm25_index = bm25.BM25Index.from_corpus(
        corpus,
        k1=float(cfg["retrieval"]["bm25"]["k1"]),
        b=float(cfg["retrieval"]["bm25"]["b"]),
    )
    reranker = rerank.CrossEncoderReranker(
        cfg["reranker"]["model"],
        max_length=int(cfg["reranker"]["max_length"]),
        batch_size=int(cfg["reranker"]["batch_size"]),
        device=cfg["reranker"].get("device", "cpu"),
    )
    claims = data.dev_claims()
    gold = evaluation.gold_by_id(claims)
    rrf_by_id = {}
    reranked_by_id = {}
    same_set = 0
    same_length = 0
    changed_order = 0
    pairs = 0
    for claim in claims:
        dense_ranking = dense.dense_search(
            record,
            query_model,
            claim["claim"],
            top_k=20,
        )
        bm25_ranking = bm25_index.search(claim["claim"], top_k=20)
        rrf_ranking = fusion.reciprocal_rank_fusion(
            dense_ranking,
            bm25_ranking,
            depth=int(cfg["retrieval"]["rrf_depth"]),
            c=float(cfg["retrieval"]["rrf_c"]),
            top_k=int(cfg["retrieval"]["top_k"]),
        )
        reranked = reranker.rerank(
            claim["claim"],
            rrf_ranking,
            corpus,
            top_k=int(cfg["retrieval"]["top_k"]),
        )
        pairs += len(rrf_ranking)
        same_length += len(rrf_ranking) == len(reranked)
        same_set += set(ids(rrf_ranking)) == set(ids(reranked))
        changed_order += ids(rrf_ranking) != ids(reranked)
        rrf_by_id[claim["id"]] = rrf_ranking
        reranked_by_id[claim["id"]] = reranked

    q_plus = metrics.q_plus_ids(gold)
    before = retrieval.retrieval_scores(q_plus, rrf_by_id, gold, 20)
    after = retrieval.retrieval_scores(q_plus, reranked_by_id, gold, 20)
    return {
        "n_claims": len(claims),
        "n_q_plus": len(q_plus),
        "pairs": pairs,
        "same_length": same_length,
        "same_set": same_set,
        "changed_order": changed_order,
        "recall20_before": before["recall"],
        "recall20_after": after["recall"],
        "model": reranker.passport(),
        "pass": (
            same_length == len(claims)
            and same_set == len(claims)
            and before["recall"] == after["recall"]
        ),
    }


def main() -> None:
    cfg = files.read_yaml(files.PROJECT_ROOT / "configs" / "lr3" / "exp2.yaml")
    checks = {
        "bm25": _bm25_check(),
        "rrf": _rrf_check(),
        "dense_reference": _dense_reference_check(cfg),
        "rerank": _rerank_check(cfg),
    }
    checks["pass"] = all(
        checks[name]["pass"]
        for name in ("bm25", "rrf", "dense_reference", "rerank")
    )
    path = files.PROJECT_ROOT / "reports" / "lr3" / "checks.json"
    files.write_json(path, checks)
    print(f"checks: {path}", flush=True)
    if not checks["pass"]:
        raise SystemExit("Проверки ЛР3 не пройдены")


if __name__ == "__main__":
    main()
