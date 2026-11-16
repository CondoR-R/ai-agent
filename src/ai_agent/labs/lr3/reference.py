"""Сверка Dense с сохранёнными ответами ЛР2."""

from ai_agent import files
from ai_agent.labs.lr2 import data

from .ranking import ids


def load_lr2_dense_records(cfg: dict) -> list[dict]:
    """90 RAG k=3 записей Dense из финального запуска ЛР2."""
    run = cfg.get("dense_reference_run")
    if not run:
        raise ValueError("В конфиге нет dense_reference_run")
    path = files.PROJECT_ROOT / run / files.RESULTS_FILENAME
    records = [
        record
        for record in files.read_jsonl(path)
        if record.get("mode") == "rag" and record.get("k") == 3
    ]
    expected = [claim["id"] for claim in data.dev_claims()]
    by_id = {record["claim_id"]: record for record in records}
    missing = [claim_id for claim_id in expected if claim_id not in by_id]
    if missing:
        raise ValueError(f"В Dense ЛР2 нет утверждений: {missing[:5]}")
    return [by_id[claim_id] for claim_id in expected]


def dense_reference_check(
    retrieval_records: list[dict],
    lr2_records: list[dict],
) -> dict:
    """Сравнивает первые пять Dense ЛР3 с первыми пятью ЛР2."""
    lr2_by_id = {record["claim_id"]: record for record in lr2_records}
    mismatches = []
    max_score_delta = 0.0
    for record in retrieval_records:
        claim_id = record["claim_id"]
        lr2_ranking = lr2_by_id[claim_id]["retrieved"]
        got = ids(record["dense"], 5)
        expected = ids(lr2_ranking, 5)
        if got != expected:
            mismatches.append(
                {
                    "claim_id": claim_id,
                    "got": got,
                    "expected": expected,
                }
            )
            continue
        for item, old in zip(record["dense"][:5], lr2_ranking[:5], strict=True):
            max_score_delta = max(
                max_score_delta,
                abs(float(item["score"]) - float(old["score"])),
            )
    return {
        "n": len(retrieval_records),
        "matched": len(retrieval_records) - len(mismatches),
        "all_match": not mismatches,
        "max_score_delta": max_score_delta,
        "mismatches": mismatches,
    }
