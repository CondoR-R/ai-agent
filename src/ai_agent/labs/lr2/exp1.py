"""Эксперимент 1 ЛР2: LLM-only, RAG k=3 и oracle на 90 dev."""

import statistics
from pathlib import Path

from ai_agent import console, files

from . import evaluation, retrieval, runner


def _rankings(records: list[dict]) -> dict[int, list[dict]]:
    return {record["claim_id"]: record["retrieved"] for record in records}


def _subset(records: list[dict], ids: set[int]) -> list[dict]:
    return [record for record in records if record["claim_id"] in ids]


def exp1_lab(cfg: dict, run_dir: Path) -> None:
    """240 вызовов: 90 LLM-only, 90 RAG k=3, 60 oracle."""
    session = runner.setup(cfg, run_dir)
    runner.warmup(session, run_dir)
    rag_k = int(cfg["exp1"]["rag_k"])
    results_path = run_dir / files.RESULTS_FILENAME
    positives = [
        claim
        for claim in session.claims
        if session.gold[claim["id"]]["label"] in evaluation.POSITIVE_LABELS
    ]

    console.info("режим LLM-only: 90 утверждений")
    for number, claim in enumerate(session.claims, start=1):
        record = runner.run_llm_only(session, claim, results_path)
        if number % 10 == 0:
            console.info(f"LLM-only: {number}/90 ({record['parse_status']})")

    console.info(f"режим RAG k={rag_k}: 90 утверждений")
    for number, claim in enumerate(session.claims, start=1):
        record = runner.run_rag(session, claim, rag_k, results_path)
        if number % 10 == 0:
            console.info(
                f"RAG k={rag_k}: {number}/90 ({record['parse_status']})"
            )

    console.info("режим oracle: 60 утверждений SUPPORT/CONTRADICT")
    for number, claim in enumerate(positives, start=1):
        record = runner.run_oracle(session, claim, results_path)
        if number % 10 == 0:
            console.info(f"oracle: {number}/60 ({record['parse_status']})")

    records = files.read_jsonl(results_path)
    llm_records = [r for r in records if r["mode"] == "llm_only"]
    rag_records = [r for r in records if r["mode"] == "rag"]
    oracle_records = [r for r in records if r["mode"] == "oracle"]
    oracle_ids = {record["claim_id"] for record in oracle_records}

    metrics = {
        "n_calls": len(records),
        "n_planned": 240,
        "modes": {
            "llm_only": evaluation.answer_metrics(
                llm_records, session.gold, "llm_only"
            ),
            f"rag_k{rag_k}": evaluation.answer_metrics(
                rag_records, session.gold, "rag"
            ),
            "oracle": evaluation.answer_metrics(
                oracle_records, session.gold, "oracle"
            ),
        },
        "oracle_subset": {
            "n": len(oracle_ids),
            "llm_only": evaluation.answer_metrics(
                _subset(llm_records, oracle_ids), session.gold, "llm_only"
            ),
            f"rag_k{rag_k}": evaluation.answer_metrics(
                _subset(rag_records, oracle_ids), session.gold, "rag"
            ),
            "oracle": evaluation.answer_metrics(
                oracle_records, session.gold, "oracle"
            ),
        },
        f"pairwise_llm_only_to_rag_k{rag_k}": evaluation.pairwise(
            llm_records, rag_records, session.gold
        ),
        "retrieval": retrieval.metrics_by_k(
            _rankings(rag_records), session.gold, (1, 3, 5)
        ),
        f"context_rag_k{rag_k}": evaluation.context_metrics(
            rag_records, session.gold
        ),
        "context_oracle": evaluation.context_metrics(
            oracle_records, session.gold
        ),
        "prompt_tokens_expected_max": max(
            record["prompt_tokens_expected"] for record in records
        ),
        "prompt_tokens_max": max(
            record["prompt_tokens"] or 0 for record in records
        ),
        "budget_overflows": sum(
            1
            for record in records
            if (record["prompt_tokens"] or 0) > session.budget
        ),
        "search_s_median": statistics.median(
            record["search_s"] for record in rag_records
        ),
        "generation_s_median": statistics.median(
            record["generation_s"] for record in records
        ),
    }
    files.write_json(run_dir / files.METRICS_FILENAME, metrics)

    acc = metrics["modes"]
    console.info(
        f"Accuracy: LLM-only={acc['llm_only']['accuracy']:.3f}, "
        f"RAG k={rag_k}={acc[f'rag_k{rag_k}']['accuracy']:.3f}, "
        f"oracle={acc['oracle']['accuracy']:.3f}"
    )
    transitions = metrics[f"pairwise_llm_only_to_rag_k{rag_k}"]
    console.info(
        f"LLM-only → RAG k={rag_k}: исправлено {transitions['fixed']}, "
        f"испорчено {transitions['broke']}"
    )
