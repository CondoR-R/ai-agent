"""Эксперимент 2 ЛР2: RAG при k=1 и k=5 на 90 dev."""

import statistics
from pathlib import Path

from ai_agent import console, files

from . import evaluation, retrieval, runner


def _rankings(records: list[dict]) -> dict[int, list[dict]]:
    return {record["claim_id"]: record["retrieved"] for record in records}


def exp2_lab(cfg: dict, run_dir: Path) -> None:
    """180 вызовов: по 90 утверждений для k=1 и k=5."""
    session = runner.setup(cfg, run_dir)
    results_path = run_dir / files.RESULTS_FILENAME

    for k in cfg["exp2"]["k_values"]:
        console.info(f"режим RAG k={k}: 90 утверждений")
        for number, claim in enumerate(session.claims, start=1):
            record = runner.run_rag(session, claim, k, results_path)
            if number % 10 == 0:
                console.info(
                    f"RAG k={k}: {number}/90 ({record['parse_status']})"
                )

    records = files.read_jsonl(results_path)
    k1_records = [record for record in records if record["k"] == 1]
    k5_records = [record for record in records if record["k"] == 5]

    metrics = {
        "n_calls": len(records),
        "n_planned": 180,
        "modes": {
            "rag_k1": evaluation.answer_metrics(
                k1_records, session.gold, "rag"
            ),
            "rag_k5": evaluation.answer_metrics(
                k5_records, session.gold, "rag"
            ),
        },
        "pairwise_rag_k1_to_rag_k5": evaluation.pairwise(
            k1_records, k5_records, session.gold
        ),
        "retrieval": retrieval.metrics_by_k(
            _rankings(k1_records), session.gold, (1, 3, 5)
        ),
        "context_rag_k1": evaluation.context_metrics(k1_records, session.gold),
        "context_rag_k5": evaluation.context_metrics(k5_records, session.gold),
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
            record["search_s"] for record in records
        ),
        "generation_s_median": statistics.median(
            record["generation_s"] for record in records
        ),
    }
    files.write_json(run_dir / files.METRICS_FILENAME, metrics)

    modes = metrics["modes"]
    console.info(
        f"Accuracy: RAG k=1={modes['rag_k1']['accuracy']:.3f}, "
        f"RAG k=5={modes['rag_k5']['accuracy']:.3f}"
    )
    transitions = metrics["pairwise_rag_k1_to_rag_k5"]
    console.info(
        f"RAG k=1 → k=5: исправлено {transitions['fixed']}, "
        f"испорчено {transitions['broke']}"
    )
