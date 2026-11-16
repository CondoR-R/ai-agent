"""Эксперимент 2 ЛР3: RRF и RRF + cross-encoder на dev."""

from ai_agent import console, files
from ai_agent.labs.lr2 import evaluation

from . import metrics, reference, runner


def _by_variant(records: list[dict], variant: str) -> list[dict]:
    return [record for record in records if record["variant"] == variant]


def _rankings(records: list[dict], variants: tuple[str, ...]) -> dict:
    return {
        variant: {record["claim_id"]: record[variant] for record in records}
        for variant in variants
    }


def exp2_lab(cfg: dict, run_dir) -> None:
    """Сравнивает RRF и RRF+CE по поиску, ответам и времени."""
    session = runner.setup(cfg, run_dir, split="dev", need_reranker=True)
    runner.warmup_generation(session, cfg, run_dir, need_rerank=True)
    top_k, depth, c = runner._retrieval_config(cfg)
    retrieval_path = run_dir / "retrieval.jsonl"
    results_path = run_dir / files.RESULTS_FILENAME

    for claim in session.claims:
        retrieval = runner.retrieve_all(
            session,
            claim,
            top_k=top_k,
            depth=depth,
            c=c,
            need_rerank=True,
        )
        files.append_jsonl(retrieval_path, retrieval)
        runner.generate(
            session,
            claim,
            "rrf",
            retrieval["rrf"],
            results_path,
            search_s=retrieval["timings_s"]["rrf"],
        )
        runner.generate(
            session,
            claim,
            "rrf_ce",
            retrieval["rrf_ce"],
            results_path,
            search_s=retrieval["timings_s"]["rerank"],
        )

    retrieval_records = files.read_jsonl(retrieval_path)
    answer_records = files.read_jsonl(results_path)
    lr2_records = reference.load_lr2_dense_records(cfg)
    dense_check = reference.dense_reference_check(
        retrieval_records, lr2_records
    )
    files.write_json(run_dir / "dense_reference_check.json", dense_check)
    if not dense_check["all_match"]:
        raise ValueError(
            "Первые пять Dense не совпали с ЛР2: "
            f"{dense_check['mismatches'][:3]}"
        )

    rankings = _rankings(
        retrieval_records,
        ("dense", "bm25", "rrf", "rrf_ce"),
    )
    search = metrics.search_metrics(rankings, session.gold)
    timings = {
        name: metrics.timing_summary(
            [record["timings_s"][name] for record in retrieval_records]
        )
        for name in ("dense", "bm25", "rrf", "rerank")
    }
    answers = {
        "dense_lr2": metrics.answer_metrics(lr2_records, session.gold),
        "rrf": metrics.answer_metrics(
            _by_variant(answer_records, "rrf"), session.gold
        ),
        "rrf_ce": metrics.answer_metrics(
            _by_variant(answer_records, "rrf_ce"), session.gold
        ),
    }
    pairwise = {
        "rrf_ce_vs_dense": evaluation.pairwise(
            lr2_records,
            _by_variant(answer_records, "rrf_ce"),
            session.gold,
        ),
        "rrf_vs_dense": evaluation.pairwise(
            lr2_records,
            _by_variant(answer_records, "rrf"),
            session.gold,
        ),
        "rrf_ce_vs_rrf": evaluation.pairwise(
            _by_variant(answer_records, "rrf"),
            _by_variant(answer_records, "rrf_ce"),
            session.gold,
        ),
    }
    result = {
        "split": "dev",
        "n": len(retrieval_records),
        "n_q_plus": search["n_q_plus"],
        "search": search,
        "timings_s": timings,
        "answers": answers,
        "pairwise": pairwise,
        "dense_reference": dense_check,
    }
    files.write_json(run_dir / files.METRICS_FILENAME, result)
    files.write_json(run_dir / "pairwise.json", pairwise)

    for variant in ("dense", "bm25", "rrf", "rrf_ce"):
        values = search["variants"][variant]["k3"]
        console.info(
            f"{variant}: Hit@3={values['hit']:.3f}, "
            f"Recall@3={values['recall']:.3f}, "
            f"MRR@3={values['mrr']:.3f}"
        )
    for name, value in pairwise.items():
        console.info(
            f"{name}: исправлено {value['fixed']}, "
            f"испорчено {value['broke']} из {value['n']}"
        )
    console.info(
        f"Dense top-5 совпал с ЛР2: {dense_check['matched']}/{dense_check['n']}"
    )
