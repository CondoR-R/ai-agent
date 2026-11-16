"""Эксперимент 1 ЛР3: Dense, BM25 и RRF на dev."""

from ai_agent import console, files

from . import metrics, reference, runner


def _rankings(records: list[dict], variant: str) -> dict[int, list[dict]]:
    return {record["claim_id"]: record[variant] for record in records}


def exp1_lab(cfg: dict, run_dir) -> None:
    """Считает выдачи трёх вариантов и таблицу дополнительности."""
    session = runner.setup(cfg, run_dir, split="dev", need_reranker=False)
    runner.warmup_retrieval(session, cfg, need_rerank=False)
    top_k, depth, c = runner._retrieval_config(cfg)
    results_path = run_dir / files.RESULTS_FILENAME

    for claim in session.claims:
        record = runner.retrieve_all(
            session,
            claim,
            top_k=top_k,
            depth=depth,
            c=c,
            need_rerank=False,
        )
        files.append_jsonl(results_path, record)

    records = files.read_jsonl(results_path)
    lr2_records = reference.load_lr2_dense_records(cfg)
    dense_check = reference.dense_reference_check(records, lr2_records)
    files.write_json(run_dir / "dense_reference_check.json", dense_check)
    if not dense_check["all_match"]:
        raise ValueError(
            "Первые пять Dense не совпали с ЛР2: "
            f"{dense_check['mismatches'][:3]}"
        )

    rankings = {
        variant: _rankings(records, variant)
        for variant in ("dense", "bm25", "rrf")
    }
    search = metrics.search_metrics(rankings, session.gold)
    q_plus = metrics.q_plus_ids(session.gold)
    complementarity = metrics.complementarity(
        q_plus,
        rankings["dense"],
        rankings["bm25"],
        rankings["rrf"],
        session.gold,
    )
    timings = {
        name: metrics.timing_summary(
            [record["timings_s"][name] for record in records]
        )
        for name in ("dense", "bm25", "rrf")
    }
    result = {
        "split": "dev",
        "n": len(records),
        "n_q_plus": len(q_plus),
        "search": search,
        "complementarity": complementarity,
        "timings_s": timings,
        "dense_reference": dense_check,
    }
    files.write_json(run_dir / files.METRICS_FILENAME, result)
    files.write_json(run_dir / "complementarity.json", complementarity)

    for variant in ("dense", "bm25", "rrf"):
        values = search["variants"][variant]["k3"]
        console.info(
            f"{variant}: Hit@3={values['hit']:.3f}, "
            f"Recall@3={values['recall']:.3f}, "
            f"MRR@3={values['mrr']:.3f}"
        )
    console.info(
        f"Dense top-5 совпал с ЛР2: {dense_check['matched']}/{dense_check['n']}"
    )
