"""Эксперимент 3 ЛР3: выбранный вариант и Dense на holdout."""

from pathlib import Path

from ai_agent import console, files, metadata
from ai_agent.labs.lr2 import evaluation

from . import metrics, runner, selection


def _answers(records: list[dict], variant: str) -> list[dict]:
    return [record for record in records if record["variant"] == variant]


def holdout_lab(cfg: dict, run_dir: Path) -> None:
    """Один запуск Dense и выбранного варианта на всех 210 утверждениях."""
    selection_path = files.PROJECT_ROOT / cfg["selection_path"]
    selected_record = files.read_json(selection_path)
    selected = selected_record["selected"]
    if selected not in selection.CANDIDATES:
        raise ValueError(f"Неизвестный выбранный вариант: {selected}")
    need_rerank = selected == "rrf_ce"

    session = runner.setup(
        cfg,
        run_dir,
        split="holdout",
        need_reranker=need_rerank,
    )
    metadata.add(
        run_dir,
        {
            "selection": {
                "path": cfg["selection_path"],
                "sha256": files.get_sha256(selection_path),
                "selected": selected,
                "commit": selected_record.get("commit"),
                "dev_metrics": selected_record.get("dev_metrics"),
            }
        },
    )
    runner.warmup_generation(
        session,
        cfg,
        run_dir,
        need_rerank=need_rerank,
    )
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
            need_rerank=need_rerank,
        )
        files.append_jsonl(retrieval_path, retrieval)
        runner.generate(
            session,
            claim,
            "dense",
            retrieval["dense"],
            results_path,
            search_s=retrieval["timings_s"]["dense"],
        )
        runner.generate(
            session,
            claim,
            selected,
            retrieval[selected],
            results_path,
            search_s=retrieval["timings_s"].get(
                "rerank" if selected == "rrf_ce" else selected,
                0.0,
            ),
        )

    retrieval_records = files.read_jsonl(retrieval_path)
    answer_records = files.read_jsonl(results_path)
    if len(answer_records) != 2 * len(session.claims):
        raise ValueError(
            f"Ожидалось {2 * len(session.claims)} ответов, "
            f"получено {len(answer_records)}"
        )
    rankings = {
        variant: {
            record["claim_id"]: record[variant] for record in retrieval_records
        }
        for variant in ("dense", selected)
    }
    search = metrics.search_metrics(rankings, session.gold)
    search_by_class = metrics.by_class_search(rankings, session.gold, k=3)
    answers = {
        "dense": metrics.answer_metrics(
            _answers(answer_records, "dense"), session.gold
        ),
        selected: metrics.answer_metrics(
            _answers(answer_records, selected), session.gold
        ),
    }
    pairwise = {
        f"{selected}_vs_dense": evaluation.pairwise(
            _answers(answer_records, "dense"),
            _answers(answer_records, selected),
            session.gold,
        )
    }
    timing_names = ["dense", "bm25", "rrf"]
    if need_rerank:
        timing_names.append("rerank")
    result = {
        "split": "holdout",
        "n": len(session.claims),
        "n_q_plus": search["n_q_plus"],
        "selected": selected,
        "selection": {
            "path": cfg["selection_path"],
            "sha256": files.get_sha256(selection_path),
            "commit": selected_record.get("commit"),
        },
        "search": search,
        "search_by_class": search_by_class,
        "timings_s": {
            name: metrics.timing_summary(
                [record["timings_s"][name] for record in retrieval_records]
            )
            for name in timing_names
        },
        "answers": answers,
        "pairwise": pairwise,
        "dense_reference": {
            "lr2_holdout_available": False,
            "note": "Dense на holdout считается в этом запуске",
        },
    }
    files.write_json(run_dir / files.METRICS_FILENAME, result)
    files.write_json(run_dir / "pairwise.json", pairwise)

    for variant in ("dense", selected):
        values = search["variants"][variant]["k3"]
        console.info(
            f"{variant}: Hit@3={values['hit']:.3f}, "
            f"Recall@3={values['recall']:.3f}, "
            f"MRR@3={values['mrr']:.3f}"
        )
    for variant, value in answers.items():
        console.info(
            f"{variant}: accuracy={value['answers']['accuracy']:.3f}, "
            f"reference={value['answers']['reference_accuracy']}, "
            "rationale="
            f"{value['context']['rationale_passed_fraction']:.3f}"
        )
    transition = pairwise[f"{selected}_vs_dense"]
    console.info(
        f"{selected} vs dense: исправлено {transition['fixed']}, "
        f"испорчено {transition['broke']} из {transition['n']}"
    )
