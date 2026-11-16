"""Отчёт ЛР3: поиск, ответы, holdout и четыре разобранных случая."""

import argparse
from collections import Counter
from pathlib import Path

from ai_agent import files, utils
from ai_agent.labs.lr2 import data, evaluation

VARIANT_TITLES = {
    "dense": "Dense",
    "bm25": "BM25",
    "rrf": "RRF",
    "rrf_ce": "RRF + CE",
}

CASE_SPECS = (
    {
        "kind": "bm25_only",
        "claim_id": 314,
        "title": "BM25 нашёл, Dense пропустил",
        "stage": (
            "Обоснование появилось на BM25 (ранг 5), но потерялось "
            "при усечении RRF до top-3: у документа был только вклад "
            "BM25, и объединение отодвинуло его на 13-е место."
        ),
    },
    {
        "kind": "dense_only",
        "claim_id": 239,
        "title": "Dense нашёл, BM25 пропустил",
        "stage": (
            "Dense поставил документ на 11-е место, BM25 его не нашёл. "
            "В RRF у документа остался только одиночный вклад Dense, "
            "и он вытеснился из top-20; CE не мог его вернуть."
        ),
    },
    {
        "kind": "rrf_order",
        "claim_id": 636,
        "title": "RRF изменил первую тройку",
        "stage": (
            "RRF поставил на первое место документ с суммой вкладов "
            "0.03252 против 0.03226 у эталонного: dense-ранг 2 и "
            "bm25-ранг 1 перевесили dense-ранг 1 и bm25-ранг 3. "
            "CE вернул эталонный документ на первое место, но ответ "
            "модели всё равно остался CONTRADICT вместо SUPPORT."
        ),
    },
    {
        "kind": "rerank_answer",
        "claim_id": 1336,
        "title": "Переранжирование изменило ответ",
        "stage": (
            "RRF поставил эталон на 2-е место, CE — на 1-е. RRF передал "
            "первому документу только предложения 0–4, а модель сослалась "
            "на 4–6: ссылка стала invalid_reference. После CE эталонный "
            "документ с предложениями 0–6 оказался первым, и ссылка 4–6 "
            "стала валидной. Содержательная метка не изменилась."
        ),
    },
)


def _latest_run(experiment: str, status: str = "completed") -> Path:
    candidates = []
    for path in files.RUNS_DIR.glob(f"lr3-experiment{experiment}-*"):
        meta_path = path / files.METADATA_FILENAME
        if not meta_path.exists():
            continue
        meta = files.read_json(meta_path)
        if meta.get("status") == status:
            candidates.append((meta.get("started_at", ""), path))
    if not candidates:
        raise FileNotFoundError(
            f"Нет запуска lr3-experiment{experiment} со статусом {status}"
        )
    return max(candidates)[1]


def _run_dir(value: str | Path | None, experiment: str) -> Path:
    if value is None:
        return _latest_run(experiment)
    path = Path(value)
    return path if path.is_absolute() else files.PROJECT_ROOT / path


def _load_records(run_dir: Path, filename: str) -> list[dict]:
    path = run_dir / filename
    if not path.exists():
        return []
    return files.read_jsonl(path)


def _by_claim(records: list[dict]) -> dict[int, dict]:
    return {record["claim_id"]: record for record in records}


def _by_variant(records: list[dict]) -> dict[tuple[int, str], dict]:
    return {
        (record["claim_id"], record["variant"]): record for record in records
    }


def _search_values(search: dict, variant: str) -> dict:
    values = search["variants"][variant]
    return {
        "hit@3": values["k3"]["hit"],
        "recall@3": values["k3"]["recall"],
        "mrr@3": values["k3"]["mrr"],
        "recall@20": values["k20"]["recall"],
    }


def _answer_values(answers: dict, variant: str) -> dict:
    values = answers[variant]
    return {
        "accuracy": values["answers"]["accuracy"],
        "reference": values["answers"]["reference_accuracy"],
        "rationale": values["context"]["rationale_passed_fraction"],
    }


def _metrics_values(
    search: dict,
    answers: dict,
    search_variant: str,
    answer_variant: str | None = None,
) -> dict:
    return {
        **_search_values(search, search_variant),
        **_answer_values(answers, answer_variant or search_variant),
    }


def _delta(after: dict, before: dict) -> dict:
    return {key: after[key] - before[key] for key in before}


def _error_counts(records: list[dict], gold: dict) -> dict:
    counts = Counter()
    for record in records:
        category = evaluation.classify_error(record, gold[record["claim_id"]])
        counts[category or "correct"] += 1
    return dict(counts)


def _sentence_texts(document: dict, sentences: list[int]) -> list[dict]:
    abstract = document.get("abstract", [])
    return [
        {"sentence": sentence, "text": abstract[sentence]}
        for sentence in sentences
        if sentence < len(abstract)
    ]


def _gold_evidence(claim_id: int, gold: dict, corpus: dict) -> list[dict]:
    result = []
    for doc_id, rows in gold[claim_id]["evidence"].items():
        document = corpus.get(str(doc_id), {})
        for row in rows:
            result.append(
                {
                    "doc_id": str(doc_id),
                    "title": document.get("title"),
                    "label": row["label"],
                    "sentences": row["sentences"],
                    "text": _sentence_texts(document, row["sentences"]),
                }
            )
    return result


def _ranking_item(record: dict, variant: str, doc_id: str) -> dict | None:
    for item in record[variant]:
        if str(item["doc_id"]) == str(doc_id):
            return item
    return None


def _top_items(
    record: dict,
    variant: str,
    corpus: dict,
    *,
    depth: int = 5,
) -> list[dict]:
    result = []
    for item in record[variant][:depth]:
        document = corpus.get(str(item["doc_id"]), {})
        result.append({**item, "title": document.get("title")})
    return result


def _passed(record: dict, corpus: dict) -> list[dict]:
    result = []
    for document in record.get("passed", []):
        source = corpus.get(str(document["doc_id"]), {})
        result.append(
            {
                **document,
                "text": _sentence_texts(source, document["sentences"]),
            }
        )
    return result


def _answer_trace(record: dict, corpus: dict) -> dict:
    return {
        "variant": record["variant"],
        "parse_status": record["parse_status"],
        "parsed": record.get("parsed"),
        "raw_response": record.get("raw_response"),
        "passed": _passed(record, corpus),
    }


def _case(
    spec: dict,
    retrieval: dict[int, dict],
    results: dict[tuple[int, str], dict],
    gold: dict,
    corpus: dict,
) -> dict:
    claim_id = spec["claim_id"]
    record = retrieval[claim_id]
    gold_ids = [str(doc_id) for doc_id in gold[claim_id]["evidence"]]
    rankings = {}
    for variant in ("dense", "bm25", "rrf", "rrf_ce"):
        rankings[variant] = {
            "top5": _top_items(record, variant, corpus),
            "gold": {
                doc_id: _ranking_item(record, variant, doc_id)
                for doc_id in gold_ids
            },
        }
    return {
        "kind": spec["kind"],
        "title": spec["title"],
        "claim_id": claim_id,
        "claim": record["claim"],
        "gold_label": gold[claim_id]["label"],
        "gold_evidence": _gold_evidence(claim_id, gold, corpus),
        "rankings": rankings,
        "answers": {
            variant: _answer_trace(results[(claim_id, variant)], corpus)
            for variant in ("rrf", "rrf_ce")
        },
        "stage": spec["stage"],
    }


def _compare_holdout_attempts(
    failed_dir: Path | None, repeat_dir: Path
) -> dict | None:
    if failed_dir is None:
        return None
    failed_meta = files.read_json(failed_dir / files.METADATA_FILENAME)
    repeat_meta = files.read_json(repeat_dir / files.METADATA_FILENAME)
    failed_metrics = files.read_json(failed_dir / files.METRICS_FILENAME)
    repeat_metrics = files.read_json(repeat_dir / files.METRICS_FILENAME)
    failed_results = _load_records(failed_dir, files.RESULTS_FILENAME)
    repeat_results = _load_records(repeat_dir, files.RESULTS_FILENAME)
    failed_retrieval = _load_records(failed_dir, "retrieval.jsonl")
    repeat_retrieval = _load_records(repeat_dir, "retrieval.jsonl")

    def core(record: dict) -> dict:
        return {
            key: value
            for key, value in record.items()
            if key not in {"timings_s", "search_s", "generation_s"}
        }

    after_by_id = {(r["claim_id"], r["variant"]): r for r in repeat_results}
    same_results = all(
        core(record)
        == core(after_by_id[(record["claim_id"], record["variant"])])
        for record in failed_results
    )

    after_retrieval = {r["claim_id"]: r for r in repeat_retrieval}
    same_rankings = all(
        {
            variant: [item["doc_id"] for item in record[variant]]
            for variant in ("dense", "bm25", "rrf", "rrf_ce")
        }
        == {
            variant: [
                item["doc_id"]
                for item in after_retrieval[record["claim_id"]][variant]
            ]
            for variant in ("dense", "bm25", "rrf", "rrf_ce")
        }
        for record in failed_retrieval
    )

    return {
        "failed_run": failed_dir.name,
        "failed_status": failed_meta.get("status"),
        "repeat_run": repeat_dir.name,
        "repeat_series": repeat_meta.get("series"),
        "same_search_metrics": failed_metrics["search"]
        == repeat_metrics["search"],
        "same_answer_metrics": failed_metrics["answers"]
        == repeat_metrics["answers"],
        "same_pairwise": failed_metrics["pairwise"]
        == repeat_metrics["pairwise"],
        "same_rankings": same_rankings,
        "same_results": same_results,
        "note": (
            "Первый holdout завершил расчёты, но упал на логировании MLflow "
            "из-за символа @ в имени метрики. Исходный результат сохранён; "
            "повторный запуск после исправления только логирования дал те же "
            "ранжирования и ответы."
        ),
    }


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def _pp(delta: float) -> str:
    sign = "+" if delta >= 0 else ""
    return f"{sign}{100 * delta:.1f} п.п."


def _signed(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.3f}"


def _rank_text(item: dict | None) -> str:
    return "нет в top-20" if item is None else f"ранг {item['rank']}"


def _case_report(case: dict) -> list[str]:
    gold_docs = sorted({item["doc_id"] for item in case["gold_evidence"]})
    lines = [
        f"### {case['title']} (claim {case['claim_id']})",
        "",
        f"Утверждение: *{case['claim']}*  ",
        f"Эталон: **{case['gold_label']}**, документы "
        + ", ".join(f"`{doc_id}`" for doc_id in gold_docs)
        + ".",
        "",
    ]
    for evidence in case["gold_evidence"][:2]:
        for sentence in evidence["text"]:
            lines.append(
                f"Эталонное предложение `{evidence['doc_id']}` "
                f"[{sentence['sentence']}]: {sentence['text'].strip()}"
            )
    lines.extend(
        [
            "",
            "| Вариант | Ранг эталона | top-3 (doc_id) |",
            "|---|---:|---|",
        ]
    )
    for variant in ("dense", "bm25", "rrf", "rrf_ce"):
        ranking = case["rankings"][variant]
        gold_item = next(iter(ranking["gold"].values()), None)
        top = ", ".join(f"`{item['doc_id']}`" for item in ranking["top5"][:3])
        lines.append(
            f"| {VARIANT_TITLES[variant]} | {_rank_text(gold_item)} | {top} |"
        )
    rrf_gold = next(iter(case["rankings"]["rrf"]["gold"].values()), None)
    ce_gold = next(iter(case["rankings"]["rrf_ce"]["gold"].values()), None)
    if rrf_gold is None:
        traces = (
            "RRF: эталона нет в top-20, одиночный вклад Dense не удержал его."
        )
    else:
        traces = (
            f"RRF: dense-ранг {rrf_gold.get('dense_rank') or '—'}, "
            f"bm25-ранг {rrf_gold.get('bm25_rank') or '—'}, вклады "
            f"{rrf_gold.get('dense_contribution', 0.0):.5f} + "
            f"{rrf_gold.get('bm25_contribution', 0.0):.5f} = "
            f"{rrf_gold['score']:.5f}."
        )
    if ce_gold is not None:
        traces += (
            f" CE: ранг {ce_gold['rank']}, rerank_score "
            f"{ce_gold.get('rerank_score', ce_gold['score']):.3f}."
        )
    lines.extend(
        [
            "",
            f"**Этап:** {case['stage']}",
            "",
            traces,
            "",
        ]
    )
    for variant in ("rrf", "rrf_ce"):
        answer = case["answers"][variant]
        parsed = answer["parsed"] or {}
        lines.append(
            f"- {VARIANT_TITLES[variant]}: `{parsed.get('label')}` "
            f"doc `{parsed.get('doc_id')}`, предложения "
            f"{parsed.get('sentences')}, статус `{answer['parse_status']}`; "
            "передано "
            + "; ".join(
                f"`{document['doc_id']}` {document['sentences']}"
                for document in answer["passed"]
            )
            + "."
        )
    lines.append("")
    return lines


def build(
    run_dirs: dict[str, Path], failed_holdout: Path | None, out_dir: Path
) -> None:
    exp1 = files.read_json(run_dirs["exp1"] / files.METRICS_FILENAME)
    exp2 = files.read_json(run_dirs["exp2"] / files.METRICS_FILENAME)
    holdout = files.read_json(run_dirs["holdout"] / files.METRICS_FILENAME)
    selection = files.read_json(
        files.PROJECT_ROOT / "reports/lr3/selection.json"
    )

    dev_results = _load_records(run_dirs["exp2"], files.RESULTS_FILENAME)
    dev_retrieval = _by_claim(
        _load_records(run_dirs["exp2"], "retrieval.jsonl")
    )
    holdout_results = _load_records(run_dirs["holdout"], files.RESULTS_FILENAME)
    dev_gold = evaluation.gold_by_id(data.dev_claims())
    holdout_gold = evaluation.gold_by_id(data.holdout_claims())
    corpus = data.corpus_by_id()

    answers_dev = _by_variant(dev_results)
    selected = holdout["selected"]

    comparison = {}
    for (
        split,
        search,
        answers,
        dense_variant,
        dense_answer,
        selected_variant,
    ) in (
        (
            "dev",
            exp2["search"],
            exp2["answers"],
            "dense",
            "dense_lr2",
            "rrf_ce",
        ),
        (
            "holdout",
            holdout["search"],
            holdout["answers"],
            "dense",
            "dense",
            selected,
        ),
    ):
        dense = _metrics_values(search, answers, dense_variant, dense_answer)
        alternative = _metrics_values(search, answers, selected_variant)
        comparison[split] = {
            "dense": dense,
            selected_variant: alternative,
            f"{selected_variant}_minus_dense": _delta(alternative, dense),
        }
    comparison["dev_to_holdout"] = {
        variant: _delta(
            comparison["holdout"][variant], comparison["dev"][variant]
        )
        for variant in ("dense", selected)
    }

    analysis = {
        "generated_at": utils.time_now(),
        "runs": {
            "exp1": run_dirs["exp1"].name,
            "exp2": run_dirs["exp2"].name,
            "holdout": run_dirs["holdout"].name,
        },
        "selection": selection,
        "dev": {
            "n": exp2["n"],
            "n_q_plus": exp2["n_q_plus"],
            "search": exp1["search"],
            "complementarity": exp1["complementarity"],
            "answers": exp2["answers"],
            "pairwise": exp2["pairwise"],
            "timings_s": {
                "dense": exp1["timings_s"]["dense"],
                "bm25": exp1["timings_s"]["bm25"],
                "rrf": exp1["timings_s"]["rrf"],
                "rerank": exp2["timings_s"]["rerank"],
                "exp2_context": exp2["timings_s"],
            },
            "errors": {
                "rrf": _error_counts(
                    [
                        record
                        for record in dev_results
                        if record["variant"] == "rrf"
                    ],
                    dev_gold,
                ),
                "rrf_ce": _error_counts(
                    [
                        record
                        for record in dev_results
                        if record["variant"] == "rrf_ce"
                    ],
                    dev_gold,
                ),
            },
        },
        "holdout": {
            "n": holdout["n"],
            "n_q_plus": holdout["n_q_plus"],
            "search": holdout["search"],
            "search_by_class": holdout["search_by_class"],
            "answers": holdout["answers"],
            "pairwise": holdout["pairwise"],
            "timings_s": holdout["timings_s"],
            "errors": {
                "dense": _error_counts(
                    [
                        record
                        for record in holdout_results
                        if record["variant"] == "dense"
                    ],
                    holdout_gold,
                ),
                selected: _error_counts(
                    [
                        record
                        for record in holdout_results
                        if record["variant"] == selected
                    ],
                    holdout_gold,
                ),
            },
        },
        "comparison": comparison,
        "holdout_attempts": _compare_holdout_attempts(
            failed_holdout, run_dirs["holdout"]
        ),
    }
    files.write_json(out_dir / "analysis.json", analysis)

    cases = [
        _case(spec, dev_retrieval, answers_dev, dev_gold, corpus)
        for spec in CASE_SPECS
    ]
    files.write_json(out_dir / "cases.json", {"cases": cases})

    dh = comparison["holdout"]
    dd = comparison["dev"]
    selected_dev = dd[selected]
    selected_hold = dh[selected]
    dense_dev = dd["dense"]
    dense_hold = dh["dense"]
    comp = analysis["dev"]["complementarity"]["by_depth"]
    t = analysis["dev"]["timings_s"]

    lines = [
        "# ЛР3. Улучшение поиска в RAG",
        "",
        "## 1. Постановка и правило выбора",
        "",
        "Четыре варианта поиска получают три документа в контекст ЛР2: "
        "Dense по индексу BGE-small, BM25 по полному корпусу, RRF объединяет "
        "их top-20, RRF + CE переставляет те же 20 кандидатов cross-encoder "
        "`cross-encoder/ms-marco-MiniLM-L6-v2`. Модель, промпты, JSON Schema, "
        "бюджет 2048 токенов и k=3 не менялись.",
        "",
        "Правило выбора: наибольший `MRR@3` на dev среди BM25, RRF и RRF+CE; "
        "при равенстве — `Recall@3`; затем более простой вариант. Dense — "
        "baseline, но в выборе не участвует. Выбор записан в "
        f"`reports/lr3/selection.json` до holdout: **{selected}**. "
        f"Коммит выбора `{selection['commit'][:7]}`, хеш конфига "
        f"`{selection['config']['sha256'][:12]}…`.",
        "",
        f"Официальные запуски: `{run_dirs['exp1'].name}` (dev, поиск), "
        f"`{run_dirs['exp2'].name}` (dev, ответы), "
        f"`{run_dirs['holdout'].name}` (holdout).",
        "",
        "## 2. Прогноз и проверки",
        "",
        "Прогноз был записан до запусков в `reports/lr3/forecast.md`. Он "
        "верно предсказал состав групп дополнительности и то, что RRF+CE "
        "улучшит поиск, но ответы изменятся лишь на нескольких утверждениях. "
        "Ошибки прогноза: медиана cross-encoder оказалась 1.27 с вместо "
        "0.15–0.40 с; Dense на dev — 15.7 мс, ниже ожидаемых 20–60 мс; "
        "BM25 дал `Hit@3 = 0.783` чуть ниже диапазона 0.80–0.88. "
        "Рост `MRR@3` RRF+CE над RRF составил 0.050, в ожидаемом диапазоне "
        "0.04–0.10.",
        "",
        "Проверки перед опытами пройдены: BM25 на трёх коротких документах "
        "совпал с ручным расчётом, RRF на игрушечном примере поставил "
        "документ из обоих списков выше, после CE множество top-20 и "
        "`Recall@20` не изменились, первые пять Dense совпали с ЛР2 на "
        "90/90 утверждениях. Результаты — `reports/lr3/checks.json`.",
        "",
        "## 3. Поиск на dev",
        "",
        "| Вариант | Hit@3 | Recall@3 | MRR@3 | Recall@20 |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in ("dense", "bm25", "rrf", "rrf_ce"):
        values = _search_values(exp2["search"], variant)
        lines.append(
            f"| {VARIANT_TITLES[variant]} | {_fmt(values['hit@3'])} | "
            f"{_fmt(values['recall@3'])} | {_fmt(values['mrr@3'])} | "
            f"{_fmt(values['recall@20'])} |"
        )
    lines.extend(
        [
            "",
            "BM25 уступает Dense по всем метрикам, но RRF выигрывает у Dense "
            "по `Hit@3` и `Recall@3` и немного проигрывает по `MRR@3`: "
            "объединение поднимает найденные обоими методами документы, но "
            "размывает первый результат Dense. CE возвращает `MRR@3` выше "
            "Dense и RRF за счёт перестановки тех же 20 кандидатов.",
            "",
            "### Дополнительность",
            "",
            "| Глубина | Оба (n; RRF top-3) | Только Dense | Только BM25 | Ни один |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for depth in ("3", "20"):
        cells = []
        for group in ("both", "dense_only", "bm25_only", "neither"):
            value = comp[depth][group]
            fraction = (
                "—"
                if value["rrf_top3_fraction"] is None
                else f"{100 * value['rrf_top3_fraction']:.0f}%"
            )
            cells.append(f"{value['n']}; {value['rrf_top3']} ({fraction})")
        lines.append(f"| {depth} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "На глубине 3 Dense и BM25 не полностью дублируют друг друга: "
            "7 утверждений нашёл только Dense, 3 — только BM25. RRF сохранил "
            "все 3 находки BM25, но только 4 из 7 находок Dense. Поэтому "
            "`MRR@3` RRF ниже, чем у Dense. На глубине 20 различие почти "
            "исчезает: 57 из 60 нашли оба метода, по одному — только один "
            "из них, и ещё одно не нашёл никто.",
            "",
            "## 4. Ответы на dev",
            "",
            "| Вариант | Accuracy | Верные ссылки | Передано обоснование | parse ok / invalid_ref / invalid_json |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for variant, key in (
        ("Dense (ЛР2)", "dense_lr2"),
        ("RRF", "rrf"),
        ("RRF + CE", "rrf_ce"),
    ):
        values = exp2["answers"][key]
        status = values["answers"]["parse_status"]
        lines.append(
            f"| {variant} | {_pct(values['answers']['accuracy'])} | "
            f"{_pct(values['answers']['reference_accuracy'])} | "
            f"{_pct(values['context']['rationale_passed_fraction'])} | "
            f"{status.get('ok', 0)} / {status.get('invalid_reference', 0)} / "
            f"{status.get('invalid_json', 0)} |"
        )
    lines.extend(
        [
            "",
            "| Переход | Исправлено | Испорчено | Итог |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, label in (
        ("rrf_vs_dense", "RRF против Dense"),
        ("rrf_ce_vs_dense", "RRF + CE против Dense"),
        ("rrf_ce_vs_rrf", "RRF + CE против RRF"),
    ):
        value = exp2["pairwise"][name]
        lines.append(
            f"| {label} | {value['fixed']} | {value['broke']} | "
            f"{value['net_correct']:+d} ({_pp(value['net_pp'] / 100)}) |"
        )
    lines.extend(
        [
            "",
            "Главный разрыв — между поиском и ответом. RRF+CE поднял `MRR@3` "
            "с 0.747 до 0.797 и `Recall@3` с 0.827 до 0.844, но Accuracy "
            "упала с 32.2% до 27.8%, а относительно Dense переход дал "
            "+4 исправленных и −2 испорченных. RRF без CE дал +6 и 0. "
            "Модель по-прежнему почти всегда выбирает CONTRADICT: на dev "
            f"`SUPPORT` recall {_pct(exp2['answers']['rrf_ce']['answers']['class_recall']['SUPPORT'])}, "
            f"`CONTRADICT` {_pct(exp2['answers']['rrf_ce']['answers']['class_recall']['CONTRADICT'])}, "
            f"`NOINFO` {_pct(exp2['answers']['rrf_ce']['answers']['class_recall']['NOINFO'])}.",
            "",
            "Ошибки по этапам на dev: у RRF+CE 25 верных, 1 `evidence_not_found`, "
            "3 `found_not_passed`, 31 `passed_wrong` и 30 `noinfo_wrong`. "
            "У RRF — 29 верных, 1, 3, 27 и 30 соответственно. То есть "
            "переранжирование чаще доводит обоснование до контекста, но "
            "модель чаще ошибается в метке, а не теряет документ.",
            "",
            "## 5. Время поиска",
            "",
            "| Этап | Медиана | Диапазон | Комментарий |",
            "|---|---:|---:|---|",
            f"| Dense | {1000 * t['dense']['median_s']:.1f} мс | "
            f"{1000 * t['dense']['min_s']:.1f}–{1000 * t['dense']['max_s']:.1f} мс | "
            "кодирование запроса + поиск |",
            f"| BM25 | {1000 * t['bm25']['median_s']:.1f} мс | "
            f"{1000 * t['bm25']['min_s']:.1f}–{1000 * t['bm25']['max_s']:.1f} мс | "
            "поиск по инвертированному индексу |",
            f"| RRF | {1000 * t['rrf']['median_s']:.3f} мс | "
            f"{1000 * t['rrf']['min_s']:.3f}–{1000 * t['rrf']['max_s']:.3f} мс | "
            "слияние двух списков |",
            f"| Rerank | {1000 * t['rerank']['median_s']:.0f} мс | "
            f"{1000 * t['rerank']['min_s']:.0f}–{1000 * t['rerank']['max_s']:.0f} мс | "
            "20 пар cross-encoder |",
            "",
            "Dense, BM25 и RRF измерены в exp1 без загруженного CE; rerank — "
            "в exp2. В exp2 с уже загруженным CE медиана Dense была 196 мс, "
            "поэтому для изолированной оценки Dense использую exp1. "
            "Переранжирование занимает 1800 пар на dev и 4200 на holdout; "
            "на CPU это самая дорогая часть поиска, примерно в 80 раз "
            "медленнее Dense. Итоговые стенды: exp2 — 846 с на 180 вызовов, "
            "holdout — 2726 с на 420 вызовов.",
            "",
            "## 6. Holdout: итог рядом с dev",
            "",
            f"Выбранный вариант — **{selected}**. Он применён к holdout; "
            "после просмотра результаты и параметры не менялись. Технический "
            "повтор после сбоя логирования описан в п. 9. В таблице метрики "
            "поиска — на Q+ (60 dev / 128 holdout), ответы — на всех "
            "90 / 210 утверждениях.",
            "",
            "| Метрика | Dense dev | RRF+CE dev | Dense holdout | RRF+CE holdout |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    metric_specs = (
        ("Hit@3", "hit@3"),
        ("Recall@3", "recall@3"),
        ("MRR@3", "mrr@3"),
        ("Recall@20", "recall@20"),
        ("Accuracy", "accuracy"),
        ("Верные ссылки", "reference"),
        ("Передано обоснование", "rationale"),
    )
    for label, key in metric_specs:
        lines.append(
            f"| {label} | {_fmt(dense_dev[key])} | {_fmt(selected_dev[key])} | "
            f"{_fmt(dense_hold[key])} | {_fmt(selected_hold[key])} |"
        )
    lines.extend(
        [
            "",
            "На holdout преимущество поиска сохранилось, но стало меньше: "
            f"разница RRF+CE − Dense по `MRR@3` — {_signed(dd['rrf_ce_minus_dense']['mrr@3'])} "
            f"на dev и {_signed(dh['rrf_ce_minus_dense']['mrr@3'])} на holdout; "
            f"по Accuracy — {_pp(dd['rrf_ce_minus_dense']['accuracy'])} на dev и "
            f"{_pp(dh['rrf_ce_minus_dense']['accuracy'])} на holdout. "
            f"Поиск: `Hit@3` {_fmt(dense_hold['hit@3'])} → "
            f"{_fmt(selected_hold['hit@3'])}, `Recall@3` "
            f"{_fmt(dense_hold['recall@3'])} → {_fmt(selected_hold['recall@3'])}. "
            "Ответы: Accuracy почти не изменилась, верные ссылки стали хуже "
            f"({_pct(dense_hold['reference'])} → {_pct(selected_hold['reference'])}), "
            "а доля переданного обоснования выросла.",
            "",
            "| Класс holdout | n | Dense Hit@3 | RRF+CE Hit@3 | Dense MRR@3 | RRF+CE MRR@3 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label, values in holdout["search_by_class"].items():
        dense = values["variants"]["dense"]
        ce = values["variants"][selected]
        lines.append(
            f"| {label} | {values['n']} | {_fmt(dense['hit'])} | "
            f"{_fmt(ce['hit'])} | {_fmt(dense['mrr'])} | {_fmt(ce['mrr'])} |"
        )
    transition = holdout["pairwise"][f"{selected}_vs_dense"]
    lines.extend(
        [
            "",
            f"Попарно на holdout RRF+CE исправил {transition['fixed']} и "
            f"испортил {transition['broke']} из {transition['n']} "
            f"({transition['net_correct']:+d}, {_pp(transition['net_pp'] / 100)}). "
            "Ошибки по этапам: Dense — 29 верных, 2 `evidence_not_found`, "
            "8 `found_not_passed`, 89 `passed_wrong`, 82 `noinfo_wrong`; "
            "RRF+CE — 28 верных, 0, 4, 96 и 82. Поиск перестал терять "
            "обоснование, но ошибки сместились в генерацию и выбор метки. "
            "Итоговая рекомендация: **оставить Dense для качества ответа**, "
            "если поиск не является самостоятельной целью. RRF+CE даёт "
            "лучший поиск, но не переносит выигрыш в Accuracy и стоит "
            "около 1.4 с на запрос; для внедрения нужен более подходящий "
            "биомедицинский reranker или работа с калибровкой модели.",
            "",
            "## 7. Четыре случая",
            "",
        ]
    )
    for case in cases:
        lines.extend(_case_report(case))
        if case["kind"] == "rerank_answer":
            lines.extend(
                [
                    "Обратный пример: claim 1370 — CE поднял документ "
                    "`21553394`, модель сослалась на предложения 6–10, "
                    "а передан был текст 0–8; ссылка стала "
                    "`invalid_reference`, хотя RRF был валиден. Это "
                    "объясняет часть переходов «испорчено».",
                    "",
                ]
            )
    lines.extend(
        [
            "## 8. Выводы",
            "",
            "1. BM25 полезен там, где редкий токен совпадает точно, но на "
            "всём dev уступает Dense; выигрыш даёт объединение, а не замена.",
            "2. RRF повышает recall и `Hit@3`, но может размывать первый "
            "результат Dense: на dev `MRR@3` ниже, чем у Dense, а ответы "
            "улучшаются сильнее, чем у RRF+CE (+6 исправлений против −2).",
            "3. Cross-encoder улучшает поисковые метрики и устраняет "
            "`evidence_not_found` на holdout, но не улучшает Accuracy: "
            "ошибки уходят в `passed_wrong` и выбор метки.",
            "4. Правило выбрало RRF+CE по `MRR@3`, однако итоговая "
            "рекомендация может оставить Dense: на holdout Accuracy "
            "снизилась на 0.5 п.п., а стоимость выросла примерно в 80 раз.",
            "",
            "Ограничения: один выбранный вариант на holdout, 210 утверждений, CPU, "
            "одна модель и один генератор; все ответы зависят от "
            "калибровки Qwen и парсера ссылок. Это не отменяет главный "
            "вывод: улучшение поиска само по себе не гарантирует "
            "улучшения ответа.",
            "",
        ]
    )
    if analysis["holdout_attempts"] is not None:
        attempt = analysis["holdout_attempts"]
        lines.extend(
            [
                "## 9. Повторный holdout",
                "",
                f"Первый holdout `{attempt['failed_run']}` завершил расчёты, "
                "но упал на финальном логировании MLflow из-за символа `@` "
                "в имени метрики. Исходный результат сохранён. После "
                "исправления только логирования выполнен повторный запуск "
                f"`{attempt['repeat_run']}` (серия `{attempt['repeat_series']}`). "
                f"Совпадение поисковых метрик: {attempt['same_search_metrics']}, "
                f"ответов: {attempt['same_answer_metrics']}, pairwise: "
                f"{attempt['same_pairwise']}, ранжирований: "
                f"{attempt['same_rankings']}, ответов построчно: "
                f"{attempt['same_results']}. Параметры и конфиг не менялись.",
                "",
            ]
        )

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp1")
    parser.add_argument("--exp2")
    parser.add_argument("--holdout")
    parser.add_argument("--failed-holdout")
    parser.add_argument("--out-dir", default="reports/lr3")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = files.PROJECT_ROOT / out_dir
    run_dirs = {
        "exp1": _run_dir(args.exp1, "1"),
        "exp2": _run_dir(args.exp2, "2"),
        "holdout": _run_dir(args.holdout, "3"),
    }
    failed = None
    if args.failed_holdout:
        failed = _run_dir(args.failed_holdout, "3")
    else:
        failed_candidates = [
            path
            for path in files.RUNS_DIR.glob("lr3-experiment3-*")
            if path != run_dirs["holdout"]
            and (path / files.METADATA_FILENAME).exists()
            and files.read_json(path / files.METADATA_FILENAME).get("status")
            == "failed"
        ]
        failed = max(failed_candidates) if failed_candidates else None
    build(run_dirs, failed, out_dir)


if __name__ == "__main__":
    main()
