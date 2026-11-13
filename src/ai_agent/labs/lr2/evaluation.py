"""Оценка ответов и разбор ошибок ЛР2."""

from collections import Counter

from .prompts import LABELS

POSITIVE_LABELS = {"SUPPORT", "CONTRADICT"}
ERROR_CATEGORIES = (
    "evidence_not_found",
    "found_not_passed",
    "passed_wrong",
    "noinfo_wrong",
)


def gold_by_id(claims: list[dict]) -> dict:
    """Разметка утверждений: метка и evidence (только для оценки)."""
    result = {}
    for claim in claims:
        labels = {
            row["label"] for rows in claim["evidence"].values() for row in rows
        }
        label = "NOINFO" if not claim["evidence"] else labels.pop()
        result[claim["id"]] = {
            "label": label,
            "evidence": {
                str(doc_id): rows for doc_id, rows in claim["evidence"].items()
            },
        }
    return result


def _answer_label(record: dict) -> str | None:
    if record.get("parse_status") != "ok":
        return None
    parsed = record.get("parsed")
    return parsed.get("label") if isinstance(parsed, dict) else None


def _answer_sentences(record: dict) -> set[int]:
    parsed = record.get("parsed")
    if not isinstance(parsed, dict):
        return set()
    return {int(sentence) for sentence in parsed.get("sentences", [])}


def _passed_map(record: dict) -> dict[str, set[int]]:
    return {
        str(document["doc_id"]): {int(s) for s in document["sentences"]}
        for document in record.get("passed", [])
    }


def _retrieved_ids(record: dict) -> set[str]:
    return {str(item["doc_id"]) for item in record.get("retrieved", [])}


def rationale_passed(record: dict, gold_record: dict) -> bool:
    """Есть ли в переданном тексте хотя бы одно эталонное обоснование."""
    passed = _passed_map(record)
    for doc_id, rows in gold_record["evidence"].items():
        for row in rows:
            if set(row["sentences"]) <= passed.get(str(doc_id), set()):
                return True
    return False


def reference_correct(record: dict, gold_record: dict) -> bool | None:
    """Верная ссылка: метка и одно эталонное обоснование того же документа."""
    if gold_record["label"] not in POSITIVE_LABELS:
        return None
    if _answer_label(record) != gold_record["label"]:
        return False
    parsed = record.get("parsed") or {}
    doc_id = str(parsed.get("doc_id"))
    answer = _answer_sentences(record)
    for row in gold_record["evidence"].get(doc_id, []):
        if set(row["sentences"]) <= answer:
            return True
    return False


def answer_metrics(records: list[dict], gold: dict, mode: str) -> dict:
    """Accuracy, полнота по классам и доля верных ссылок."""
    correct = 0
    by_class_total = Counter()
    by_class_correct = Counter()
    statuses = Counter()
    ref_total = 0
    ref_correct = 0

    for record in records:
        gold_record = gold[record["claim_id"]]
        statuses[record.get("parse_status", "unknown")] += 1
        label = _answer_label(record)
        by_class_total[gold_record["label"]] += 1
        if label == gold_record["label"]:
            correct += 1
            by_class_correct[gold_record["label"]] += 1

        if mode != "llm_only" and gold_record["label"] in POSITIVE_LABELS:
            ref_total += 1
            if reference_correct(record, gold_record):
                ref_correct += 1

    n = len(records)
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "class_recall": {
            label: (
                by_class_correct[label] / by_class_total[label]
                if by_class_total[label]
                else None
            )
            for label in LABELS
        },
        "class_total": dict(by_class_total),
        "reference_n": ref_total,
        "reference_accuracy": ref_correct / ref_total if ref_total else None,
        "parse_status": dict(statuses),
    }


def pairwise(before: list[dict], after: list[dict], gold: dict) -> dict:
    """Попарные переходы ответов на одних и тех же claim_id."""
    before_by_id = {record["claim_id"]: record for record in before}
    after_by_id = {record["claim_id"]: record for record in after}
    common = sorted(set(before_by_id) & set(after_by_id))
    fixed = 0
    broke = 0
    same_correct = 0
    same_wrong = 0
    for claim_id in common:
        expected = gold[claim_id]["label"]
        old = _answer_label(before_by_id[claim_id]) == expected
        new = _answer_label(after_by_id[claim_id]) == expected
        if not old and new:
            fixed += 1
        elif old and not new:
            broke += 1
        elif old and new:
            same_correct += 1
        else:
            same_wrong += 1
    n = len(common)
    return {
        "n": n,
        "fixed": fixed,
        "broke": broke,
        "same_correct": same_correct,
        "same_wrong": same_wrong,
        "net_correct": fixed - broke,
        "net_pp": 100 * (fixed - broke) / n if n else 0.0,
    }


def classify_error(record: dict, gold_record: dict) -> str | None:
    """Этап, на котором потерялась информация (None — ответ верный)."""
    if _answer_label(record) == gold_record["label"]:
        return None
    if gold_record["label"] == "NOINFO":
        return "noinfo_wrong"
    relevant = {str(doc_id) for doc_id in gold_record["evidence"]}
    if not (relevant & _retrieved_ids(record)):
        return "evidence_not_found"
    if not rationale_passed(record, gold_record):
        return "found_not_passed"
    return "passed_wrong"


def _gold_snippets(
    claim_id: int,
    gold_record: dict,
    corpus: dict[str, dict],
) -> list[dict]:
    snippets = []
    for doc_id, rows in gold_record["evidence"].items():
        document = corpus.get(str(doc_id), {})
        for row in rows:
            snippets.append(
                {
                    "doc_id": str(doc_id),
                    "title": document.get("title"),
                    "label": row["label"],
                    "sentences": row["sentences"],
                    "text": [
                        document.get("abstract", [])[sentence]
                        for sentence in row["sentences"]
                        if sentence < len(document.get("abstract", []))
                    ],
                }
            )
    return snippets


def _detail(
    record: dict,
    gold_record: dict,
    corpus: dict[str, dict],
    category: str,
) -> dict:
    retrieved = []
    for item in record.get("retrieved", []):
        document = corpus.get(str(item["doc_id"]), {})
        retrieved.append(
            {
                **item,
                "title": document.get("title"),
            }
        )
    return {
        "category": category,
        "claim_id": record["claim_id"],
        "claim": record["claim"],
        "gold_label": gold_record["label"],
        "gold_evidence": _gold_snippets(
            record["claim_id"], gold_record, corpus
        ),
        "retrieved": retrieved,
        "passed": record.get("passed", []),
        "prompt_tokens": record.get("prompt_tokens"),
        "raw_response": record.get("raw_response"),
        "parsed": record.get("parsed"),
        "parse_status": record.get("parse_status"),
    }


def error_analysis(
    records: list[dict], gold: dict, corpus: dict[str, dict]
) -> dict:
    """Счётчики этапов и по одному примеру каждого содержательного случая."""
    counts = Counter()
    examples = {}
    for record in records:
        category = classify_error(record, gold[record["claim_id"]])
        if category is None:
            continue
        counts[category] += 1
        if category not in examples:
            examples[category] = _detail(
                record,
                gold[record["claim_id"]],
                corpus,
                category,
            )
    return {
        "counts": {
            category: counts.get(category, 0) for category in ERROR_CATEGORIES
        },
        "examples": examples,
    }


def context_metrics(records: list[dict], gold: dict) -> dict:
    """Для Q+: доля переданного обоснования, документы и токены."""
    positive = [
        record
        for record in records
        if gold[record["claim_id"]]["label"] in POSITIVE_LABELS
    ]
    n = len(positive)
    passed_n = sum(
        rationale_passed(record, gold[record["claim_id"]])
        for record in positive
    )
    docs = sum(len(record.get("passed", [])) for record in positive)
    tokens = sum(record.get("prompt_tokens_expected", 0) for record in positive)
    return {
        "n": n,
        "rationale_passed": passed_n,
        "rationale_passed_fraction": passed_n / n if n else 0.0,
        "avg_passed_docs": docs / n if n else 0.0,
        "avg_prompt_tokens": tokens / n if n else 0.0,
    }
