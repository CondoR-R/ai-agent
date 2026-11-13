"""Проверки ЛР2 без сервера: метрики, контекст, разбор ответа."""

import pytest

from ai_agent.labs.lr2 import (
    context,
    data,
    evaluation,
    prompts,
    response,
    retrieval,
)


class WordTokenizer:
    """Токен = слово; шаблон диалога делает длину предсказуемой."""

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids):
        return " ".join(ids)

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True
    ):
        body = "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
            for m in messages
        )
        return body + "<|im_start|>assistant\n"


def test_claim_label_and_split_counts():
    claims = data.load_claims_dev()
    assert data.label_counts(claims) == {
        "SUPPORT": 124,
        "CONTRADICT": 64,
        "NOINFO": 112,
    }
    entry = data.load_entry()
    assert len(entry["split"]["dev"]) == 90
    assert len(entry["split"]["holdout"]) == 210
    assert data.label_counts(data.dev_claims(claims)) == {
        "SUPPORT": 30,
        "CONTRADICT": 30,
        "NOINFO": 30,
    }


def test_retrieval_known_example():
    gold = {
        1: {
            "label": "SUPPORT",
            "evidence": {
                "a": [{"sentences": [0], "label": "SUPPORT"}],
                "b": [{"sentences": [1], "label": "SUPPORT"}],
            },
        }
    }
    rankings = {
        1: [
            {"doc_id": "y", "score": 0.9, "rank": 1},
            {"doc_id": "x", "score": 0.8, "rank": 2},
            {"doc_id": "a", "score": 0.7, "rank": 3},
        ]
    }
    metrics = retrieval.retrieval_scores([1], rankings, gold, 3)
    assert metrics["hit"] == 1
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["mrr"] == pytest.approx(1 / 3)
    at_one = retrieval.retrieval_scores([1], rankings, gold, 1)
    assert at_one["hit"] == 0
    assert at_one["recall"] == 0
    assert at_one["mrr"] == 0


def _doc(doc_id, title, abstract):
    return {"doc_id": doc_id, "title": title, "abstract": abstract}


def test_context_stops_after_partial_document():
    tokenizer = WordTokenizer()
    claim = "A short claim."
    doc1 = _doc(1, "One", ["first sentence", "second sentence"])
    doc2 = _doc(2, "Two", ["other sentence"])
    candidates = [(doc1, [0, 1]), (doc2, [0])]

    one_sentence = context.assemble(
        "rag", claim, [(doc1, [0])], tokenizer, budget=10_000
    )
    budget = one_sentence.n_tokens
    got = context.assemble("rag", claim, candidates, tokenizer, budget=budget)
    assert got.passed == [{"doc_id": "1", "title": "One", "sentences": [0]}]
    assert got.stop_reason == "budget"


def test_context_drops_document_if_no_sentence_fits():
    tokenizer = WordTokenizer()
    claim = "A short claim."
    doc1 = _doc(1, "One", ["first sentence", "second sentence"])
    base = context.assemble("rag", claim, [], tokenizer)
    got = context.assemble(
        "rag", claim, [(doc1, [0, 1])], tokenizer, budget=base.n_tokens
    )
    assert got.passed == []
    assert got.stop_reason == "budget"


def test_response_rejects_unknown_label_extra_field_and_bad_reference():
    passed = [{"doc_id": "1", "sentences": [0, 1]}]
    assert prompts.messages("rag", "claim")
    bad_label = '{"label": "MAYBE", "doc_id": null, "sentences": []}'
    assert response.parse_response(bad_label, passed, "rag")["status"] == (
        "invalid_schema"
    )
    extra = '{"label": "NOINFO", "doc_id": null, "sentences": [], "x": 1}'
    assert response.parse_response(extra, passed, "rag")["status"] == (
        "invalid_schema"
    )
    wrong_doc = '{"label": "SUPPORT", "doc_id": "2", "sentences": [0]}'
    assert response.parse_response(wrong_doc, passed, "rag")["status"] == (
        "invalid_reference"
    )
    wrong_sentence = '{"label": "SUPPORT", "doc_id": "1", "sentences": [9]}'
    assert response.parse_response(wrong_sentence, passed, "rag")["status"] == (
        "invalid_reference"
    )
    ok = '{"label": "SUPPORT", "doc_id": "1", "sentences": [0, 1]}'
    assert response.parse_response(ok, passed, "rag")["status"] == "ok"


def test_response_normalizes_decorated_doc_id():
    passed = [{"doc_id": "17628888", "sentences": [7]}]
    for raw_id in ("DOC 17628888", ":[DOC 17628888]", "log_17628888"):
        raw = (
            '{"label": "SUPPORT", "doc_id": "' + raw_id + '", "sentences": [7]}'
        )
        parsed = response.parse_response(raw, passed, "oracle")
        assert parsed["status"] == "ok"
        assert parsed["parsed"]["doc_id"] == "17628888"


def test_rationale_passed_detects_removed_sentence():
    gold_record = {
        "label": "SUPPORT",
        "evidence": {"1": [{"sentences": [0, 1], "label": "SUPPORT"}]},
    }
    full = {"passed": [{"doc_id": "1", "sentences": [0, 1]}]}
    partial = {"passed": [{"doc_id": "1", "sentences": [0]}]}
    assert evaluation.rationale_passed(full, gold_record)
    assert not evaluation.rationale_passed(partial, gold_record)


def test_answer_metrics_counts_invalid_as_wrong_and_transitions():
    gold = {
        1: {"label": "SUPPORT", "evidence": {"1": []}},
        2: {"label": "NOINFO", "evidence": {}},
    }
    before = [
        {
            "claim_id": 1,
            "parse_status": "ok",
            "parsed": {"label": "NOINFO"},
        },
        {
            "claim_id": 2,
            "parse_status": "ok",
            "parsed": {"label": "NOINFO"},
        },
    ]
    after = [
        {
            "claim_id": 1,
            "parse_status": "invalid_json",
            "parsed": None,
        },
        {
            "claim_id": 2,
            "parse_status": "ok",
            "parsed": {"label": "NOINFO"},
        },
    ]
    metrics = evaluation.answer_metrics(before, gold, "llm_only")
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["class_recall"]["NOINFO"] == pytest.approx(1.0)
    transition = evaluation.pairwise(before, after, gold)
    assert transition["same_correct"] == 1
    assert transition["same_wrong"] == 1
    assert transition["net_correct"] == 0
