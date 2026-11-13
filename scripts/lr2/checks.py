"""Проверки ЛР2 перед опытами (п. 3.4)."""

import json

import numpy as np

from ai_agent import files
from ai_agent.labs.lr2 import index, response, retrieval

ROOT = files.PROJECT_ROOT


def _known_retrieval_example() -> dict:
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
    return {
        f"k{k}": retrieval.retrieval_scores([1], rankings, gold, k)
        for k in (1, 3)
    }


def _parser_checks() -> dict:
    passed = [{"doc_id": "1", "sentences": [0, 1]}]
    cases = {
        "unknown_label": '{"label": "MAYBE", "doc_id": null, "sentences": []}',
        "extra_field": (
            '{"label": "NOINFO", "doc_id": null, "sentences": [], "x": 1}'
        ),
        "unpassed_document": (
            '{"label": "SUPPORT", "doc_id": "2", "sentences": [0]}'
        ),
        "unpassed_sentence": (
            '{"label": "SUPPORT", "doc_id": "1", "sentences": [9]}'
        ),
        "valid": '{"label": "SUPPORT", "doc_id": "1", "sentences": [0, 1]}',
    }
    return {
        name: response.parse_response(raw, passed, "rag")["status"]
        for name, raw in cases.items()
    }


def main() -> None:
    record = index.load()
    matrix = record["matrix"]
    norms = np.linalg.norm(matrix, axis=1)
    text = (
        "Of the 32,441 appendix samples 16 were positive for abnormal PrP, "
        "indicating an overall prevalence of 493 per million population."
    )
    model = index.load_model_for_search(record)
    query = index.encode_queries(model, [text], record["query_prefix"])[0]
    document = index.encode_documents(model, [text])[0]
    plain_query = index.encode_queries(model, [text], "")[0]

    passed = [{"doc_id": "1", "sentences": [0]}]
    gold_record = {
        "label": "SUPPORT",
        "evidence": {"1": [{"sentences": [0, 1], "label": "SUPPORT"}]},
    }
    from ai_agent.labs.lr2.evaluation import rationale_passed

    checks = {
        "index": {
            "model": record["passport"]["model"],
            "revision": record["passport"]["revision"],
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
            "norms_min": float(norms.min()),
            "norms_max": float(norms.max()),
            "clipped_fraction": record["passport"]["clipped_fraction"],
            "self_similarity_plain": float(plain_query @ document),
            "self_similarity_with_query_prefix": float(query @ document),
        },
        "known_retrieval_example": _known_retrieval_example(),
        "parser": _parser_checks(),
        "rationale_passed": {
            "full_sentence_list": rationale_passed(
                {"passed": [{"doc_id": "1", "sentences": [0, 1]}]},
                gold_record,
            ),
            "removed_sentence": rationale_passed(
                {"passed": passed}, gold_record
            ),
        },
    }
    path = ROOT / "reports" / "lr2" / "checks.json"
    files.write_json(path, checks)
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
