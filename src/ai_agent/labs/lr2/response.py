"""Разбор и проверка JSON-ответа модели ЛР2."""

import json
import re

from .prompts import LABELS

REQUIRED_FIELDS = {"label", "doc_id", "sentences"}


def _passed_map(passed: list[dict]) -> dict[str, set[int]]:
    return {
        str(document["doc_id"]): {int(s) for s in document["sentences"]}
        for document in passed
    }


def _canonical_doc_id(doc_id: str, available: dict[str, set[int]]) -> str:
    """Убирает обёртки вроде 'DOC 123' / ':[DOC 123]', если id однозначен."""
    if doc_id in available:
        return doc_id
    matches = {
        value for value in re.findall(r"\d+", doc_id) if value in available
    }
    return matches.pop() if len(matches) == 1 else doc_id


def parse_response(raw: str | None, passed: list[dict], mode: str) -> dict:
    """Проверяет JSON-схему и ссылки на переданные документы/предложения.

    return: {"parsed": объект или None, "status": ..., "error": ...}
    """
    if not raw:
        return {
            "parsed": None,
            "status": "invalid_json",
            "error": "пустой ответ",
        }

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {
            "parsed": None,
            "status": "invalid_json",
            "error": str(exc),
        }

    if not isinstance(parsed, dict):
        return {
            "parsed": parsed,
            "status": "invalid_schema",
            "error": "ответ не является JSON-объектом",
        }
    if set(parsed) != REQUIRED_FIELDS:
        extra = sorted(set(parsed) - REQUIRED_FIELDS)
        missing = sorted(REQUIRED_FIELDS - set(parsed))
        return {
            "parsed": parsed,
            "status": "invalid_schema",
            "error": f"лишние поля {extra}, нет полей {missing}",
        }

    label = parsed["label"]
    doc_id = parsed["doc_id"]
    sentences = parsed["sentences"]
    if label not in LABELS:
        return {
            "parsed": parsed,
            "status": "invalid_schema",
            "error": f"неизвестная метка {label!r}",
        }
    if doc_id is not None and not isinstance(doc_id, str):
        return {
            "parsed": parsed,
            "status": "invalid_schema",
            "error": "doc_id должен быть строкой или null",
        }
    if not isinstance(sentences, list) or any(
        isinstance(sentence, bool) or not isinstance(sentence, int)
        for sentence in sentences
    ):
        return {
            "parsed": parsed,
            "status": "invalid_schema",
            "error": "sentences должен быть списком целых чисел",
        }
    if any(sentence < 0 for sentence in sentences):
        return {
            "parsed": parsed,
            "status": "invalid_schema",
            "error": "номера предложений не могут быть отрицательными",
        }

    if label == "NOINFO":
        if doc_id is not None or sentences:
            return {
                "parsed": parsed,
                "status": "invalid_schema",
                "error": "для NOINFO нужны doc_id=null и sentences=[]",
            }
        return {"parsed": parsed, "status": "ok", "error": None}

    if mode == "llm_only":
        return {"parsed": parsed, "status": "ok", "error": None}

    if doc_id is None or not sentences:
        return {
            "parsed": parsed,
            "status": "invalid_reference",
            "error": "для SUPPORT/CONTRADICT нужны документ и предложения",
        }

    available = _passed_map(passed)
    doc_id = _canonical_doc_id(doc_id, available)
    parsed["doc_id"] = doc_id
    if doc_id not in available:
        return {
            "parsed": parsed,
            "status": "invalid_reference",
            "error": f"документ {doc_id} не передан модели",
        }
    unknown = sorted(set(sentences) - available[doc_id])
    if unknown:
        return {
            "parsed": parsed,
            "status": "invalid_reference",
            "error": f"предложения {unknown} не переданы для {doc_id}",
        }
    return {"parsed": parsed, "status": "ok", "error": None}
