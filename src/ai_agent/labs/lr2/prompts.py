"""Промпты ЛР2, определения меток и JSON Schema ответа."""

from typing import Any

LABELS = ("SUPPORT", "CONTRADICT", "NOINFO")

LABEL_DEFINITIONS = """\
SUPPORT: the provided evidence supports the claim.
CONTRADICT: the provided evidence contradicts the claim.
NOINFO: the provided evidence does not contain enough information to decide.
"""

SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "verdict",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["label", "doc_id", "sentences"],
            "properties": {
                "label": {"enum": list(LABELS)},
                "doc_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "sentences": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
            },
        },
    },
}

OUTPUT_RULES = """\
Return exactly one JSON object with keys "label", "doc_id", "sentences".
Use sentence numbers exactly as they appear in square brackets.
Cite only documents and sentence numbers that are shown to you. Do not cite
any other sentence. For NOINFO, set "doc_id" to null and "sentences" to [].
"""

SYSTEM_RAG = f"""\
You are a scientific fact-checking assistant.
Use only the documents provided by the user. Do not use outside knowledge.
{LABEL_DEFINITIONS}
{OUTPUT_RULES}\
"""

SYSTEM_ORACLE = f"""\
You are a scientific fact-checking assistant.
Use only the evidence provided by the user. Do not use outside knowledge.
{LABEL_DEFINITIONS}
{OUTPUT_RULES}\
"""

SYSTEM_LLM_ONLY = f"""\
You are a scientific fact-checking assistant.
No documents are provided. Use your own knowledge, and choose NOINFO if you
cannot decide from that knowledge alone.
{LABEL_DEFINITIONS}
{OUTPUT_RULES}\
Because no documents are provided, set "doc_id" to null and "sentences" to [].
"""

SYSTEM = {
    "rag": SYSTEM_RAG,
    "oracle": SYSTEM_ORACLE,
    "llm_only": SYSTEM_LLM_ONLY,
}


def _document_block(document: dict, sentence_ids: list[int]) -> str:
    lines = [f"[DOC {document['doc_id']}] {str(document['title']).strip()}"]
    for sentence_id in sentence_ids:
        lines.append(f"[{sentence_id}] {document['abstract'][sentence_id]}")
    return "\n".join(lines)


def user_prompt(
    claim: str,
    documents: list[dict] | None = None,
    sentence_ids: list[list[int]] | None = None,
) -> str:
    """Текст пользовательского сообщения с необязательными документами."""
    parts = [f"Claim: {claim}"]
    if documents:
        blocks = []
        for index, document in enumerate(documents):
            ids = sentence_ids[index] if sentence_ids else []
            blocks.append(_document_block(document, ids))
        parts.append("Documents:\n\n" + "\n\n".join(blocks))
    else:
        parts.append("No documents are provided.")
    return "\n\n".join(parts)


def messages(
    mode: str,
    claim: str,
    documents: list[dict] | None = None,
    sentence_ids: list[list[int]] | None = None,
) -> list[dict[str, str]]:
    """Сообщения чата для одного режима."""
    return [
        {"role": "system", "content": SYSTEM[mode]},
        {
            "role": "user",
            "content": user_prompt(claim, documents, sentence_ids),
        },
    ]


def count_tokens(tokenizer, chat_messages: list[dict]) -> int:
    """Длина промпта после шаблона диалога, в токенах модели."""
    rendered = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )
    return len(tokenizer.encode(rendered, add_special_tokens=False))


def render(tokenizer, chat_messages: list[dict]) -> str:
    """Полный промпт строкой (для results.jsonl)."""
    return tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )
