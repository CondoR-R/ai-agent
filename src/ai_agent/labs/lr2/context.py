"""Сборка контекста в бюджете токенов для ЛР2."""

from dataclasses import dataclass

from . import prompts


@dataclass
class Context:
    """Промпт и документы, которые действительно в него вошли."""

    messages: list[dict[str, str]]
    prompt: str
    n_tokens: int
    passed: list[dict]
    stop_reason: str


def _candidate(
    mode: str,
    claim: str,
    selected: list[tuple[dict, list[int]]],
) -> list[dict[str, str]]:
    documents = [document for document, _ in selected]
    sentence_ids = [ids for _, ids in selected] if documents else None
    return prompts.messages(mode, claim, documents, sentence_ids)


def _passport(selected: list[tuple[dict, list[int]]]) -> list[dict]:
    return [
        {
            "doc_id": str(document["doc_id"]),
            "title": str(document["title"]).strip(),
            "sentences": list(sentence_ids),
        }
        for document, sentence_ids in selected
    ]


def assemble(
    mode: str,
    claim: str,
    candidates: list[tuple[dict, list[int]]],
    tokenizer,
    budget: int = 2048,
) -> Context:
    """Собирает документы по рангу, пока помещается очередной фрагмент.

    Если документ не помещается целиком, в контекст идут его первые
    предложения, после чего сборка останавливается. Документ без единого
    предложения не добавляется.
    """
    selected: list[tuple[dict, list[int]]] = []
    stop_reason = "all_documents"

    base_messages = _candidate(mode, claim, [])
    if prompts.count_tokens(tokenizer, base_messages) > budget:
        raise ValueError(f"Утверждение не помещается в бюджет {budget} токенов")

    for document, sentence_ids in candidates:
        if not sentence_ids:
            continue

        full = [*selected, (document, sentence_ids)]
        full_messages = _candidate(mode, claim, full)
        if prompts.count_tokens(tokenizer, full_messages) <= budget:
            selected = full
            continue

        stop_reason = "budget"
        partial_ids: list[int] = []
        for sentence_id in sentence_ids:
            trial_ids = [*partial_ids, sentence_id]
            trial = [*selected, (document, trial_ids)]
            trial_messages = _candidate(mode, claim, trial)
            if prompts.count_tokens(tokenizer, trial_messages) > budget:
                break
            partial_ids = trial_ids

        if partial_ids:
            selected.append((document, partial_ids))
        break

    final_messages = _candidate(mode, claim, selected)
    return Context(
        messages=final_messages,
        prompt=prompts.render(tokenizer, final_messages),
        n_tokens=prompts.count_tokens(tokenizer, final_messages),
        passed=_passport(selected),
        stop_reason=stop_reason,
    )


def from_ranked(
    mode: str,
    claim: dict,
    ranked: list[dict],
    corpus: dict[str, dict],
    tokenizer,
    budget: int = 2048,
) -> Context:
    """Контекст RAG из top-k поиска (целые аннотации по рангу)."""
    candidates = []
    for item in ranked:
        document = corpus[item["doc_id"]]
        candidates.append((document, list(range(len(document["abstract"])))))
    return assemble(mode, claim["claim"], candidates, tokenizer, budget)


def from_oracle(
    mode: str,
    claim: dict,
    document: dict,
    sentence_ids: list[int],
    tokenizer,
    budget: int = 2048,
) -> Context:
    """Контекст oracle: эталонный документ и его предложения-обоснования."""
    return assemble(
        mode,
        claim["claim"],
        [(document, list(sentence_ids))],
        tokenizer,
        budget,
    )
