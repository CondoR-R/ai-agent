"""Общая подготовка запусков ЛР2: сервер, индекс, токенизатор, вызов."""

import time
from dataclasses import dataclass
from pathlib import Path

from ai_agent import console, files, metadata
from ai_agent.labs.lr1.endpoint import load_endpoint
from ai_agent.labs.lr1.server import collect_server_info

from . import client, context, data, evaluation, index, response


@dataclass
class Session:
    """Всё, что нужно серии ЛР2."""

    endpoint: object
    client: object
    index_record: dict
    query_model: object
    tokenizer: object
    corpus: dict
    claims: list[dict]
    gold: dict
    budget: int
    max_tokens: int


def _load_tokenizer(endpoint, info: dict):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        endpoint.model, revision=info["model"].get("revision")
    )


def setup(cfg: dict, run_dir: Path) -> Session:
    """Читает endpoint и данные, добавляет паспорт в metadata.json."""
    endpoint = load_endpoint(files.PROJECT_ROOT / cfg["endpoint_path"])
    info = collect_server_info(endpoint, cfg["server"])
    index_record = index.load(cfg)
    tokenizer = _load_tokenizer(endpoint, info)
    claims = data.dev_claims()
    corpus = data.corpus_by_id()

    metadata.add(
        run_dir,
        {
            "endpoint": endpoint.to_record(),
            **info,
            "index": index_record["passport"],
            "prompt_budget": int(cfg["budget_tokens"]),
            "generation_max_tokens": int(cfg["generation"]["max_tokens"]),
            "data": {
                "manifest_key": data.MANIFEST_KEY,
                "archive_sha256": data.ARCHIVE_SHA256,
                "dev_ids": [claim["id"] for claim in claims],
                "n_dev": len(claims),
            },
        },
    )
    console.info(
        f"сервер: {info['server']['name']} {info['server']['version']}, "
        f"модель {info['model']['repo']}, контекст "
        f"{info['server']['max_model_len']}"
    )
    console.info(
        f"индекс: {index_record['passport']['model']} "
        f"({index_record['passport']['dimension']}d), "
        f"обрезано {index_record['passport']['clipped_fraction']:.1%}"
    )
    console.info(
        f"данные: dev={len(claims)}, Q+={sum(1 for c in claims if c['evidence'])}"
    )
    return Session(
        endpoint=endpoint,
        client=client.make_client(endpoint),
        index_record=index_record,
        query_model=index.load_model_for_search(index_record),
        tokenizer=tokenizer,
        corpus=corpus,
        claims=claims,
        gold=evaluation.gold_by_id(claims),
        budget=int(cfg["budget_tokens"]),
        max_tokens=int(cfg["generation"]["max_tokens"]),
    )


def _retrieve(session: Session, claim: dict) -> tuple[list[dict], float]:
    started = time.perf_counter()
    ranked = index.search_claim(
        claim, session.index_record, session.query_model, top_k=5
    )
    return ranked, time.perf_counter() - started


def _oracle_input(session: Session, claim: dict) -> tuple[dict, list[int]]:
    evidence = session.gold[claim["id"]]["evidence"]
    doc_id, rows = next(iter(evidence.items()))
    document = session.corpus[doc_id]
    return document, list(rows[0]["sentences"])


def _record(
    session: Session,
    claim: dict,
    mode: str,
    *,
    k: int | None,
    ranked: list[dict],
    ctx: context.Context,
    search_s: float,
    completion: dict,
) -> dict:
    if completion["status"] == "ok":
        parsed_result = response.parse_response(
            completion["raw_response"], ctx.passed, mode
        )
    else:
        parsed_result = {
            "parsed": None,
            "status": "call_error",
            "error": completion["error"],
        }
    return {
        "claim_id": claim["id"],
        "claim": claim["claim"],
        "mode": mode,
        "k": k,
        "retrieved": ranked,
        "passed": ctx.passed,
        "stop_reason": ctx.stop_reason,
        "prompt": ctx.prompt,
        "prompt_tokens_expected": ctx.n_tokens,
        "prompt_tokens": completion["prompt_tokens"],
        "raw_response": completion["raw_response"],
        "parsed": parsed_result["parsed"],
        "parse_status": parsed_result["status"],
        "parse_error": parsed_result["error"],
        "finish_reason": completion["finish_reason"],
        "completion_tokens": completion["completion_tokens"],
        "search_s": search_s,
        "generation_s": completion["generation_s"],
    }


def run_llm_only(session: Session, claim: dict, results_path) -> dict:
    """Один вызов LLM-only без поиска."""
    ctx = context.assemble(
        "llm_only", claim["claim"], [], session.tokenizer, session.budget
    )
    completion = client.chat(
        session.client,
        session.endpoint,
        ctx.messages,
        max_tokens=session.max_tokens,
    )
    record = _record(
        session,
        claim,
        "llm_only",
        k=None,
        ranked=[],
        ctx=ctx,
        search_s=0.0,
        completion=completion,
    )
    files.append_jsonl(results_path, record)
    return record


def run_rag(session: Session, claim: dict, k: int, results_path) -> dict:
    """Один вызов RAG: top-5 поиска, в промпт идут первые k документов."""
    ranked, search_s = _retrieve(session, claim)
    ctx = context.from_ranked(
        "rag",
        claim,
        ranked[:k],
        session.corpus,
        session.tokenizer,
        session.budget,
    )
    completion = client.chat(
        session.client,
        session.endpoint,
        ctx.messages,
        max_tokens=session.max_tokens,
    )
    record = _record(
        session,
        claim,
        "rag",
        k=k,
        ranked=ranked,
        ctx=ctx,
        search_s=search_s,
        completion=completion,
    )
    files.append_jsonl(results_path, record)
    return record


def run_oracle(session: Session, claim: dict, results_path) -> dict:
    """Один вызов oracle: эталонный документ и предложение-обоснование."""
    document, sentence_ids = _oracle_input(session, claim)
    ctx = context.from_oracle(
        "oracle",
        claim,
        document,
        sentence_ids,
        session.tokenizer,
        session.budget,
    )
    completion = client.chat(
        session.client,
        session.endpoint,
        ctx.messages,
        max_tokens=session.max_tokens,
    )
    record = _record(
        session,
        claim,
        "oracle",
        k=None,
        ranked=[],
        ctx=ctx,
        search_s=0.0,
        completion=completion,
    )
    files.append_jsonl(results_path, record)
    return record


def warmup(session: Session, run_dir: Path) -> dict:
    """Один прогревочный вызов на claims_train, вне results.jsonl."""
    train = data.load_claims_train()
    claim = next(claim for claim in train if claim["claim"])
    ranked, search_s = _retrieve(session, claim)
    ctx = context.from_ranked(
        "rag",
        claim,
        ranked[:3],
        session.corpus,
        session.tokenizer,
        session.budget,
    )
    completion = client.chat(
        session.client,
        session.endpoint,
        ctx.messages,
        max_tokens=session.max_tokens,
    )
    record = _record(
        session,
        claim,
        "warmup",
        k=3,
        ranked=ranked,
        ctx=ctx,
        search_s=search_s,
        completion=completion,
    )
    record["warmup"] = True
    files.write_json(run_dir / "warmup.json", record)
    console.info(
        f"прогрев: поиск={search_s * 1000:.0f} мс, "
        f"промпт={ctx.n_tokens} ток., ответ={completion['status']}"
    )
    return record
