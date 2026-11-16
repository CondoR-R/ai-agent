"""Общая подготовка и вызовы экспериментов ЛР3."""

import time
from dataclasses import dataclass
from pathlib import Path

from ai_agent import console, files, metadata
from ai_agent.labs.lr1.endpoint import load_endpoint
from ai_agent.labs.lr1.server import collect_server_info
from ai_agent.labs.lr2 import client, context, data, evaluation, index, response

from . import bm25, dense, fusion, rerank


@dataclass
class Session:
    """Всё, что нужно серии ЛР3."""

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
    answer_k: int
    split: str
    bm25_index: bm25.BM25Index
    reranker: rerank.CrossEncoderReranker | None


def _load_tokenizer(endpoint, info: dict):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        endpoint.model, revision=info["model"].get("revision")
    )


def _lr2_reference(cfg: dict) -> dict | None:
    run = cfg.get("dense_reference_run")
    if not run:
        return None
    run_dir = files.PROJECT_ROOT / run
    results_path = run_dir / files.RESULTS_FILENAME
    meta_path = run_dir / files.METADATA_FILENAME
    if not results_path.exists():
        raise FileNotFoundError(f"Нет результатов Dense ЛР2: {results_path}")
    meta = files.read_json(meta_path) if meta_path.exists() else {}
    return {
        "run": run,
        "commit": meta.get("commit"),
        "results_sha256": files.get_sha256(results_path),
    }


def setup(
    cfg: dict,
    run_dir: Path,
    *,
    split: str = "dev",
    need_reranker: bool = False,
) -> Session:
    """Читает endpoint, данные, индекс и добавляет паспорта в metadata."""
    endpoint = load_endpoint(files.PROJECT_ROOT / cfg["endpoint_path"])
    info = collect_server_info(endpoint, cfg["server"])
    index_record = index.load(cfg)
    tokenizer = _load_tokenizer(endpoint, info)
    claims = data.dev_claims() if split == "dev" else data.holdout_claims()
    corpus = data.corpus_by_id()
    gold = evaluation.gold_by_id(claims)
    retrieval_cfg = cfg["retrieval"]
    bm25_index = bm25.BM25Index.from_corpus(
        corpus,
        k1=float(retrieval_cfg["bm25"]["k1"]),
        b=float(retrieval_cfg["bm25"]["b"]),
    )

    reranker = None
    reranker_cfg = cfg.get("reranker") or {}
    if need_reranker:
        reranker = rerank.CrossEncoderReranker(
            reranker_cfg["model"],
            max_length=int(reranker_cfg["max_length"]),
            batch_size=int(reranker_cfg["batch_size"]),
            device=reranker_cfg.get("device", "cpu"),
        )

    fields = {
        "endpoint": endpoint.to_record(),
        **info,
        "index": index_record["passport"],
        "prompt_budget": int(cfg["budget_tokens"]),
        "generation_max_tokens": int(cfg["generation"]["max_tokens"]),
        "split": split,
        "retrieval": {
            "top_k": int(retrieval_cfg["top_k"]),
            "answer_k": int(retrieval_cfg["answer_k"]),
            "rrf_depth": int(retrieval_cfg["rrf_depth"]),
            "rrf_c": float(retrieval_cfg["rrf_c"]),
            "bm25": bm25_index.passport(),
        },
        "data": {
            "manifest_key": data.MANIFEST_KEY,
            "archive_sha256": data.ARCHIVE_SHA256,
            "n_claims": len(claims),
            "n_q_plus": sum(1 for claim in claims if claim["evidence"]),
            "claim_ids": [claim["id"] for claim in claims],
        },
    }
    reference = _lr2_reference(cfg)
    if reference is not None:
        fields["lr2_dense_reference"] = reference
    if reranker is not None:
        fields["reranker"] = reranker.passport()
    metadata.add(run_dir, fields)

    console.info(
        f"сервер: {info['server']['name']} {info['server']['version']}, "
        f"модель {info['model']['repo']}, контекст "
        f"{info['server']['max_model_len']}"
    )
    console.info(
        f"данные: {split}={len(claims)}, Q+={fields['data']['n_q_plus']}"
    )
    console.info(
        f"BM25: N={bm25_index.n_docs}, "
        f"средняя длина={bm25_index.avg_length:.2f} токена"
    )
    if reranker is not None:
        console.info(
            f"reranker: {reranker.model_name}, max_length={reranker.max_length}"
        )
    return Session(
        endpoint=endpoint,
        client=client.make_client(endpoint),
        index_record=index_record,
        query_model=index.load_model_for_search(index_record),
        tokenizer=tokenizer,
        corpus=corpus,
        claims=claims,
        gold=gold,
        budget=int(cfg["budget_tokens"]),
        max_tokens=int(cfg["generation"]["max_tokens"]),
        answer_k=int(cfg["retrieval"]["answer_k"]),
        split=split,
        bm25_index=bm25_index,
        reranker=reranker,
    )


def _timed(function, *args, **kwargs):
    started = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - started


def dense_claim(session: Session, claim: dict, *, top_k: int) -> tuple:
    """Dense top-k и время кодирования запроса вместе с поиском."""
    return _timed(
        dense.dense_search,
        session.index_record,
        session.query_model,
        claim["claim"],
        top_k=top_k,
    )


def bm25_claim(session: Session, claim: dict, *, top_k: int) -> tuple:
    """BM25 top-k и время поиска."""
    return _timed(
        session.bm25_index.search,
        claim["claim"],
        top_k=top_k,
    )


def rrf_claim(
    session: Session,
    dense_ranking: list[dict],
    bm25_ranking: list[dict],
    *,
    depth: int,
    c: float,
    top_k: int,
) -> tuple:
    """RRF top-k и время объединения."""
    return _timed(
        fusion.reciprocal_rank_fusion,
        dense_ranking,
        bm25_ranking,
        depth=depth,
        c=c,
        top_k=top_k,
    )


def rerank_claim(
    session: Session,
    claim: dict,
    ranking: list[dict],
    *,
    top_k: int,
) -> tuple:
    """Cross-encoder rerank и время переранжирования."""
    if session.reranker is None:
        raise RuntimeError("Reranker не загружен")
    return _timed(
        session.reranker.rerank,
        claim["claim"],
        ranking,
        session.corpus,
        top_k=top_k,
    )


def retrieve_all(
    session: Session,
    claim: dict,
    *,
    top_k: int,
    depth: int,
    c: float,
    need_rerank: bool,
) -> dict:
    """Считает Dense, BM25, RRF и при необходимости RRF+CE."""
    dense_ranking, dense_s = dense_claim(session, claim, top_k=top_k)
    bm25_ranking, bm25_s = bm25_claim(session, claim, top_k=top_k)
    rrf_ranking, rrf_s = rrf_claim(
        session,
        dense_ranking,
        bm25_ranking,
        depth=depth,
        c=c,
        top_k=top_k,
    )
    record = {
        "claim_id": claim["id"],
        "claim": claim["claim"],
        "split": session.split,
        "dense": dense_ranking,
        "bm25": bm25_ranking,
        "rrf": rrf_ranking,
        "timings_s": {
            "dense": dense_s,
            "bm25": bm25_s,
            "rrf": rrf_s,
        },
    }
    if need_rerank:
        reranked, rerank_s = rerank_claim(
            session,
            claim,
            rrf_ranking,
            top_k=top_k,
        )
        record["rrf_ce"] = reranked
        record["timings_s"]["rerank"] = rerank_s
    return record


def answer(
    session: Session,
    claim: dict,
    variant: str,
    ranking: list[dict],
    *,
    search_s: float,
) -> dict:
    """Один вызов RAG на готовом ранжировании."""
    passed_ranking = ranking[: int(session.answer_k)]
    ctx = context.from_ranked(
        "rag",
        claim,
        passed_ranking,
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
    if completion["status"] == "ok":
        parsed_result = response.parse_response(
            completion["raw_response"],
            ctx.passed,
            "rag",
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
        "split": session.split,
        "variant": variant,
        "mode": "rag",
        "k": int(session.answer_k),
        "retrieved": ranking,
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


def generate(
    session: Session,
    claim: dict,
    variant: str,
    ranking: list[dict],
    results_path: Path,
    *,
    search_s: float = 0.0,
) -> dict:
    """Считает ответ и сразу дописывает его в results.jsonl."""
    record = answer(
        session,
        claim,
        variant,
        ranking,
        search_s=search_s,
    )
    files.append_jsonl(results_path, record)
    return record


def _train_claim() -> dict:
    return next(claim for claim in data.load_claims_train() if claim["claim"])


def _retrieval_config(cfg: dict) -> tuple[int, int, float]:
    retrieval_cfg = cfg["retrieval"]
    return (
        int(retrieval_cfg["top_k"]),
        int(retrieval_cfg["rrf_depth"]),
        float(retrieval_cfg["rrf_c"]),
    )


def warmup_retrieval(
    session: Session,
    cfg: dict,
    *,
    need_rerank: bool,
) -> dict:
    """Прогрев поиска на train вне статистики времени."""
    top_k, depth, c = _retrieval_config(cfg)
    record = retrieve_all(
        session,
        _train_claim(),
        top_k=top_k,
        depth=depth,
        c=c,
        need_rerank=need_rerank,
    )
    record["warmup"] = True
    return record


def warmup_generation(
    session: Session,
    cfg: dict,
    run_dir: Path,
    *,
    need_rerank: bool,
) -> dict:
    """Прогрев модели на train; результат не входит в results.jsonl."""
    top_k, depth, c = _retrieval_config(cfg)
    claim = _train_claim()
    retrieval = retrieve_all(
        session,
        claim,
        top_k=top_k,
        depth=depth,
        c=c,
        need_rerank=need_rerank,
    )
    ranking = retrieval["rrf_ce"] if need_rerank else retrieval["rrf"]
    record = answer(
        session,
        claim,
        "warmup",
        ranking,
        search_s=0.0,
    )
    record["retrieval"] = retrieval
    record["warmup"] = True
    files.write_json(run_dir / "warmup.json", record)
    console.info(
        f"прогрев: prompt={record['prompt_tokens_expected']} ток., "
        f"ответ={record['parse_status']}"
    )
    return record
