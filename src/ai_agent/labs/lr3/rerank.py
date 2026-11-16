"""Переранжирование 20 кандидатов RRF cross-encoder-моделью."""

from collections.abc import Callable, Sequence

from ai_agent.labs.lr2 import data

from .ranking import doc_key, item_id


def rerank_with_scores(
    claim: str,
    ranking: Sequence[dict],
    corpus: dict[str, dict],
    score_fn: Callable[[str, list[str]], Sequence[float]],
    *,
    top_k: int = 20,
) -> list[dict]:
    """Переставляет кандидатов по score_fn, не меняя множество документов."""
    if not ranking:
        return []
    documents = [data.document_text(corpus[item_id(item)]) for item in ranking]
    scores = score_fn(claim, documents)
    if len(scores) != len(ranking):
        raise ValueError("Число оценок cross-encoder не совпало с числом пар")

    rescored = []
    for item, score in zip(ranking, scores, strict=True):
        rescored.append(
            {
                **item,
                "rrf_score": float(item["score"]),
                "score": float(score),
                "rerank_score": float(score),
            }
        )

    ordered = sorted(
        rescored,
        key=lambda item: (-item["score"], doc_key(item["doc_id"])),
    )[:top_k]
    return [
        {**item, "rank": rank} for rank, item in enumerate(ordered, start=1)
    ]


class CrossEncoderReranker:
    """ms-marco cross-encoder через Transformers, только CPU."""

    def __init__(
        self,
        model_name: str,
        *,
        max_length: int = 512,
        batch_size: int = 20,
        device: str = "cpu",
    ) -> None:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self.model_name = model_name
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).eval()
        self.model.to(device)
        self._revision = getattr(self.model.config, "_commit_hash", None)
        if self._revision is None:
            try:
                from huggingface_hub import HfApi

                self._revision = HfApi().model_info(model_name).sha
            except Exception:
                self._revision = None

    @property
    def revision(self) -> str | None:
        """Ревизия весов модели для метаданных."""
        return self._revision

    def score(self, claim: str, documents: list[str]) -> list[float]:
        """Оценки пар «утверждение — документ»; обрезается только документ."""
        import torch

        result: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(documents), self.batch_size):
                batch_documents = documents[start : start + self.batch_size]
                batch = self.tokenizer(
                    [claim] * len(batch_documents),
                    batch_documents,
                    truncation="only_second",
                    max_length=self.max_length,
                    padding=True,
                    return_tensors="pt",
                )
                batch = {
                    name: value.to(self.device) for name, value in batch.items()
                }
                logits = self.model(**batch).logits.reshape(-1)
                result.extend(float(value) for value in logits.cpu())
        return result

    def rerank(
        self,
        claim: str,
        ranking: Sequence[dict],
        corpus: dict[str, dict],
        *,
        top_k: int = 20,
    ) -> list[dict]:
        """Возвращает новый порядок тех же кандидатов."""
        return rerank_with_scores(
            claim,
            ranking,
            corpus,
            self.score,
            top_k=top_k,
        )

    def passport(self) -> dict:
        """Параметры модели для metadata.json и отчёта."""
        return {
            "model": self.model_name,
            "revision": self.revision,
            "max_length": self.max_length,
            "truncation": "only_second",
            "batch_size": self.batch_size,
            "device": self.device,
            "new_documents": False,
        }
