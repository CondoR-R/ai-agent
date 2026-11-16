"""BM25 с токенизацией и формулой из ЛР3."""

import math
import re
from collections import Counter, defaultdict

from ai_agent.labs.lr2 import data

from .ranking import doc_key

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
TOKENIZER = 're.findall(r"[a-z0-9]+", text.lower())'


def tokenize(text: str) -> list[str]:
    """Нижний регистр и последовательности латинских букв и цифр."""
    return TOKEN_PATTERN.findall(str(text).lower())


class BM25Index:
    """Инвертированный индекс BM25 с параметрами Lucene."""

    def __init__(
        self,
        documents: dict[str, str],
        *,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        if not documents:
            raise ValueError("BM25 требует непустой корпус")
        self.k1 = float(k1)
        self.b = float(b)
        self.doc_ids = sorted(documents, key=doc_key)
        self.doc_lengths: dict[str, int] = {}
        self.postings: dict[str, dict[str, int]] = defaultdict(dict)

        for doc_id in self.doc_ids:
            tokens = tokenize(documents[doc_id])
            self.doc_lengths[doc_id] = len(tokens)
            for term, frequency in Counter(tokens).items():
                self.postings[term][doc_id] = frequency

        self.n_docs = len(self.doc_ids)
        self.avg_length = sum(self.doc_lengths.values()) / self.n_docs

    @classmethod
    def from_corpus(
        cls,
        corpus: dict[str, dict],
        *,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> "BM25Index":
        """Индекс по заголовку и аннотации каждого документа."""
        documents = {
            str(doc_id): data.document_text(document)
            for doc_id, document in corpus.items()
        }
        return cls(documents, k1=k1, b=b)

    def _idf(self, term: str) -> float:
        n_term = len(self.postings[term])
        return math.log(1.0 + (self.n_docs - n_term + 0.5) / (n_term + 0.5))

    def search(self, query: str, *, top_k: int = 20) -> list[dict]:
        """До top_k документов с положительной оценкой."""
        scores: dict[str, float] = defaultdict(float)
        for term in set(tokenize(query)):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self._idf(term)
            for doc_id, frequency in postings.items():
                length_norm = self.doc_lengths[doc_id] / self.avg_length
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * length_norm
                )
                scores[doc_id] += (
                    idf * frequency * (self.k1 + 1.0) / denominator
                )

        ordered = sorted(
            (doc_id for doc_id, score in scores.items() if score > 0.0),
            key=lambda doc_id: (-scores[doc_id], doc_key(doc_id)),
        )[:top_k]
        return [
            {
                "doc_id": doc_id,
                "score": float(scores[doc_id]),
                "rank": rank,
            }
            for rank, doc_id in enumerate(ordered, start=1)
        ]

    def passport(self) -> dict:
        """Параметры метода для metadata.json и отчёта."""
        return {
            "implementation": "own",
            "tokenizer": TOKENIZER,
            "k1": self.k1,
            "b": self.b,
            "idf": "ln(1 + (N - n_t + 0.5) / (n_t + 0.5))",
            "n_docs": self.n_docs,
            "avg_document_length": self.avg_length,
            "term_count": len(self.postings),
        }
