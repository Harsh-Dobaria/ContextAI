import re
from typing import cast

from rank_bm25 import BM25Okapi
from sqlalchemy.orm import Session

from app.models.document_chunk import DocumentChunk


def tokenize_text(text: str) -> list[str]:
    """
    Tokenizes text for BM25 lexical search.
    Extracts alphanumeric tokens while preserving hyphens and underscores
    (e.g., 'api_v1', 'luna-1', 'bert-base', '768', 'ye-1a').
    Strips punctuation attached to words (e.g., 'space?' -> 'space', 'planet,' -> 'planet').
    """
    if not text:
        return []
    return re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())


def _execute_bm25_search(
    bm25: BM25Okapi | None,
    chunk_ids: list[int],
    query: str,
    top_k: int = 20
) -> list[tuple[int, float]]:
    if bm25 is None:
        return []

    query_tokens = tokenize_text(query)

    scores = bm25.get_scores(
        query_tokens
    )

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True
    )[:top_k]

    return [
        (
            chunk_ids[index],
            float(scores[index])
        )
        for index in ranked_indices
        if scores[index] > 0
    ]


class BM25Service:
    def __init__(self):
        self.bm25: BM25Okapi | None = None

        self.chunk_ids: list[int] = []

    def build(
        self,
        chunks: list[DocumentChunk]
    ) -> None:

        if not chunks:

            self.bm25 = None

            self.chunk_ids = []

            return

        tokenized_corpus = [
            tokenize_text(chunk.content)
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(
            tokenized_corpus
        )

        self.chunk_ids = [
            cast(int, chunk.id)
            for chunk in chunks
        ]

        print(
            f"Built BM25 index: "
            f"{len(self.chunk_ids)} chunks"
        )

    def build_from_database(
        self,
        db: Session
    ) -> int:

        chunks = (
            db.query(DocumentChunk)
            .order_by(
                DocumentChunk.id
            )
            .all()
        )

        self.build(chunks)

        return len(
            self.chunk_ids
        )

    def search(
        self,
        query: str,
        top_k: int = 20
    ) -> list[tuple[int, float]]:
        return _execute_bm25_search(
            bm25=self.bm25,
            chunk_ids=self.chunk_ids,
            query=query,
            top_k=top_k
        )


bm25_service = BM25Service()