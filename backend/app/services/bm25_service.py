from typing import cast

from rank_bm25 import BM25Okapi
from sqlalchemy.orm import Session

from app.models.document_chunk import DocumentChunk


class BM25Service:
    def __init__(self):
        self.bm25 = None

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
            chunk.content.lower().split()
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

        if self.bm25 is None:

            return []

        query_tokens = (
            query.lower()
            .split()
        )

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True
        )[:top_k]

        return [
            (
                self.chunk_ids[index],
                float(scores[index])
            )
            for index in ranked_indices
            if scores[index] > 0
        ]


bm25_service = BM25Service()