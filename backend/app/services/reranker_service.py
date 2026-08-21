from typing import cast

from sentence_transformers import CrossEncoder


class RerankerService:
    def __init__(self):
        self.model: CrossEncoder | None = None

    def _load_model(self) -> CrossEncoder:
        if self.model is None:
            print("Loading reranker model...")

            self.model = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-2-v2",
                device="cpu"
            )

            print("Reranker model loaded.")

        return cast(CrossEncoder, self.model)

    def rerank(
        self,
        question: str,
        chunks: list[tuple[int, str]],
        top_k: int = 3
    ) -> list[int]:
        if not chunks:
            return []

        model = self._load_model()

        pairs = [
            (question, content)
            for _, content in chunks
        ]

        scores = model.predict(pairs)

        ranked = sorted(
            zip(chunks, scores),
            key=lambda item: item[1],
            reverse=True
        )

        return [
            chunk_id
            for (chunk_id, _), _ in ranked[:top_k]
        ]


reranker_service = RerankerService()