from typing import cast

from langsmith import traceable
from sentence_transformers import CrossEncoder


@traceable(name="cross_encoder_rerank", run_type="chain")
def _execute_rerank(
    model: CrossEncoder,
    question: str,
    chunks: list[tuple[int, str]],
    top_k: int = 3
) -> list[int]:
    if not chunks:
        return []

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
        return _execute_rerank(
            model=model,
            question=question,
            chunks=chunks,
            top_k=top_k
        )


reranker_service = RerankerService()