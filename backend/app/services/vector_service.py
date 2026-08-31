import json
import os
from typing import Any

import numpy as np

from app.models.document_chunk import DocumentChunk

try:
    import faiss
    FAISS_AVAILABLE = True
except (ImportError, OSError):
    faiss = None  # type: ignore
    FAISS_AVAILABLE = False


class NumpyVectorIndex:
    """
    Pure-NumPy drop-in replacement for FAISS IndexFlatL2.
    Implements L2 (Euclidean) distance similarity search.
    """
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors: np.ndarray = np.empty((0, dimension), dtype="float32")

    @property
    def ntotal(self) -> int:
        return self.vectors.shape[0]

    def add(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype="float32")
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if self.vectors.shape[0] == 0:
            self.vectors = x
        else:
            self.vectors = np.vstack([self.vectors, x])

    def search(self, query_vector: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.ntotal == 0 or top_k <= 0:
            return np.empty((1, 0), dtype="float32"), np.empty((1, 0), dtype=int)

        query_vector = np.asarray(query_vector, dtype="float32")
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        diff = self.vectors - query_vector
        squared_distances = np.sum(diff * diff, axis=1)

        k = min(top_k, self.ntotal)
        indices = np.argsort(squared_distances)[:k]
        distances = squared_distances[indices]

        return distances.reshape(1, -1), indices.reshape(1, -1)


FAISS_DIR = "faiss_data"

INDEX_PATH = os.path.join(
    FAISS_DIR,
    "index.faiss"
)

NUMPY_INDEX_PATH = os.path.join(
    FAISS_DIR,
    "index.npy"
)

CHUNK_IDS_PATH = os.path.join(
    FAISS_DIR,
    "chunk_ids.json"
)


def _create_empty_index(dimension: int) -> Any:
    if FAISS_AVAILABLE and faiss is not None:
        return faiss.IndexFlatL2(dimension)
    return NumpyVectorIndex(dimension)


def _execute_faiss_search(
    index: Any,
    dimension: int,
    chunk_ids: list[int],
    embedding: list[float],
    top_k: int = 5
) -> list[tuple[int, float]]:

    if index.ntotal == 0:
        print(
            "FAISS search skipped: "
            "index is empty"
        )
        return []

    vector = np.array(
        [embedding],
        dtype="float32"
    )

    # Validate query embedding dimension
    if vector.shape[1] != dimension:
        raise ValueError(
            f"Query embedding dimension mismatch. "
            f"Expected {dimension}, "
            f"got {vector.shape[1]}"
        )

    actual_top_k = min(
        top_k,
        index.ntotal
    )

    distances, indices = index.search(
        vector,
        actual_top_k
    )

    results: list[tuple[int, float]] = []

    for distance, idx in zip(
        distances[0],
        indices[0]
    ):
        pos = int(idx)

        # FAISS returns -1 for invalid results
        if pos == -1:
            continue

        # Prevent mapping errors
        if pos < 0 or pos >= len(chunk_ids):
            print(
                f"WARNING: Invalid FAISS index "
                f"position: {pos}"
            )
            continue

        chunk_id = chunk_ids[pos]

        results.append(
            (
                int(chunk_id),
                float(distance)
            )
        )

    print(
        f"FAISS returned "
        f"{len(results)} results"
    )

    return results


class FAISSVectorStore:

    def __init__(
        self,
        dimension: int = 768
    ):
        self.dimension = dimension
        self.chunk_ids: list[int] = []

        os.makedirs(
            FAISS_DIR,
            exist_ok=True
        )

        self._load()

    # ---------------------------------
    # Load FAISS / NumPy index
    # ---------------------------------

    def _load(self) -> None:

        if (
            os.path.exists(INDEX_PATH)
            and os.path.exists(CHUNK_IDS_PATH)
            and FAISS_AVAILABLE
            and faiss is not None
        ):
            try:
                self.index = faiss.read_index(INDEX_PATH)

                with open(
                    CHUNK_IDS_PATH,
                    "r",
                    encoding="utf-8"
                ) as file:
                    self.chunk_ids = [
                        int(chunk_id)
                        for chunk_id in json.load(file)
                    ]

                if self.index.ntotal != len(self.chunk_ids):
                    print(
                        "WARNING: FAISS index and "
                        "chunk ID mapping do not match"
                    )
                    print(f"Vectors: {self.index.ntotal}")
                    print(f"Chunk IDs: {len(self.chunk_ids)}")
                else:
                    print(
                        f"Loaded FAISS index: "
                        f"{self.index.ntotal} vectors"
                    )
                return
            except Exception as error:
                print(f"Failed to load FAISS index: {error}")

        if os.path.exists(NUMPY_INDEX_PATH) and os.path.exists(CHUNK_IDS_PATH):
            try:
                vectors = np.load(NUMPY_INDEX_PATH)
                idx = NumpyVectorIndex(self.dimension)
                idx.add(vectors)
                self.index = idx

                with open(
                    CHUNK_IDS_PATH,
                    "r",
                    encoding="utf-8"
                ) as file:
                    self.chunk_ids = [
                        int(chunk_id)
                        for chunk_id in json.load(file)
                    ]

                print(
                    f"Loaded NumPy vector index: "
                    f"{self.index.ntotal} vectors"
                )
                return
            except Exception as error:
                print(f"Failed to load NumPy vector index: {error}")

        self.index = _create_empty_index(self.dimension)
        self.chunk_ids = []

        print(
            "Created new empty vector index"
        )

    # ---------------------------------
    # Save FAISS / NumPy index
    # ---------------------------------

    def _save(self) -> None:

        if FAISS_AVAILABLE and faiss is not None and not isinstance(self.index, NumpyVectorIndex):
            try:
                faiss.write_index(
                    self.index,
                    INDEX_PATH
                )
            except Exception as error:
                print(f"Failed to write FAISS index: {error}")
        elif isinstance(self.index, NumpyVectorIndex):
            np.save(NUMPY_INDEX_PATH, self.index.vectors)

        with open(
            CHUNK_IDS_PATH,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                self.chunk_ids,
                file
            )

    # ---------------------------------
    # Add embedding
    # ---------------------------------

    def add_embedding(
        self,
        embedding: list[float],
        chunk_id: int
    ) -> None:

        vector = np.array(
            [embedding],
            dtype="float32"
        )

        if vector.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch. "
                f"Expected {self.dimension}, "
                f"got {vector.shape[1]}"
            )

        self.index.add(vector)

        self.chunk_ids.append(
            int(chunk_id)
        )

        self._save()

    # ---------------------------------
    # Search FAISS
    # ---------------------------------

    def search(
        self,
        embedding: list[float],
        top_k: int = 5
    ) -> list[tuple[int, float]]:
        return _execute_faiss_search(
            index=self.index,
            dimension=self.dimension,
            chunk_ids=self.chunk_ids,
            embedding=embedding,
            top_k=top_k
        )

    # ---------------------------------
    # Rebuild from database
    # ---------------------------------

    def rebuild_from_database(
        self,
        db
    ) -> int:

        chunks = (
            db.query(DocumentChunk)
            .filter(DocumentChunk.embedding.isnot(None))
            .order_by(DocumentChunk.id)
            .all()
        )

        self.index = _create_empty_index(self.dimension)
        self.chunk_ids = []

        for chunk in chunks:
            if not chunk.embedding:
                continue

            embedding = json.loads(chunk.embedding)

            vector = np.array(
                [embedding],
                dtype="float32"
            )

            if vector.shape[1] != self.dimension:
                print(
                    f"Skipping chunk {chunk.id}: invalid "
                    f"embedding dimension {vector.shape[1]}"
                )
                continue

            self.index.add(vector)
            self.chunk_ids.append(int(chunk.id))

        self._save()

        print(
            f"Rebuilt vector index: "
            f"{self.index.ntotal} vectors"
        )

        print(
            f"Vector chunk mapping: "
            f"{len(self.chunk_ids)} IDs"
        )

        return self.index.ntotal


vector_store = FAISSVectorStore(
    dimension=768
)