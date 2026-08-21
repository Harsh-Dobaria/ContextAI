import json
import os

import faiss
import numpy as np

from app.models.document_chunk import DocumentChunk


FAISS_DIR = "faiss_data"

INDEX_PATH = os.path.join(
    FAISS_DIR,
    "index.faiss"
)

CHUNK_IDS_PATH = os.path.join(
    FAISS_DIR,
    "chunk_ids.json"
)


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
    # Load FAISS index
    # ---------------------------------

    def _load(self) -> None:

        if (
            os.path.exists(INDEX_PATH)
            and os.path.exists(CHUNK_IDS_PATH)
        ):

            try:

                self.index = faiss.read_index(
                    INDEX_PATH
                )

                with open(
                    CHUNK_IDS_PATH,
                    "r",
                    encoding="utf-8"
                ) as file:

                    self.chunk_ids = [
                        int(chunk_id)
                        for chunk_id
                        in json.load(file)
                    ]

                # Validate index mapping
                if (
                    self.index.ntotal
                    != len(self.chunk_ids)
                ):

                    print(
                        "WARNING: FAISS index and "
                        "chunk ID mapping do not match"
                    )

                    print(
                        f"Vectors: "
                        f"{self.index.ntotal}"
                    )

                    print(
                        f"Chunk IDs: "
                        f"{len(self.chunk_ids)}"
                    )

                else:

                    print(
                        f"Loaded FAISS index: "
                        f"{self.index.ntotal} vectors"
                    )

                return

            except Exception as error:

                print(
                    f"Failed to load "
                    f"FAISS index: {error}"
                )


        self.index = faiss.IndexFlatL2(
            self.dimension
        )

        self.chunk_ids = []

        print(
            "Created new empty FAISS index"
        )


    # ---------------------------------
    # Save FAISS index
    # ---------------------------------

    def _save(self) -> None:

        faiss.write_index(
            self.index,
            INDEX_PATH
        )

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

        # Validate embedding dimension
        if (
            vector.shape[1]
            != self.dimension
        ):

            raise ValueError(
                f"Embedding dimension mismatch. "
                f"Expected {self.dimension}, "
                f"got {vector.shape[1]}"
            )

        self.index.add(
            vector
        )

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

        if self.index.ntotal == 0:

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
        if (
            vector.shape[1]
            != self.dimension
        ):

            raise ValueError(
                f"Query embedding dimension mismatch. "
                f"Expected {self.dimension}, "
                f"got {vector.shape[1]}"
            )


        actual_top_k = min(
            top_k,
            self.index.ntotal
        )


        distances, indices = self.index.search(
            vector,
            actual_top_k
        )


        results: list[
            tuple[int, float]
        ] = []


        for distance, index in zip(
            distances[0],
            indices[0]
        ):

            index = int(index)

            # FAISS returns -1 for invalid results
            if index == -1:

                continue


            # Prevent mapping errors
            if (
                index < 0
                or index >= len(
                    self.chunk_ids
                )
            ):

                print(
                    f"WARNING: Invalid FAISS index "
                    f"position: {index}"
                )

                continue


            chunk_id = self.chunk_ids[
                index
            ]


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


    # ---------------------------------
    # Rebuild from database
    # ---------------------------------

    def rebuild_from_database(
        self,
        db
    ) -> int:

        chunks = (
            db.query(
                DocumentChunk
            )
            .filter(
                DocumentChunk.embedding.isnot(
                    None
                )
            )
            .order_by(
                DocumentChunk.id
            )
            .all()
        )


        self.index = faiss.IndexFlatL2(
            self.dimension
        )

        self.chunk_ids = []


        for chunk in chunks:

            if not chunk.embedding:

                continue


            embedding = json.loads(
                chunk.embedding
            )


            vector = np.array(
                [embedding],
                dtype="float32"
            )


            # Skip invalid embeddings
            if (
                vector.shape[1]
                != self.dimension
            ):

                print(
                    f"Skipping chunk "
                    f"{chunk.id}: invalid "
                    f"embedding dimension "
                    f"{vector.shape[1]}"
                )

                continue


            self.index.add(
                vector
            )


            self.chunk_ids.append(
                int(chunk.id)
            )


        self._save()


        print(
            f"Rebuilt FAISS index: "
            f"{self.index.ntotal} vectors"
        )


        print(
            f"FAISS chunk mapping: "
            f"{len(self.chunk_ids)} IDs"
        )


        return self.index.ntotal


vector_store = FAISSVectorStore(
    dimension=768
)