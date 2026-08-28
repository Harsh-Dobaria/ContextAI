import time
from typing import Any, cast
from concurrent.futures import ThreadPoolExecutor

from langsmith import traceable

from app.database.database import SessionLocal
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.services.embedding_service import generate_embedding
from app.services.vector_service import vector_store
from app.services.bm25_service import bm25_service
from app.services.reranker_service import reranker_service

_chunk_map: dict[int, dict[str, Any]] = {}
_preloaded: bool = False

def preload_chunks() -> None:
    global _preloaded
    if _preloaded:
        return
        
    db = SessionLocal()
    try:
        start_time = time.perf_counter()
        results = (
            db.query(DocumentChunk, Document.workspace_id)
            .join(Document, DocumentChunk.document_id == Document.id)
            .all()
        )
        
        for chunk, workspace_id in results:
            _chunk_map[cast(int, chunk.id)] = {
                "chunk": chunk,
                "workspace_id": workspace_id
            }
            
        _preloaded = True
        latency = time.perf_counter() - start_time
        print(f"Preloaded {len(_chunk_map)} chunks into memory in {latency:.4f}s")
    finally:
        db.close()


@traceable(name="reciprocal_rank_fusion", run_type="chain")
def weighted_reciprocal_rank_fusion(
    result_lists: list[tuple[list[int], float]],
    k: int = 10
) -> list[int]:

    scores: dict[int, float] = {}

    for results, weight in result_lists:

        for rank, chunk_id in enumerate(
            results,
            start=1
        ):

            scores[chunk_id] = (
                scores.get(chunk_id, 0.0)
                + weight / (k + rank)
            )

    return [
        chunk_id
        for chunk_id, _ in sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True
        )
    ]


@traceable(name="retrieve_documents", run_type="retriever")
def retrieve_documents(
    question: str,
    workspace_id: int,
    retrieval_count: int = 3,
    retrieval_strategy: str = "semantic",
    retrieval_mode: str = "hybrid",
    query_embedding: list[float] | None = None
) -> dict[str, Any]:

    latencies: dict[str, float] = {}

    # ---------------------------------
    # Generate query embedding
    # ---------------------------------
    t_embed = time.perf_counter()
    if query_embedding is None:
        query_embedding = generate_embedding(question)
    latencies["embedding"] = time.perf_counter() - t_embed


    # ---------------------------------
    # Concurrent FAISS and BM25 retrieval
    # ---------------------------------
    def run_faiss() -> tuple[list[tuple[int, float]], float]:
        t_f = time.perf_counter()
        res = vector_store.search(query_embedding, top_k=50)
        return res, time.perf_counter() - t_f

    def run_bm25() -> tuple[list[tuple[int, float]], float]:
        t_b = time.perf_counter()
        res = bm25_service.search(question, top_k=50)
        return res, time.perf_counter() - t_b

    with ThreadPoolExecutor(max_workers=2) as executor:
        faiss_future = executor.submit(run_faiss)
        bm25_future = executor.submit(run_bm25)
        
        faiss_results, latencies["faiss"] = faiss_future.result()
        bm25_results, latencies["bm25"] = bm25_future.result()


    faiss_ids = [
        chunk_id
        for chunk_id, _ in faiss_results
    ]

    bm25_ids = [
        chunk_id
        for chunk_id, _ in bm25_results
    ]


    # ---------------------------------
    # Adaptive retrieval weights
    # ---------------------------------

    if retrieval_strategy == "keyword":

        faiss_weight = 0.5
        bm25_weight = 1.5

    elif retrieval_strategy == "broad":

        faiss_weight = 1.0
        bm25_weight = 1.0

    else:

        faiss_weight = 1.5
        bm25_weight = 0.5


    # ---------------------------------
    # Retrieval mode & RRF
    # ---------------------------------
    t_rrf = time.perf_counter()

    if retrieval_mode == "faiss":

        candidate_ids = faiss_ids[
            :retrieval_count
        ]

    elif retrieval_mode == "bm25":

        candidate_ids = bm25_ids[
            :retrieval_count
        ]

    elif retrieval_mode == "hybrid_rerank":

        hybrid_ids = (
            weighted_reciprocal_rank_fusion(
                [
                    (
                        faiss_ids,
                        faiss_weight
                    ),
                    (
                        bm25_ids,
                        bm25_weight
                    )
                ]
            )
        )
        
        t_rerank = time.perf_counter()
        
        if not _preloaded:
            preload_chunks()
            
        chunks_to_rerank = []
        for chunk_id in hybrid_ids[:50]:
            chunk_data = _chunk_map.get(chunk_id)
            if chunk_data and chunk_data["workspace_id"] == workspace_id:
                chunks_to_rerank.append((chunk_id, chunk_data["chunk"].content))
                
        candidate_ids = reranker_service.rerank(
            question, 
            chunks_to_rerank, 
            top_k=retrieval_count
        )
        
        latencies["rerank"] = time.perf_counter() - t_rerank

    else:

        hybrid_ids = (
            weighted_reciprocal_rank_fusion(
                [
                    (
                        faiss_ids,
                        faiss_weight
                    ),
                    (
                        bm25_ids,
                        bm25_weight
                    )
                ]
            )
        )

        candidate_ids = hybrid_ids[
            :retrieval_count
        ]
        
    latencies["rrf"] = time.perf_counter() - t_rrf


    # ---------------------------------
    # Memory chunk lookup
    # ---------------------------------
    t_lookup = time.perf_counter()
    
    if not _preloaded:
        preload_chunks()

    matched_chunks = []
    for chunk_id in candidate_ids:
        chunk_data = _chunk_map.get(chunk_id)
        if chunk_data and chunk_data["workspace_id"] == workspace_id:
            matched_chunks.append(chunk_data["chunk"])
            
    latencies["lookup"] = time.perf_counter() - t_lookup

    print(
        "\n========== RETRIEVAL DEBUG =========="
    )

    print(
        "Workspace ID:",
        workspace_id
    )

    print(
        "FAISS result count:",
        len(faiss_results)
    )

    print(
        "FAISS IDs:",
        faiss_ids[:10]
    )

    print(
        "BM25 result count:",
        len(bm25_results)
    )

    print(
        "BM25 IDs:",
        bm25_ids[:10]
    )

    print(
        "Candidate IDs:",
        candidate_ids
    )
    
    print(
        "Matched chunks:",
        len(matched_chunks)
    )

    print(
        "Matched chunk IDs:",
        [
            cast(int, chunk.id)
            for chunk in matched_chunks
        ]
    )
    print(
        "====================================\n"
    )

    context_parts: list[str] = []
    final_chunk_ids: list[int] = []

    # ---------------------------------
    # Preserve retrieval order
    # ---------------------------------
    
    matched_map = {
        cast(int, chunk.id): chunk
        for chunk in matched_chunks
    }

    for chunk_id in candidate_ids:
        chunk = matched_map.get(chunk_id)
        if chunk:
            final_chunk_ids.append(chunk_id)
            context_parts.append(cast(str, chunk.content))

    context = "\n\n".join(context_parts)

    # ---------------------------------
    # FAISS distances
    # ---------------------------------

    faiss_distance_map = {
        chunk_id: distance
        for chunk_id, distance
        in faiss_results
    }

    retrieved_distances = [
        faiss_distance_map[chunk_id]
        for chunk_id in final_chunk_ids
        if chunk_id in faiss_distance_map
    ]

    # ---------------------------------
    # Final response
    # ---------------------------------

    return {
        "context": context,

        "chunk_ids": final_chunk_ids,

        "retrieved_count": len(
            final_chunk_ids
        ),

        "retrieval_strategy":
            retrieval_strategy,

        "retrieval_mode":
            retrieval_mode,

        "faiss_distances":
            retrieved_distances,

        "faiss_ids":
            faiss_ids[:10],

        "bm25_ids":
            bm25_ids[:10],
            
        "latencies": latencies
    }