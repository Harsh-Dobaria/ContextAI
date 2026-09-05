import json
import time
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np


# ---------------------------------
# Add backend directory to Python path
# ---------------------------------

BACKEND_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

sys.path.insert(
    0,
    str(BACKEND_DIR)
)


# ---------------------------------
# Imports from backend
# ---------------------------------

from app.database.database import SessionLocal

from app.services.agents.retrieval_agent import (
    retrieve_documents,
    preload_chunks
)

from app.services.bm25_service import (
    bm25_service
)
from app.services.reranker_service import reranker_service
from app.services.embedding_service import generate_embedding
from benchmarks.metrics import evaluate_retrieval_pipeline

# ---------------------------------
# Configuration
# ---------------------------------

BASE_DIR = Path(__file__).resolve().parent

QUESTIONS_FILE = (
    BASE_DIR / "benchmark_questions_labeled.json"
)

WORKSPACE_ID = 2

TOP_K = 5


# ---------------------------------
# Load questions
# ---------------------------------

def load_questions():

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ---------------------------------
# Validate questions ground truth
# ---------------------------------

def validate_ground_truth(questions):

    unlabeled = []
    for idx, q in enumerate(questions):
        rel = q.get("relevant_chunk_ids")
        if not rel or not isinstance(rel, list) or len(rel) == 0:
            if "relevant_chunk_ids" not in q:
                reason = "Missing 'relevant_chunk_ids' field"
            elif rel is None:
                reason = "'relevant_chunk_ids' is None"
            elif not isinstance(rel, list):
                reason = f"'relevant_chunk_ids' is not a list (got {type(rel).__name__})"
            else:
                reason = "'relevant_chunk_ids' is an empty list []"
            unlabeled.append((idx + 1, q, rel, reason))

    if unlabeled:
        print("\n" + "!" * 75)
        print(
            f"[GROUND TRUTH VALIDATION] Warning: Found {len(unlabeled)} unlabeled question(s):"
        )
        for num, q, rel, reason in unlabeled:
            qid = q.get("id", num)
            qtext = q.get("question", "")
            print(f"  * Question #{num} (ID: {qid}):")
            print(f"    - Text: \"{qtext}\"")
            print(f"    - Ground-truth chunk IDs: {rel!r}")
            print(f"    - Reason unlabeled: {reason}")
        print("!" * 75 + "\n")
    else:
        print(f"\n[GROUND TRUTH VALIDATION] Validation passed: All {len(questions)} questions have valid ground-truth labels.\n")


# ---------------------------------
# Build BM25 index
# ---------------------------------

def initialize_bm25():

    db = SessionLocal()

    try:

        chunk_count = (
            bm25_service.build_from_database(
                db
            )
        )

        print(
            f"Loaded BM25 index: "
            f"{chunk_count} chunks"
        )

    finally:

        db.close()


# ---------------------------------
# Calculate Retrieval Metrics
# ---------------------------------

def calculate_recall(
    results,
    questions,
    k=TOP_K
):
    """
    Evaluates Hit@K and true macro-averaged Recall@K.

    IMPLEMENTATION NOTE (Hit@K vs Recall@K):
    When each question in a dataset has exactly 1 relevant chunk, Hit@K and
    Recall@K are mathematically equivalent.
    In this benchmark dataset (benchmark_questions_labeled.json):
      - 18 questions have 1 relevant chunk (36%)
      - 32 questions have multiple relevant chunks (64% - between 2 and 4 chunks)
    Therefore:
      - Hit@K measures binary coverage: at least 1 relevant chunk in top-K (1 or 0).
      - Recall@K measures chunk coverage: len(rel & retrieved[:K]) / len(rel),
        macro-averaged across all valid questions.
    """
    eval_res = evaluate_retrieval_pipeline(results, questions, k=k)
    total_valid = eval_res["valid_questions"]
    hits = eval_res.get("hits_count", 0)
    unlabeled = eval_res["excluded_questions"]
    recall = (eval_res["recall_at_k"] / 100.0) if eval_res["recall_at_k"] is not None else None
    return total_valid, hits, unlabeled, recall


# ---------------------------------
# Run retrieval
# ---------------------------------

def run_retrieval(
    question,
    mode,
    query_embedding
):

    start_time = time.perf_counter()

    result = retrieve_documents(
        question=question,
        workspace_id=WORKSPACE_ID,
        retrieval_count=TOP_K,
        retrieval_mode=mode,
        query_embedding=query_embedding
    )

    latency = (
        time.perf_counter()
        - start_time
    )

    return result, latency


# ---------------------------------
# Main benchmark
# ---------------------------------

def main():

    # ---------------------------------
    # Initialize BM25 from database
    # ---------------------------------

    initialize_bm25()
    
    # ---------------------------------
    # Preload Chunk Map & Reranker Model
    # ---------------------------------

    preload_chunks()
    reranker_service._load_model()

    # ---------------------------------
    # Warm-up pass (ensures steady-state execution without cold-start artifacts)
    # ---------------------------------
    _warm_emb = generate_embedding("space exploration warmup query")
    for _mode in ["faiss", "bm25", "hybrid", "hybrid_rerank"]:
        retrieve_documents(
            question="space exploration warmup query",
            workspace_id=WORKSPACE_ID,
            retrieval_count=TOP_K,
            retrieval_mode=_mode,
            query_embedding=_warm_emb
        )

    # ---------------------------------
    # Load benchmark questions
    # ---------------------------------


    START = 0
    END = 50


    questions = load_questions()[START:END]
    validate_ground_truth(questions)

    faiss_results = []
    bm25_results = []
    hybrid_results = []
    rerank_results = []
    
    faiss_latencies = []
    bm25_latencies = []
    hybrid_latencies = []
    rerank_latencies = []
    
    granular_latencies = defaultdict(list)

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RAG BENCHMARK STARTED"
    )

    print(
        f"Questions: {len(questions)}"
    )

    print(
        f"Workspace ID: {WORKSPACE_ID}"
    )

    print(
        f"Top K: {TOP_K}"
    )

    print(
        "=" * 70
        + "\n"
    )


    # ---------------------------------
    # Run all questions
    # ---------------------------------

    for index, item in enumerate(
        questions,
        start=1
    ):

        question = item["question"]

        print(
            f"\nQuestion {index}: "
            f"{question}"
        )

        print(
            "-" * 70
        )
        
        # ---------------------------------
        # Generate Embedding Once
        # ---------------------------------
        t_start_embed = time.perf_counter()
        query_embedding = generate_embedding(question)
        embedding_time = time.perf_counter() - t_start_embed
        granular_latencies["embedding"].append(embedding_time)

        # ---------------------------------
        # FAISS ONLY
        # ---------------------------------

        faiss_result, faiss_latency = (
            run_retrieval(
                question,
                "faiss",
                query_embedding
            )
        )

        faiss_results.append(
            faiss_result
        )

        faiss_latencies.append(
            faiss_latency
        )
        
        for k, v in faiss_result.get("latencies", {}).items():
            if k != "embedding":
                granular_latencies[f"faiss_{k}"].append(v)


        print(
            "FAISS IDs:"
        )

        print(
            faiss_result["chunk_ids"]
        )


        print(
            f"FAISS Latency: "
            f"{faiss_latency:.4f}s"
        )


        # ---------------------------------
        # BM25 ONLY
        # ---------------------------------

        bm25_result, bm25_latency = (
            run_retrieval(
                question,
                "bm25",
                query_embedding
            )
        )

        bm25_results.append(
            bm25_result
        )

        bm25_latencies.append(
            bm25_latency
        )

        for k, v in bm25_result.get("latencies", {}).items():
            if k != "embedding":
                granular_latencies[f"bm25_{k}"].append(v)


        print(
            "\nBM25 IDs:"
        )

        print(
            bm25_result["chunk_ids"]
        )

        print(
            f"BM25 Latency: "
            f"{bm25_latency:.4f}s"
        )


        # ---------------------------------
        # HYBRID
        # ---------------------------------

        hybrid_result, hybrid_latency = (
            run_retrieval(
                question,
                "hybrid",
                query_embedding
            )
        )

        hybrid_results.append(
            hybrid_result
        )

        hybrid_latencies.append(
            hybrid_latency
        )
        
        for k, v in hybrid_result.get("latencies", {}).items():
            if k != "embedding":
                granular_latencies[f"hybrid_{k}"].append(v)


        print(
            "\nHybrid IDs:"
        )

        print(
            hybrid_result["chunk_ids"]
        )


        # ---------------------------------
        # BM25 DEBUG OUTPUT
        # ---------------------------------

        print(
            "\nBM25 Top 10 IDs:"
        )

        print(
            hybrid_result.get(
                "bm25_ids",
                []
            )[:10]
        )


        print(
            f"\nHybrid Latency: "
            f"{hybrid_latency:.4f}s"
        )


        # ---------------------------------
        # HYBRID + RERANKER
        # ---------------------------------
        
        rerank_result, rerank_latency = (
            run_retrieval(
                question,
                "hybrid_rerank",
                query_embedding
            )
        )
        
        rerank_results.append(rerank_result)
        rerank_latencies.append(rerank_latency)
        
        for k, v in rerank_result.get("latencies", {}).items():
            if k != "embedding":
                granular_latencies[f"rerank_{k}"].append(v)
                
        print("\nRerank IDs:")
        print(rerank_result["chunk_ids"])
        print(f"\nRerank Latency: {rerank_latency:.4f}s")



    # ---------------------------------
    # Evaluate Retrieval Metrics (Hit@K, Recall@K, MRR@K)
    # ---------------------------------

    faiss_eval = evaluate_retrieval_pipeline(faiss_results, questions, k=TOP_K)
    bm25_eval = evaluate_retrieval_pipeline(bm25_results, questions, k=TOP_K)
    hybrid_eval = evaluate_retrieval_pipeline(hybrid_results, questions, k=TOP_K)
    rerank_eval = evaluate_retrieval_pipeline(rerank_results, questions, k=TOP_K)

    # ---------------------------------
    # Calculate Latency Statistics (Retrieval Latency)
    # ---------------------------------

    faiss_avg_ms = float(np.mean(faiss_latencies)) * 1000.0
    bm25_avg_ms = float(np.mean(bm25_latencies)) * 1000.0
    hybrid_avg_ms = float(np.mean(hybrid_latencies)) * 1000.0
    rerank_avg_ms = float(np.mean(rerank_latencies)) * 1000.0

    faiss_p95_ms = float(np.percentile(faiss_latencies, 95)) * 1000.0
    bm25_p95_ms = float(np.percentile(bm25_latencies, 95)) * 1000.0
    hybrid_p95_ms = float(np.percentile(hybrid_latencies, 95)) * 1000.0
    rerank_p95_ms = float(np.percentile(rerank_latencies, 95)) * 1000.0

    # ---------------------------------
    # Final results output
    # ---------------------------------

    print("\n" + "=" * 70)
    print("FINAL BENCHMARK RESULTS")
    print("=" * 70)

    total_q = len(questions)
    valid_q = faiss_eval["valid_questions"]
    excluded_q = faiss_eval["excluded_questions"]

    print(f"\nTotal Questions: {total_q}")
    print(f"Valid Questions (evaluated): {valid_q}")
    if excluded_q > 0:
        print(f"Excluded Questions (no ground truth): {excluded_q}")

    configs = [
        ("FAISS", faiss_eval, faiss_avg_ms, faiss_p95_ms),
        ("BM25", bm25_eval, bm25_avg_ms, bm25_p95_ms),
        ("Hybrid RRF", hybrid_eval, hybrid_avg_ms, hybrid_p95_ms),
        ("Hybrid + Reranker", rerank_eval, rerank_avg_ms, rerank_p95_ms),
    ]

    for name, ev, avg_ms, p95_ms in configs:
        print(f"\n=== {name} ===")
        print(f"Valid Questions: {ev['valid_questions']}")
        if ev["hit_at_k"] is not None:
            print(f"Hit@{TOP_K}: {ev['hit_at_k']:.2f}%")
            print(f"Recall@{TOP_K}: {ev['recall_at_k']:.2f}%")
            print(f"MRR@{TOP_K}: {ev['mrr_at_k']:.4f}")
        else:
            print(f"Hit@{TOP_K}: N/A (no ground truth)")
            print(f"Recall@{TOP_K}: N/A (no ground truth)")
            print(f"MRR@{TOP_K}: N/A (no ground truth)")
        print(f"Avg Retrieval Latency: {avg_ms:.2f} ms")
        print(f"P95 Retrieval Latency: {p95_ms:.2f} ms")

    # ---------------------------------
    # Comparative Deltas (vs FAISS Baseline)
    # ---------------------------------

    print("\n" + "=" * 70)
    print("BENCHMARK COMPARISON & DELTAS (vs FAISS Baseline)")
    print("=" * 70)

    if faiss_eval["recall_at_k"] is not None:
        print("\nRecall@5 Comparison:")
        print(f"  FAISS Baseline:       {faiss_eval['recall_at_k']:.2f}%")
        if bm25_eval["recall_at_k"] is not None:
            print(f"  BM25 vs FAISS:        {bm25_eval['recall_at_k'] - faiss_eval['recall_at_k']:+.2f} percentage points")
        if hybrid_eval["recall_at_k"] is not None:
            print(f"  Hybrid RRF vs FAISS:  {hybrid_eval['recall_at_k'] - faiss_eval['recall_at_k']:+.2f} percentage points")
        if rerank_eval["recall_at_k"] is not None:
            print(f"  Rerank vs FAISS:      {rerank_eval['recall_at_k'] - faiss_eval['recall_at_k']:+.2f} percentage points")

    if faiss_eval["hit_at_k"] is not None:
        print("\nHit@5 Comparison:")
        print(f"  FAISS Baseline:       {faiss_eval['hit_at_k']:.2f}%")
        if bm25_eval["hit_at_k"] is not None:
            print(f"  BM25 vs FAISS:        {bm25_eval['hit_at_k'] - faiss_eval['hit_at_k']:+.2f} percentage points")
        if hybrid_eval["hit_at_k"] is not None:
            print(f"  Hybrid RRF vs FAISS:  {hybrid_eval['hit_at_k'] - faiss_eval['hit_at_k']:+.2f} percentage points")
        if rerank_eval["hit_at_k"] is not None:
            print(f"  Rerank vs FAISS:      {rerank_eval['hit_at_k'] - faiss_eval['hit_at_k']:+.2f} percentage points")

    if faiss_eval["mrr_at_k"] is not None:
        print("\nMRR@5 Comparison:")
        print(f"  FAISS Baseline:       {faiss_eval['mrr_at_k']:.4f}")
        if bm25_eval["mrr_at_k"] is not None:
            print(f"  BM25 vs FAISS:        {bm25_eval['mrr_at_k'] - faiss_eval['mrr_at_k']:+.4f}")
        if hybrid_eval["mrr_at_k"] is not None:
            print(f"  Hybrid RRF vs FAISS:  {hybrid_eval['mrr_at_k'] - faiss_eval['mrr_at_k']:+.4f}")
        if rerank_eval["mrr_at_k"] is not None:
            print(f"  Rerank vs FAISS:      {rerank_eval['mrr_at_k'] - faiss_eval['mrr_at_k']:+.4f}")

    print("\nRetrieval Latency Comparison (Average / P95):")
    print(f"  FAISS:                {faiss_avg_ms:.2f} ms / {faiss_p95_ms:.2f} ms")
    print(f"  BM25:                 {bm25_avg_ms:.2f} ms / {bm25_p95_ms:.2f} ms")
    print(f"  Hybrid RRF:           {hybrid_avg_ms:.2f} ms / {hybrid_p95_ms:.2f} ms")
    print(f"  Hybrid + Reranker:    {rerank_avg_ms:.2f} ms / {rerank_p95_ms:.2f} ms")

    # ---------------------------------
    # Granular Profiling Results
    # ---------------------------------

    print("\nGRANULAR PROFILING (Average / P95)")
    print("-" * 70)
    for key, vals in sorted(granular_latencies.items()):
        avg_ms_val = float(np.mean(vals)) * 1000.0
        p95_ms_val = float(np.percentile(vals, 95)) * 1000.0
        print(f"{key.ljust(20)}: {avg_ms_val:8.2f} ms / {p95_ms_val:8.2f} ms")

    print(
        "\n"
        + "=" * 70
    )



if __name__ == "__main__":

    main()