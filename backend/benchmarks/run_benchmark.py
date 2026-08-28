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
from app.services.embedding_service import generate_embedding
from benchmarks.experimental.hyde import generate_hyde_document

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
# Calculate Recall@K
# ---------------------------------

def calculate_recall(
    results,
    questions
):

    total_valid = 0
    hits = 0
    unlabeled = 0

    for result, question in zip(
        results,
        questions
    ):

        relevant_chunk_ids = set(
            question.get(
                "relevant_chunk_ids",
                []
            )
        )

        # Skip questions without ground truth

        if not relevant_chunk_ids:
            unlabeled += 1
            continue

        total_valid += 1

        retrieved_chunk_ids = set(
            result["chunk_ids"]
        )

        if (
            relevant_chunk_ids
            & retrieved_chunk_ids
        ):

            hits += 1

    if total_valid == 0:
        return 0, 0, unlabeled, None

    recall = hits / total_valid
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
    # Preload Chunk Map
    # ---------------------------------

    preload_chunks()

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
    hyde_results = []
    rerank_results = []
    
    faiss_latencies = []
    bm25_latencies = []
    hybrid_latencies = []
    hyde_latencies = []
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
        # HYBRID + HyDE
        # ---------------------------------
        
        t_start_hyde = time.perf_counter()
        hyde_doc = generate_hyde_document(question)
        hyde_time = time.perf_counter() - t_start_hyde
        
        t_start_hyde_embed = time.perf_counter()
        hyde_embedding = generate_embedding(hyde_doc)
        hyde_embed_time = time.perf_counter() - t_start_hyde_embed
        
        hyde_result, hyde_latency = (
            run_retrieval(
                question,
                "hybrid",
                hyde_embedding
            )
        )
        
        # Add the LLM time to total latency
        hyde_latency += hyde_time + hyde_embed_time
        
        hyde_results.append(hyde_result)
        hyde_latencies.append(hyde_latency)
        
        granular_latencies["hyde_llm"].append(hyde_time)
        granular_latencies["hyde_embedding"].append(hyde_embed_time)
        
        for k, v in hyde_result.get("latencies", {}).items():
            if k != "embedding":
                granular_latencies[f"hyde_{k}"].append(v)
                
        print("\nHyDE IDs:")
        print(hyde_result["chunk_ids"])
        print(f"\nHyDE Latency (incl LLM): {hyde_latency:.4f}s")


    # ---------------------------------
    # Calculate Recall
    # ---------------------------------

    faiss_valid, faiss_hits, faiss_unlabeled, faiss_recall = calculate_recall(
        faiss_results,
        questions
    )

    bm25_valid, bm25_hits, bm25_unlabeled, bm25_recall = calculate_recall(
        bm25_results,
        questions
    )

    hybrid_valid, hybrid_hits, hybrid_unlabeled, hybrid_recall = calculate_recall(
        hybrid_results,
        questions
    )
    
    hyde_valid, hyde_hits, hyde_unlabeled, hyde_recall = calculate_recall(
        hyde_results,
        questions
    )

    rerank_valid, rerank_hits, rerank_unlabeled, rerank_recall = calculate_recall(
        rerank_results,
        questions
    )


    # ---------------------------------
    # Calculate latency statistics
    # ---------------------------------

    faiss_average = float(np.mean(faiss_latencies))
    bm25_average = float(np.mean(bm25_latencies))
    hybrid_average = float(np.mean(hybrid_latencies))
    hyde_average = float(np.mean(hyde_latencies))
    rerank_average = float(np.mean(rerank_latencies))

    faiss_p95 = float(np.percentile(faiss_latencies, 95))
    bm25_p95 = float(np.percentile(bm25_latencies, 95))
    hybrid_p95 = float(np.percentile(hybrid_latencies, 95))
    hyde_p95 = float(np.percentile(hyde_latencies, 95))
    rerank_p95 = float(np.percentile(rerank_latencies, 95))


    # ---------------------------------
    # Final results
    # ---------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL BENCHMARK RESULTS"
    )

    print(
        "=" * 70
    )
    
    total_q = len(questions)
    
    print(
        f"\nTotal questions: {total_q}"
    )
    print(
        f"Labeled questions (valid ground truth): {faiss_valid}"
    )
    print(
        f"Unlabeled questions (skipped in recall): {faiss_unlabeled}"
    )


    # ---------------------------------
    # FAISS RESULTS
    # ---------------------------------

    print(
        "\nFAISS ONLY"
    )

    if faiss_recall is None:
        print(
            "Recall@5: "
            "Ground truth not added yet"
        )
    else:
        print(
            f"Hits: {faiss_hits} / {faiss_valid}"
        )
        print(
            f"Recall@5: "
            f"{faiss_recall * 100:.2f}%"
        )

    print(
        f"Average Latency: "
        f"{faiss_average:.4f}s"
    )

    print(
        f"P95 Latency: "
        f"{faiss_p95:.4f}s"
    )


    # ---------------------------------
    # BM25 RESULTS
    # ---------------------------------

    print(
        "\nBM25 ONLY"
    )

    if bm25_recall is None:
        print(
            "Recall@5: "
            "Ground truth not added yet"
        )
    else:
        print(
            f"Hits: {bm25_hits} / {bm25_valid}"
        )
        print(
            f"Recall@5: "
            f"{bm25_recall * 100:.2f}%"
        )

    print(
        f"Average Latency: "
        f"{bm25_average:.4f}s"
    )

    print(
        f"P95 Latency: "
        f"{bm25_p95:.4f}s"
    )


    # ---------------------------------
    # HYBRID RESULTS
    # ---------------------------------

    print(
        "\nHYBRID "
        "(FAISS + BM25 + RRF)"
    )

    if hybrid_recall is None:
        print(
            "Recall@5: "
            "Ground truth not added yet"
        )
    else:
        print(
            f"Hits: {hybrid_hits} / {hybrid_valid}"
        )
        print(
            f"Recall@5: "
            f"{hybrid_recall * 100:.2f}%"
        )

    print(
        f"Average Latency: "
        f"{hybrid_average:.4f}s"
    )

    print(
        f"P95 Latency: "
        f"{hybrid_p95:.4f}s"
    )


    # ---------------------------------
    # HYBRID + RERANKER RESULTS
    # ---------------------------------

    print("\nHYBRID + RERANKER")

    if rerank_recall is None:
        print("Recall@5: Ground truth not added yet")
    else:
        print(f"Hits: {rerank_hits} / {rerank_valid}")
        print(f"Recall@5: {rerank_recall * 100:.2f}%")

    print(f"Average Latency: {rerank_average:.4f}s")
    print(f"P95 Latency: {rerank_p95:.4f}s")


    # ---------------------------------
    # HYDE RESULTS
    # ---------------------------------

    print("\nHYBRID + HyDE")

    if hyde_recall is None:
        print("Recall@5: Ground truth not added yet")
    else:
        print(f"Hits: {hyde_hits} / {hyde_valid}")
        print(f"Recall@5: {hyde_recall * 100:.2f}%")

    print(f"Average Latency: {hyde_average:.4f}s")
    print(f"P95 Latency: {hyde_p95:.4f}s")


    # ---------------------------------
    # Recall improvement
    # ---------------------------------

    if (
        faiss_recall is not None
        and hybrid_recall is not None
    ):

        improvement = (
            hybrid_recall
            - faiss_recall
        ) * 100
        
        print(
            "\nRECALL IMPROVEMENT"
        )

        if bm25_recall is not None:
            bm25_improvement = (bm25_recall - faiss_recall) * 100
            print(f"BM25 vs FAISS: {bm25_improvement:+.2f} percentage points")

        print(f"Hybrid vs FAISS: {improvement:+.2f} percentage points")
        
        if hyde_recall is not None:
            hyde_improvement = (
                hyde_recall
                - faiss_recall
            ) * 100
            print(f"HyDE vs FAISS: {hyde_improvement:+.2f} percentage points")
            
        if rerank_recall is not None:
            rerank_improvement = (
                rerank_recall
                - faiss_recall
            ) * 100
            print(f"Rerank vs FAISS: {rerank_improvement:+.2f} percentage points")

    # ---------------------------------
    # Granular Profiling Results
    # ---------------------------------
    
    print("\nGRANULAR PROFILING (Average / P95)")
    print("-" * 70)
    for key, vals in sorted(granular_latencies.items()):
        avg = float(np.mean(vals))
        p95 = float(np.percentile(vals, 95))
        print(f"{key.ljust(20)}: {avg:.4f}s / {p95:.4f}s")


    print(
        "\n"
        + "=" * 70
    )


if __name__ == "__main__":

    main()