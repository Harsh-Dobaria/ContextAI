import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv
import numpy as np
from langsmith import Client, evaluate, traceable
from langsmith.evaluation import EvaluationResult, EvaluationResults
from langsmith.schemas import Example, Run

# ---------------------------------
# Setup backend environment and paths
# ---------------------------------
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

# Load .env configuration
load_dotenv(BACKEND_DIR / ".env")

from app.database.database import SessionLocal
from app.services.agents.retrieval_agent import (
    retrieve_documents,
    preload_chunks,
    _chunk_map
)
from app.services.bm25_service import bm25_service

# ---------------------------------
# Configuration & Constants
# ---------------------------------
BASE_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = BASE_DIR / "benchmark_questions_labeled.json"
DATASET_NAME = "ContextAI-Retrieval-Benchmark"
WORKSPACE_ID = 2
DEFAULT_TOP_K = 5

CONFIGURATIONS = {
    "faiss": {
        "label": "FAISS-Vector",
        "description": "Vector search with FAISS IndexFlatL2 using all-mpnet-base-v2 embeddings",
        "prefix": "ContextAI-FAISS"
    },
    "bm25": {
        "label": "BM25-Keyword",
        "description": "BM25 Okapi keyword search over document chunks",
        "prefix": "ContextAI-BM25"
    },
    "hybrid": {
        "label": "Hybrid-RRF",
        "description": "Hybrid retrieval combining FAISS and BM25 with Reciprocal Rank Fusion",
        "prefix": "ContextAI-Hybrid-RRF"
    },
    "hybrid_rerank": {
        "label": "Hybrid-Rerank",
        "description": "Hybrid retrieval (FAISS + BM25 + RRF) with Cross-Encoder (ms-marco-MiniLM) reranking",
        "prefix": "ContextAI-Hybrid-Rerank"
    }
}


# ---------------------------------
# Dataset Management
# ---------------------------------
def load_local_questions() -> list[dict[str, Any]]:
    """Loads benchmark questions from the local labeled JSON file."""
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_ground_truth_labels(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Inspects benchmark questions and outputs debug warnings for any questions
    missing valid ground-truth labels (relevant_chunk_ids).
    """
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
        print("\n" + "!" * 85)
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
        print("!" * 85 + "\n")
    else:
        print(f"\n[GROUND TRUTH VALIDATION] Validation passed: All {len(questions)} questions have valid ground-truth labels.\n")

    return [q for _, q, _, _ in unlabeled]


def ensure_benchmark_dataset(client: Client, sync: bool = False) -> str:
    """
    Ensures the LangSmith benchmark dataset exists and contains the 50 questions.
    If sync is True or dataset does not exist, updates/creates the examples.
    """
    questions = load_local_questions()
    validate_ground_truth_labels(questions)
    dataset_exists = client.has_dataset(dataset_name=DATASET_NAME)

    if dataset_exists and not sync:
        print(f"[LangSmith] Using existing dataset: '{DATASET_NAME}'")
        return DATASET_NAME

    if not dataset_exists:
        print(f"[LangSmith] Creating dataset: '{DATASET_NAME}'...")
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="ContextAI 50-Question Retrieval Benchmark Dataset with ground-truth chunk IDs"
        )
        dataset_id = dataset.id
    else:
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        dataset_id = dataset.id
        print(f"[LangSmith] Syncing dataset: '{DATASET_NAME}' (ID: {dataset_id})...")

    # Read existing examples if syncing to avoid duplicate rows
    existing_examples = list(client.list_examples(dataset_id=dataset_id))
    if existing_examples and sync:
        print(f"[LangSmith] Deleting {len(existing_examples)} existing examples for clean sync...")
        for ex in existing_examples:
            client.delete_example(example_id=ex.id)

    if not existing_examples or sync:
        print(f"[LangSmith] Uploading {len(questions)} benchmark questions as dataset examples...")
        inputs = [{"question": q["question"], "id": q["id"]} for q in questions]
        outputs = [
            {
                "relevant_chunk_ids": q.get("relevant_chunk_ids", []),
                "is_human_verified": q.get("is_human_verified", False)
            }
            for q in questions
        ]
        metadata = [{"candidate_chunk_ids": q.get("candidate_chunk_ids", [])} for q in questions]

        client.create_examples(
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
            dataset_id=dataset_id
        )
        print(f"[LangSmith] Successfully populated dataset '{DATASET_NAME}' with {len(questions)} examples.")

    return DATASET_NAME


# ---------------------------------
# Traced Retrieval Target Function Factory
# ---------------------------------
def create_retrieval_target(mode: str, workspace_id: int = WORKSPACE_ID, top_k: int = DEFAULT_TOP_K):
    """
    Creates a target function for LangSmith evaluation wrapping the existing retrieval pipeline.
    """
    @traceable(name=f"ContextAI_{mode}_retrieval", run_type="chain")
    def retrieval_target(inputs: dict[str, Any]) -> dict[str, Any]:
        question = inputs["question"]
        start_time = time.perf_counter()

        result = retrieve_documents(
            question=question,
            workspace_id=workspace_id,
            retrieval_count=top_k,
            retrieval_mode=mode
        )

        latency = time.perf_counter() - start_time

        # Build chunks preview for rich inspection in LangSmith UI
        chunks_preview = []
        for cid in result.get("chunk_ids", []):
            chunk_data = _chunk_map.get(cid)
            if chunk_data:
                content = cast(str, chunk_data["chunk"].content)
                preview = content[:200] + "..." if len(content) > 200 else content
                chunks_preview.append({
                    "chunk_id": cid,
                    "preview": preview
                })
            else:
                chunks_preview.append({"chunk_id": cid, "preview": ""})

        return {
            "question": question,
            "retrieval_mode": mode,
            "chunk_ids": result.get("chunk_ids", []),
            "retrieved_count": result.get("retrieved_count", 0),
            "retrieval_strategy": result.get("retrieval_strategy", "semantic"),
            "retrieved_chunks": chunks_preview,
            "latencies": result.get("latencies", {}),
            "latency": latency
        }

    return retrieval_target


# ---------------------------------
# LangSmith Evaluators
# ---------------------------------
def recall_at_5_evaluator(run: Run, example: Example | None = None) -> EvaluationResult:
    """
    Evaluates Recall@5 for each question:
    Recall@5 = 1 if at least one relevant ground-truth chunk appears in top 5 retrieved chunks.
    Recall@5 = 0 otherwise.
    If the question is unlabeled (no ground truth), score is None (excluded from recall).
    """
    relevant_ids = set(example.outputs.get("relevant_chunk_ids", [])) if (example and example.outputs) else set()

    if not relevant_ids:
        return EvaluationResult(
            key="recall_at_5",
            score=None,
            comment="Unlabeled ground truth - excluded from Recall@5 calculation"
        )

    retrieved_ids = run.outputs.get("chunk_ids", [])[:5] if (run and run.outputs) else []
    retrieved_set = set(retrieved_ids)

    overlap = relevant_ids & retrieved_set
    hit = 1.0 if bool(overlap) else 0.0

    if hit == 1.0:
        comment = f"HIT - Found relevant chunk(s): {sorted(list(overlap))} | Retrieved top 5: {retrieved_ids}"
    else:
        comment = f"MISS - Expected any of: {sorted(list(relevant_ids))} | Retrieved top 5: {retrieved_ids}"

    return EvaluationResult(
        key="recall_at_5",
        score=hit,
        comment=comment
    )


def retrieval_latency_evaluator(run: Run, example: Example | None = None) -> EvaluationResult:
    """
    Records per-query retrieval latency in seconds.
    """
    latency = run.outputs.get("latency", 0.0) if (run and run.outputs) else 0.0
    return EvaluationResult(
        key="latency_seconds",
        score=round(latency, 4)
    )


def retrieval_summary_evaluator(runs: Sequence[Run], examples: Sequence[Example]) -> EvaluationResults:
    """
    Summary evaluator computing aggregate benchmark metrics across all questions:
    - Overall Recall@5 (%)
    - Average Retrieval Latency (s)
    - P95 Retrieval Latency (s)
    - Total Hits & Total Valid Questions
    """
    recalls: list[float] = []
    latencies: list[float] = []
    unlabeled_count = 0
    hits_count = 0

    for run, example in zip(runs, examples):
        relevant_ids = set(example.outputs.get("relevant_chunk_ids", [])) if (example and example.outputs) else set()

        if not relevant_ids:
            unlabeled_count += 1
            continue

        retrieved_ids = set(run.outputs.get("chunk_ids", [])[:5]) if (run and run.outputs) else set()
        if relevant_ids & retrieved_ids:
            hits_count += 1
            recalls.append(1.0)
        else:
            recalls.append(0.0)

        if run and run.outputs and "latency" in run.outputs:
            latencies.append(run.outputs["latency"])

    valid_count = len(recalls)
    overall_recall = float(np.mean(recalls)) if recalls else 0.0
    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0

    return {
        "results": [
            EvaluationResult(key="overall_recall_at_5", score=overall_recall),
            EvaluationResult(key="average_latency_s", score=avg_latency),
            EvaluationResult(key="p95_latency_s", score=p95_latency),
            EvaluationResult(key="total_valid_questions", score=valid_count),
            EvaluationResult(key="total_hits", score=hits_count),
            EvaluationResult(key="unlabeled_questions", score=unlabeled_count)
        ]
    }


# ---------------------------------
# Experiment Runner
# ---------------------------------
def run_single_experiment(
    mode: str,
    client: Client,
    dataset_name: str,
    workspace_id: int = WORKSPACE_ID,
    top_k: int = DEFAULT_TOP_K,
    limit: int | None = None
) -> dict[str, Any]:
    """
    Runs a LangSmith evaluation experiment for a specific retrieval configuration.
    """
    config_info = CONFIGURATIONS[mode]
    target_fn = create_retrieval_target(mode=mode, workspace_id=workspace_id, top_k=top_k)

    print("\n" + "=" * 70)
    print(f"RUNNING LANGSMITH EXPERIMENT: {config_info['label']}")
    print(f"Mode: {mode}")
    print(f"Description: {config_info['description']}")
    print(f"Dataset: {dataset_name}")
    print(f"Workspace ID: {workspace_id} | Top K: {top_k}")
    print("=" * 70 + "\n")

    # If limit is specified, slice examples from dataset
    data_source: Any = dataset_name
    if limit is not None:
        examples = list(client.list_examples(dataset_name=dataset_name))[:limit]
        data_source = examples
        print(f"[LangSmith] Running on subset of {len(examples)} examples (limit={limit})")

    experiment_results = evaluate(
        target_fn,
        data=data_source,
        evaluators=[recall_at_5_evaluator, retrieval_latency_evaluator],
        summary_evaluators=[retrieval_summary_evaluator],
        metadata={
            "retrieval_mode": mode,
            "top_k": top_k,
            "workspace_id": workspace_id,
            "project": "ContextAI"
        },
        experiment_prefix=config_info["prefix"],
        max_concurrency=0,  # Sequential execution preserves clean deterministic profiling
        client=client
    )

    # Collect individual run results for local summary and failure inspection
    runs_data = []
    failed_examples = []
    latencies = []
    hits = 0
    valid_count = 0
    unlabeled_count = 0

    for result in experiment_results:
        run = result["run"]
        example = result["example"]

        relevant_ids = set(example.outputs.get("relevant_chunk_ids", [])) if (example and example.outputs) else set()
        retrieved_ids = run.outputs.get("chunk_ids", [])[:top_k] if (run and run.outputs) else []
        latency = run.outputs.get("latency", 0.0) if (run and run.outputs) else 0.0
        latencies.append(latency)

        example_inputs = example.inputs if (example and example.inputs) else {}
        question_text = example_inputs.get("question", "")
        example_id = example_inputs.get("id")

        if not relevant_ids:
            unlabeled_count += 1
            hit = None
        else:
            valid_count += 1
            if relevant_ids & set(retrieved_ids):
                hits += 1
                hit = True
            else:
                hit = False
                failed_examples.append({
                    "id": example_id,
                    "question": question_text,
                    "expected_chunk_ids": sorted(list(relevant_ids)),
                    "retrieved_chunk_ids": retrieved_ids,
                    "run_id": str(run.id) if run else ""
                })

        runs_data.append({
            "question": question_text,
            "relevant_ids": sorted(list(relevant_ids)),
            "retrieved_ids": retrieved_ids,
            "hit": hit,
            "latency": latency
        })

    recall_score = (hits / valid_count) if valid_count > 0 else 0.0
    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0

    return {
        "mode": mode,
        "label": config_info["label"],
        "experiment_name": getattr(experiment_results, "experiment_name", config_info["prefix"]),
        "valid_count": valid_count,
        "unlabeled_count": unlabeled_count,
        "hits": hits,
        "recall_at_5": recall_score,
        "avg_latency": avg_latency,
        "p95_latency": p95_latency,
        "failed_examples": failed_examples,
        "runs_data": runs_data
    }


def print_comparison_table(results_list: list[dict[str, Any]]):
    """
    Prints a formatted summary comparison table of all evaluated configurations.
    """
    print("\n" + "=" * 85)
    print(" " * 24 + "CONTEXTAI RETRIEVAL BENCHMARK SUMMARY")
    print("=" * 85)
    header = f"{'Configuration':<22} | {'Recall@5':<10} | {'Hits / Total':<14} | {'Avg Latency':<12} | {'P95 Latency':<12}"
    print(header)
    print("-" * 85)

    base_recall = None
    for res in results_list:
        if base_recall is None and res["mode"] == "faiss":
            base_recall = res["recall_at_5"]

        recall_str = f"{res['recall_at_5'] * 100:.2f}%"
        hits_str = f"{res['hits']} / {res['valid_count']}"
        avg_str = f"{res['avg_latency']:.4f}s"
        p95_str = f"{res['p95_latency']:.4f}s"

        print(f"{res['label']:<22} | {recall_str:<10} | {hits_str:<14} | {avg_str:<12} | {p95_str:<12}")

    print("=" * 85)

    # Relative Improvements
    if base_recall is not None:
        print("\nRECALL IMPROVEMENTS OVER FAISS BASELINE:")
        for res in results_list:
            if res["mode"] != "faiss":
                diff = (res["recall_at_5"] - base_recall) * 100
                print(f"  * {res['label']} vs FAISS: {diff:+.2f} percentage points")
        print()


def print_failed_examples_diagnostic(results_list: list[dict[str, Any]]):
    """
    Prints failure inspection report for each configuration to make debugging effortless.
    """
    print("\n" + "=" * 85)
    print(" " * 26 + "FAILED RETRIEVAL INSPECTION REPORT")
    print("=" * 85)

    for res in results_list:
        failed = res["failed_examples"]
        print(f"\nConfiguration: {res['label']} ({len(failed)} failed queries out of {res['valid_count']})")
        print("-" * 85)

        if not failed:
            print("  None! All questions retrieved at least one relevant ground-truth chunk.")
            continue

        for i, item in enumerate(failed, start=1):
            print(f"  [{i}] Q-ID {item['id']}: \"{item['question']}\"")
            print(f"      Expected Chunk IDs : {item['expected_chunk_ids']}")
            print(f"      Retrieved Top 5 IDs: {item['retrieved_chunk_ids']}")
            print(f"      LangSmith Run ID   : {item['run_id']}")


# ---------------------------------
# Main Entry Point
# ---------------------------------
def main():
    parser = argparse.ArgumentParser(description="LangSmith Evaluation & Tracing Runner for ContextAI")
    parser.add_argument(
        "--mode",
        choices=["all", "faiss", "bm25", "hybrid", "hybrid_rerank"],
        default="all",
        help="Retrieval configuration to evaluate (default: all)"
    )
    parser.add_argument(
        "--workspace-id",
        type=int,
        default=WORKSPACE_ID,
        help=f"Workspace ID to evaluate against (default: {WORKSPACE_ID})"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Top K chunks to retrieve (default: {DEFAULT_TOP_K})"
    )
    parser.add_argument(
        "--sync-dataset",
        action="store_true",
        help="Force re-synchronization of benchmark questions with LangSmith dataset"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of questions to evaluate (for rapid smoke testing)"
    )

    args = parser.parse_args()

    # 1. Initialize DB, BM25 and in-memory chunk map
    print("\n[Init] Initializing BM25 index and preloading chunk mapping...")
    db = SessionLocal()
    try:
        bm25_service.build_from_database(db)
    finally:
        db.close()
    preload_chunks()

    # 2. Initialize LangSmith Client & Dataset
    client = Client()
    dataset_name = ensure_benchmark_dataset(client=client, sync=args.sync_dataset)

    # 3. Determine configs to run
    if args.mode == "all":
        modes_to_run = ["faiss", "bm25", "hybrid", "hybrid_rerank"]
    else:
        modes_to_run = [args.mode]

    # 4. Execute experiments
    results = []
    for mode in modes_to_run:
        res = run_single_experiment(
            mode=mode,
            client=client,
            dataset_name=dataset_name,
            workspace_id=args.workspace_id,
            top_k=args.top_k,
            limit=args.limit
        )
        results.append(res)

    # 5. Output comparison and failure inspection
    print_comparison_table(results)
    print_failed_examples_diagnostic(results)


if __name__ == "__main__":
    main()
