"""
Retrieval evaluation metrics module for ContextAI benchmarks.

Provides technically correct implementations of:
- Hit@K
- Recall@K (macro-averaged per-question recall)
- MRR@K (Mean Reciprocal Rank @ K)

Notes on Hit@K vs Recall@K:
When each question has exactly one relevant chunk, Hit@K and Recall@K are
mathematically equivalent. However, when questions have multiple relevant
chunks (e.g. 64% of ContextAI's benchmark questions have 2 to 4 relevant chunks):
- Hit@K answers: "Was AT LEAST ONE relevant chunk found in the top-K?" (binary 1 or 0)
- Recall@K answers: "What fraction of the relevant chunks were found in the top-K?"
  (len(relevant & retrieved[:K]) / len(relevant))
"""

from typing import Any, Sequence
import numpy as np


def calculate_hit_at_k(
    relevant_chunk_ids: Sequence[int] | set[int],
    retrieved_chunk_ids: Sequence[int],
    k: int = 5
) -> float:
    """
    Computes Hit@K for a single question.

    Returns 1.0 if at least one relevant chunk appears in the top-K
    retrieved chunks, otherwise 0.0.

    Args:
        relevant_chunk_ids: Collection of ground-truth relevant chunk IDs.
        retrieved_chunk_ids: Ordered list of retrieved chunk IDs.
        k: Cutoff rank (default 5).

    Returns:
        1.0 or 0.0.
    """
    rel_set = set(relevant_chunk_ids)
    if not rel_set:
        return 0.0

    top_k_retrieved = retrieved_chunk_ids[:k]
    return 1.0 if any(cid in rel_set for cid in top_k_retrieved) else 0.0


def calculate_recall_at_k(
    relevant_chunk_ids: Sequence[int] | set[int],
    retrieved_chunk_ids: Sequence[int],
    k: int = 5
) -> float:
    """
    Computes Recall@K for a single question.

    Recall@K = |relevant ∩ retrieved[:K]| / |relevant|

    Args:
        relevant_chunk_ids: Collection of ground-truth relevant chunk IDs.
        retrieved_chunk_ids: Ordered list of retrieved chunk IDs.
        k: Cutoff rank (default 5).

    Returns:
        Float in [0.0, 1.0] representing the fraction of relevant chunks retrieved.
        Returns 0.0 if relevant_chunk_ids is empty.
    """
    rel_set = set(relevant_chunk_ids)
    if not rel_set:
        return 0.0

    top_k_set = set(retrieved_chunk_ids[:k])
    hits = len(rel_set & top_k_set)
    return hits / float(len(rel_set))


def calculate_reciprocal_rank_at_k(
    relevant_chunk_ids: Sequence[int] | set[int],
    retrieved_chunk_ids: Sequence[int],
    k: int = 5
) -> float:
    """
    Computes Reciprocal Rank @ K for a single question.

    Finds the 1-based rank of the FIRST relevant chunk among the top-K.
    - Rank 1 -> 1.0
    - Rank 2 -> 0.5
    - Rank 3 -> 0.3333...
    - Rank 4 -> 0.25
    - Rank 5 -> 0.20
    - Not in top-K -> 0.0

    Args:
        relevant_chunk_ids: Collection of ground-truth relevant chunk IDs.
        retrieved_chunk_ids: Ordered list of retrieved chunk IDs.
        k: Cutoff rank (default 5).

    Returns:
        Float representing the reciprocal rank.
    """
    rel_set = set(relevant_chunk_ids)
    if not rel_set:
        return 0.0

    top_k_retrieved = retrieved_chunk_ids[:k]
    for rank, cid in enumerate(top_k_retrieved, start=1):
        if cid in rel_set:
            return 1.0 / rank

    return 0.0


def evaluate_retrieval_pipeline(
    results: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    k: int = 5
) -> dict[str, Any]:
    """
    Evaluates retrieval results against ground-truth questions.

    Calculates:
    - Hit@K (macro-averaged hit rate)
    - Recall@K (macro-averaged per-question recall)
    - MRR@K (macro-averaged reciprocal rank)

    Questions with no ground-truth relevant chunks (None or empty list)
    are safely excluded from metric calculations, preventing division by zero.

    Args:
        results: List of retrieval result dicts, each containing 'chunk_ids'.
        questions: List of question dicts, each containing 'relevant_chunk_ids'.
        k: Evaluation rank cutoff (default 5).

    Returns:
        Dict containing:
            total_questions: int
            valid_questions: int
            excluded_questions: int
            hit_at_k: float (percentage 0.0 - 100.0) or None
            recall_at_k: float (percentage 0.0 - 100.0) or None
            mrr_at_k: float (0.0 - 1.0) or None
            per_question_metrics: list of per-question metric dictionaries
    """
    total_questions = len(questions)
    valid_questions = 0
    excluded_questions = 0

    hits_list: list[float] = []
    recalls_list: list[float] = []
    rr_list: list[float] = []
    per_question_metrics: list[dict[str, Any]] = []

    for idx, (result, question) in enumerate(zip(results, questions), start=1):
        rel_chunks = question.get("relevant_chunk_ids")
        if not rel_chunks or not isinstance(rel_chunks, (list, set, tuple)) or len(rel_chunks) == 0:
            excluded_questions += 1
            continue

        valid_questions += 1
        retrieved_ids = result.get("chunk_ids", [])

        # Ensure top-K slice
        top_k_ids = retrieved_ids[:k]

        hit = calculate_hit_at_k(rel_chunks, top_k_ids, k=k)
        recall = calculate_recall_at_k(rel_chunks, top_k_ids, k=k)
        rr = calculate_reciprocal_rank_at_k(rel_chunks, top_k_ids, k=k)

        hits_list.append(hit)
        recalls_list.append(recall)
        rr_list.append(rr)

        per_question_metrics.append({
            "question_index": idx,
            "question_id": question.get("id", idx),
            "relevant_chunk_ids": list(rel_chunks),
            "retrieved_chunk_ids": list(top_k_ids),
            "hit_at_k": hit,
            "recall_at_k": recall,
            "reciprocal_rank_at_k": rr,
        })

    if valid_questions == 0:
        return {
            "total_questions": total_questions,
            "valid_questions": 0,
            "excluded_questions": excluded_questions,
            "hit_at_k": None,
            "recall_at_k": None,
            "mrr_at_k": None,
            "per_question_metrics": [],
        }

    # Macro-average across all valid questions
    macro_hit = float(np.mean(hits_list)) * 100.0
    macro_recall = float(np.mean(recalls_list)) * 100.0
    macro_mrr = float(np.mean(rr_list))

    return {
        "total_questions": total_questions,
        "valid_questions": valid_questions,
        "excluded_questions": excluded_questions,
        "hit_at_k": macro_hit,
        "recall_at_k": macro_recall,
        "mrr_at_k": macro_mrr,
        "hits_count": int(sum(hits_list)),
        "per_question_metrics": per_question_metrics,
    }
