"""
Unit tests for the retrieval evaluation metrics module (metrics.py).
"""

import unittest
from benchmarks.metrics import (
    calculate_hit_at_k,
    calculate_recall_at_k,
    calculate_reciprocal_rank_at_k,
    evaluate_retrieval_pipeline
)


class TestRetrievalMetrics(unittest.TestCase):

    def test_hit_at_k_single_chunk(self):
        # Hit at rank 1
        self.assertEqual(calculate_hit_at_k([10], [10, 20, 30, 40, 50], k=5), 1.0)
        # Hit at rank 3
        self.assertEqual(calculate_hit_at_k([30], [10, 20, 30, 40, 50], k=5), 1.0)
        # Hit at rank 5
        self.assertEqual(calculate_hit_at_k([50], [10, 20, 30, 40, 50], k=5), 1.0)
        # Match beyond k=5 (at rank 6)
        self.assertEqual(calculate_hit_at_k([60], [10, 20, 30, 40, 50, 60], k=5), 0.0)
        # No match
        self.assertEqual(calculate_hit_at_k([99], [10, 20, 30, 40, 50], k=5), 0.0)

    def test_hit_at_k_multiple_chunks(self):
        # Multiple relevant chunks, one retrieved
        self.assertEqual(calculate_hit_at_k([10, 99], [10, 20, 30, 40, 50], k=5), 1.0)
        # Multiple relevant chunks, multiple retrieved
        self.assertEqual(calculate_hit_at_k([10, 20], [10, 20, 30, 40, 50], k=5), 1.0)
        # Multiple relevant chunks, none retrieved
        self.assertEqual(calculate_hit_at_k([88, 99], [10, 20, 30, 40, 50], k=5), 0.0)

    def test_hit_at_k_edge_cases(self):
        self.assertEqual(calculate_hit_at_k([], [10, 20], k=5), 0.0)
        self.assertEqual(calculate_hit_at_k([10], [], k=5), 0.0)

    def test_recall_at_k_single_chunk(self):
        # 1/1 retrieved
        self.assertEqual(calculate_recall_at_k([10], [10, 20, 30, 40, 50], k=5), 1.0)
        # 0/1 retrieved
        self.assertEqual(calculate_recall_at_k([99], [10, 20, 30, 40, 50], k=5), 0.0)
        # Match at rank 6 (outside top-5)
        self.assertEqual(calculate_recall_at_k([60], [10, 20, 30, 40, 50, 60], k=5), 0.0)

    def test_recall_at_k_multiple_chunks(self):
        # 2 relevant chunks, 2 retrieved
        self.assertAlmostEqual(calculate_recall_at_k([10, 20], [10, 20, 30, 40, 50], k=5), 1.0)
        # 2 relevant chunks, 1 retrieved (50% recall)
        self.assertAlmostEqual(calculate_recall_at_k([10, 99], [10, 20, 30, 40, 50], k=5), 0.5)
        # 3 relevant chunks, 1 retrieved (33.33% recall)
        self.assertAlmostEqual(calculate_recall_at_k([10, 88, 99], [10, 20, 30, 40, 50], k=5), 1.0 / 3.0)
        # 3 relevant chunks, 2 retrieved (66.67% recall)
        self.assertAlmostEqual(calculate_recall_at_k([10, 20, 99], [10, 20, 30, 40, 50], k=5), 2.0 / 3.0)
        # 4 relevant chunks, 1 retrieved (25% recall)
        self.assertAlmostEqual(calculate_recall_at_k([10, 77, 88, 99], [10, 20, 30, 40, 50], k=5), 0.25)
        # 4 relevant chunks, none retrieved
        self.assertAlmostEqual(calculate_recall_at_k([66, 77, 88, 99], [10, 20, 30, 40, 50], k=5), 0.0)

    def test_recall_at_k_edge_cases(self):
        self.assertEqual(calculate_recall_at_k([], [10, 20], k=5), 0.0)
        self.assertEqual(calculate_recall_at_k([10], [], k=5), 0.0)

    def test_reciprocal_rank_at_k(self):
        # Rank 1 -> 1.0
        self.assertAlmostEqual(calculate_reciprocal_rank_at_k([10], [10, 20, 30, 40, 50], k=5), 1.0)
        # Rank 2 -> 0.5
        self.assertAlmostEqual(calculate_reciprocal_rank_at_k([20], [10, 20, 30, 40, 50], k=5), 0.5)
        # Rank 3 -> 0.3333...
        self.assertAlmostEqual(calculate_reciprocal_rank_at_k([30], [10, 20, 30, 40, 50], k=5), 1.0 / 3.0)
        # Rank 4 -> 0.25
        self.assertAlmostEqual(calculate_reciprocal_rank_at_k([40], [10, 20, 30, 40, 50], k=5), 0.25)
        # Rank 5 -> 0.2
        self.assertAlmostEqual(calculate_reciprocal_rank_at_k([50], [10, 20, 30, 40, 50], k=5), 0.2)
        # Beyond rank 5 -> 0.0
        self.assertAlmostEqual(calculate_reciprocal_rank_at_k([60], [10, 20, 30, 40, 50, 60], k=5), 0.0)
        # No hit -> 0.0
        self.assertAlmostEqual(calculate_reciprocal_rank_at_k([99], [10, 20, 30, 40, 50], k=5), 0.0)
        # First match is used when multiple relevant chunks appear (e.g. at ranks 2 and 4)
        self.assertAlmostEqual(calculate_reciprocal_rank_at_k([20, 40], [10, 20, 30, 40, 50], k=5), 0.5)

    def test_mathematical_difference_hit_vs_recall(self):
        """
        Explicit test proving why Hit@5 != Recall@5 on multi-chunk questions.
        If a question has 2 relevant chunks and only 1 is in top 5:
        Hit@5 is 1.0 (binary hit occurred)
        Recall@5 is 0.5 (only 50% of the ground truth was retrieved)
        """
        rel = [100, 200]
        retrieved = [100, 1, 2, 3, 4]
        hit = calculate_hit_at_k(rel, retrieved, k=5)
        recall = calculate_recall_at_k(rel, retrieved, k=5)

        self.assertEqual(hit, 1.0)
        self.assertEqual(recall, 0.5)
        self.assertNotEqual(hit, recall)

    def test_evaluate_retrieval_pipeline(self):
        questions = [
            {"id": 1, "relevant_chunk_ids": [10, 20]},   # 2 relevant
            {"id": 2, "relevant_chunk_ids": [30]},       # 1 relevant
            {"id": 3, "relevant_chunk_ids": [40]},       # 1 relevant
            {"id": 4, "relevant_chunk_ids": []},         # 0 relevant (unlabeled, should be excluded)
        ]
        results = [
            {"chunk_ids": [10, 1, 2, 3, 4]},   # Hit=1.0, Recall=0.5, RR=1.0 (at rank 1)
            {"chunk_ids": [1, 30, 2, 3, 4]},   # Hit=1.0, Recall=1.0, RR=0.5 (at rank 2)
            {"chunk_ids": [1, 2, 3, 4, 5]},    # Hit=0.0, Recall=0.0, RR=0.0 (no hit)
            {"chunk_ids": [1, 2, 3, 4, 5]},    # Should be excluded
        ]

        eval_summary = evaluate_retrieval_pipeline(results, questions, k=5)

        self.assertEqual(eval_summary["total_questions"], 4)
        self.assertEqual(eval_summary["valid_questions"], 3)
        self.assertEqual(eval_summary["excluded_questions"], 1)

        # Valid questions metrics:
        # Q1: Hit=1, Recall=0.5, RR=1.0
        # Q2: Hit=1, Recall=1.0, RR=0.5
        # Q3: Hit=0, Recall=0.0, RR=0.0
        # Expected Hit@5: (1 + 1 + 0) / 3 * 100 = 66.6667%
        # Expected Recall@5: (0.5 + 1.0 + 0.0) / 3 * 100 = 50.0%
        # Expected MRR@5: (1.0 + 0.5 + 0.0) / 3 = 0.50
        self.assertAlmostEqual(eval_summary["hit_at_k"], 66.6666667)
        self.assertAlmostEqual(eval_summary["recall_at_k"], 50.0)
        self.assertAlmostEqual(eval_summary["mrr_at_k"], 0.50)
        self.assertEqual(eval_summary["hits_count"], 2)

    def test_empty_dataset(self):
        eval_summary = evaluate_retrieval_pipeline([], [], k=5)
        self.assertEqual(eval_summary["total_questions"], 0)
        self.assertEqual(eval_summary["valid_questions"], 0)
        self.assertIsNone(eval_summary["hit_at_k"])
        self.assertIsNone(eval_summary["recall_at_k"])
        self.assertIsNone(eval_summary["mrr_at_k"])


if __name__ == "__main__":
    unittest.main()
