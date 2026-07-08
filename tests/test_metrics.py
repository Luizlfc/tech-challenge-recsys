"""Unit tests for the from-scratch top-K ranking metrics."""
import pytest

from recsys.evaluation.metrics import hit_rate_at_k, mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k


def test_precision_at_k_counts_hits_within_k():
    ranked = [1, 2, 3, 4, 5]
    relevant = {2, 4}
    assert precision_at_k(ranked, relevant, k=3) == 1 / 3


def test_recall_at_k_divides_by_relevant_count():
    ranked = [1, 2, 3]
    relevant = {2, 4}
    assert recall_at_k(ranked, relevant, k=3) == 0.5


def test_recall_at_k_with_no_relevant_items_is_zero():
    assert recall_at_k([1, 2], set(), k=2) == 0.0


def test_hit_rate_at_k_is_binary():
    assert hit_rate_at_k([1, 2, 3], {3}, k=3) == 1.0
    assert hit_rate_at_k([1, 2, 3], {4}, k=3) == 0.0


def test_mrr_at_k_rewards_early_hits():
    assert mrr_at_k([5, 1, 2], {1}, k=3) == 1 / 2
    assert mrr_at_k([1, 2, 3], {9}, k=3) == 0.0


def test_ndcg_at_k_perfect_ranking_is_one():
    ranked = [1, 2, 3]
    relevant = {1, 2}
    assert ndcg_at_k(ranked, relevant, k=2) == pytest.approx(1.0)


def test_ndcg_at_k_no_relevant_is_zero():
    assert ndcg_at_k([1, 2, 3], set(), k=3) == 0.0
