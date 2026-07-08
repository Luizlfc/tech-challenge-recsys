"""Top-K ranking metrics implemented from scratch (no external recsys metrics lib)."""
import math


def precision_at_k(ranked_items: list[int], relevant_items: set[int], k: int) -> float:
    """Fraction of the top-k recommendations that are relevant."""
    if k == 0:
        return 0.0
    hits = len(set(ranked_items[:k]) & relevant_items)
    return hits / k


def recall_at_k(ranked_items: list[int], relevant_items: set[int], k: int) -> float:
    """Fraction of the relevant items captured within the top-k recommendations."""
    if not relevant_items:
        return 0.0
    hits = len(set(ranked_items[:k]) & relevant_items)
    return hits / len(relevant_items)


def hit_rate_at_k(ranked_items: list[int], relevant_items: set[int], k: int) -> float:
    """1.0 if any relevant item appears in the top-k, else 0.0."""
    return 1.0 if set(ranked_items[:k]) & relevant_items else 0.0


def mrr_at_k(ranked_items: list[int], relevant_items: set[int], k: int) -> float:
    """Reciprocal rank of the first relevant item found in the top-k."""
    for position, item in enumerate(ranked_items[:k]):
        if item in relevant_items:
            return 1.0 / (position + 1)
    return 0.0


def ndcg_at_k(ranked_items: list[int], relevant_items: set[int], k: int) -> float:
    """Normalized discounted cumulative gain of the top-k recommendations."""
    dcg = sum(1.0 / math.log2(pos + 2) for pos, item in enumerate(ranked_items[:k]) if item in relevant_items)
    ideal_hits = min(len(relevant_items), k)
    idcg = sum(1.0 / math.log2(pos + 2) for pos in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0
