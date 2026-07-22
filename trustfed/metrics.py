"""Small, dependency-free evaluation metrics (numpy only)."""

from __future__ import annotations

import numpy as np


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic.

    Implemented with tie-averaged ranks so we do not depend on scikit-learn at
    runtime. Returns ``float('nan')`` when only one class is present.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty(len(y_score), dtype=float)

    # Average ranks within groups of tied scores.
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # ranks are 1-based
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    sum_ranks_pos = float(np.sum(ranks[y_true == 1]))
    u_pos = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return u_pos / (n_pos * n_neg)


def accuracy(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_score, dtype=float) >= threshold).astype(int)
    return float(np.mean(y_pred == y_true))
