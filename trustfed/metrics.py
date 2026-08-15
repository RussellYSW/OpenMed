"""Small, dependency-free evaluation metrics (numpy only).

Used by the federated server for per-round evaluation and by
:mod:`trustfed.quality` for the subgroup-gap and membership-inference screens,
so the whole project reports numbers computed the same way.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

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
    """Fraction of correct predictions at a decision ``threshold``."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_score, dtype=float) >= threshold).astype(int)
    return float(np.mean(y_pred == y_true))


def binary_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Return AUC, accuracy, sensitivity, specificity, prevalence and count.

    Sensitivity/specificity are ``nan`` when the corresponding class is absent,
    rather than silently 0, so a downstream gap calculation cannot mistake
    "no positives in this subgroup" for "perfectly bad performance".
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    pos = y == 1
    neg = y == 0
    pred = s >= threshold
    return {
        "n": float(y.shape[0]),
        "prevalence": float(np.mean(y)) if y.size else float("nan"),
        "auc": roc_auc(y, s),
        "accuracy": accuracy(y, s, threshold) if y.size else float("nan"),
        "sensitivity": float(np.mean(pred[pos])) if pos.any() else float("nan"),
        "specificity": float(np.mean(~pred[neg])) if neg.any() else float("nan"),
    }


def subgroup_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: Sequence[object],
    *,
    threshold: float = 0.5,
    group_names: Optional[Dict[object, str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute :func:`binary_metrics` separately for each subgroup.

    Parameters
    ----------
    groups:
        Per-sample group label (any hashable). Groups are reported under their
        string form, or under ``group_names[value]`` when supplied.

    Returns a mapping ``group -> metric -> value``. Empty groups are omitted.
    """
    y = np.asarray(y_true)
    s = np.asarray(y_score, dtype=float)
    g = np.asarray(list(groups), dtype=object)
    if not (y.shape[0] == s.shape[0] == g.shape[0]):
        raise ValueError(
            f"y_true ({y.shape[0]}), y_score ({s.shape[0]}) and groups "
            f"({g.shape[0]}) must have the same length"
        )
    out: Dict[str, Dict[str, float]] = {}
    for value in sorted({str(v) for v in g.tolist()}):
        mask = np.array([str(v) == value for v in g.tolist()], dtype=bool)
        if not mask.any():
            continue
        key = value
        if group_names:
            for raw, label in group_names.items():
                if str(raw) == value:
                    key = label
                    break
        out[key] = binary_metrics(y[mask], s[mask], threshold)
    return out


def metric_gap(
    per_group: Dict[str, Dict[str, float]], metric: str = "auc"
) -> Optional[float]:
    """Largest minus smallest value of ``metric`` across subgroups.

    Returns ``None`` when fewer than two subgroups have a finite value, since a
    gap is not defined then.
    """
    values = [
        m[metric]
        for m in per_group.values()
        if metric in m and m[metric] is not None and np.isfinite(m[metric])
    ]
    if len(values) < 2:
        return None
    return float(max(values) - min(values))


__all__ = [
    "roc_auc",
    "accuracy",
    "binary_metrics",
    "subgroup_metrics",
    "metric_gap",
]
