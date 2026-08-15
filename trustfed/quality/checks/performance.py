"""Performance checks: are the reported metrics present, sane, and even?

Two failure modes matter here. The first is a submission whose headline numbers
are missing, out of range, or too good to be true (a perfect AUC almost always
means leakage between train and test, not a perfect model). The second is a
model that performs well on average and badly for one subgroup -- an average is
exactly the statistic that hides that.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Tuple

from trustfed.metrics import metric_gap
from trustfed.quality.bundle import BundleView, QualityEvidence, as_mapping
from trustfed.quality.checks.base import Check
from trustfed.quality.errors import CheckConfigError
from trustfed.quality.report import CheckResult, Status

#: Metrics a submission must report, with the range each must fall in.
REQUIRED_METRICS: Dict[str, Tuple[float, float]] = {
    "auc": (0.0, 1.0),
    "accuracy": (0.0, 1.0),
}


def _as_float(value: Any) -> Optional[float]:
    """Coerce to float, returning ``None`` for anything non-numeric."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


class MetricsPresentCheck(Check):
    """Are required evaluation metrics reported, numeric and in range?

    Parameters
    ----------
    required:
        Mapping of metric name to ``(low, high)`` inclusive bounds.
    min_eval_samples:
        Below this evaluation-set size the result is downgraded to ``WARN``: a
        metric measured on a handful of patients is not evidence.
    suspicious_high:
        A discrimination metric at or above this value is flagged, because in
        practice it usually indicates train/test contamination rather than an
        exceptional model. Flagged, not failed -- the analyzer cannot know.
    """

    check_id = "metrics_present"
    title = "Evaluation metrics present and in range"

    def __init__(
        self,
        required: Mapping[str, Tuple[float, float]] = REQUIRED_METRICS,
        *,
        min_eval_samples: int = 100,
        suspicious_high: float = 0.999,
    ):
        if not 0.0 < suspicious_high <= 1.0:
            raise CheckConfigError("suspicious_high must be in (0, 1]")
        self.required = dict(required)
        self.min_eval_samples = int(min_eval_samples)
        self.suspicious_high = float(suspicious_high)

    def run(self, bundle: BundleView, evidence: QualityEvidence) -> CheckResult:
        """Validate the reported metrics against presence and range rules."""
        metrics = bundle.metrics
        if not metrics:
            return self._result(
                Status.FAIL,
                "No evaluation metrics reported.",
                details={"required": sorted(self.required)},
                recommendation=(
                    "Report held-out evaluation metrics "
                    f"({', '.join(sorted(self.required))}) with the evaluation "
                    "set size."
                ),
            )

        missing, out_of_range, suspicious = [], [], []
        values: Dict[str, Optional[float]] = {}
        for name, (low, high) in self.required.items():
            raw = metrics.get(name)
            value = _as_float(raw)
            values[name] = value
            if value is None:
                missing.append(name)
                continue
            if not low <= value <= high:
                out_of_range.append(f"{name}={value:g} outside [{low:g}, {high:g}]")
            elif value >= self.suspicious_high:
                suspicious.append(f"{name}={value:g}")

        n_eval = _as_float(metrics.get("n_eval") or metrics.get("n_samples"))
        details: Dict[str, Any] = {
            "values": values,
            "missing": missing,
            "out_of_range": out_of_range,
            "suspiciously_high": suspicious,
            "n_eval": n_eval,
        }

        if missing or out_of_range:
            problems = missing + out_of_range
            return self._result(
                Status.FAIL,
                "Metric problems: " + "; ".join(problems),
                details=details,
                recommendation="Report every required metric as a number in range.",
            )
        if suspicious:
            return self._result(
                Status.WARN,
                "Implausibly high metric(s): "
                + ", ".join(suspicious)
                + ". This usually indicates train/test contamination.",
                details=details,
                recommendation=(
                    "Confirm the evaluation set is disjoint from training data "
                    "at the patient level, not the row level."
                ),
            )
        if n_eval is not None and n_eval < self.min_eval_samples:
            return self._result(
                Status.WARN,
                f"Metrics are reported on only {int(n_eval)} evaluation samples.",
                details=details,
                recommendation=(
                    f"Evaluate on at least {self.min_eval_samples} held-out "
                    "samples, or state the confidence interval."
                ),
            )
        return self._result(
            Status.PASS,
            "Required metrics are present and in range.",
            details=details,
        )


class SubgroupGapCheck(Check):
    """Does performance differ materially between subgroups?

    Reads subgroup metrics from the evidence, falling back to
    ``bundle.metrics["subgroups"]`` (which :class:`~trustfed.quality.bundle.BundleView`
    also synthesises from a registry bundle's ``evaluation.subgroup_metrics``).
    Reports the largest gap in ``metric`` between any two subgroups with
    sufficient support, computed by :func:`trustfed.metrics.metric_gap` rather
    than re-derived here.

    Parameters
    ----------
    metric:
        Which per-group metric to compare.
    warn_gap, fail_gap:
        Absolute gap thresholds.
    min_group_n:
        Groups smaller than this are reported but excluded from the gap, since
        their metric is dominated by sampling noise.

    A ``PASS`` means no gap was detected *on the subgroups that were reported*.
    Unreported subgroups are invisible to this check, which is why the result
    lists the groups it saw.
    """

    check_id = "subgroup_gap"
    title = "Subgroup performance gap"

    def __init__(
        self,
        *,
        metric: str = "auc",
        warn_gap: float = 0.05,
        fail_gap: float = 0.10,
        min_group_n: int = 30,
    ):
        if not 0.0 <= warn_gap <= fail_gap:
            raise CheckConfigError("require 0 <= warn_gap <= fail_gap")
        self.metric = str(metric)
        self.warn_gap = float(warn_gap)
        self.fail_gap = float(fail_gap)
        self.min_group_n = int(min_group_n)

    def _collect(
        self, bundle: BundleView, evidence: QualityEvidence
    ) -> Dict[str, Dict[str, float]]:
        """Gather per-group metrics from evidence or the bundle."""
        if evidence.subgroup_metrics:
            return {str(k): dict(v) for k, v in evidence.subgroup_metrics.items()}
        raw = bundle.metrics.get("subgroups")
        groups = as_mapping(raw)
        return {str(k): as_mapping(v) for k, v in groups.items()}

    def run(self, bundle: BundleView, evidence: QualityEvidence) -> CheckResult:
        """Compute the largest between-group gap and grade it."""
        per_group = self._collect(bundle, evidence)
        if len(per_group) < 2:
            return self._skip(
                "Fewer than two subgroups reported; no gap can be computed.",
                n_groups=len(per_group),
                groups=sorted(per_group),
            )

        usable: Dict[str, float] = {}
        small: Dict[str, float] = {}
        for name, m in per_group.items():
            value = _as_float(m.get(self.metric))
            if value is None:
                continue
            n = _as_float(m.get("n"))
            if n is not None and n < self.min_group_n:
                small[name] = value
            else:
                usable[name] = value

        if len(usable) < 2:
            return self._skip(
                f"Fewer than two subgroups report a usable '{self.metric}' "
                f"with n >= {self.min_group_n}.",
                groups_with_metric=sorted(usable),
                small_groups=sorted(small),
            )

        best = max(usable, key=lambda k: usable[k])
        worst = min(usable, key=lambda k: usable[k])
        # The gap itself comes from trustfed.metrics so that this check and the
        # federated evaluation path cannot drift apart in how they define it.
        gap = metric_gap(
            {name: {self.metric: value} for name, value in usable.items()},
            self.metric,
        )
        if gap is None:  # pragma: no cover - len(usable) >= 2 guarantees a gap
            return self._skip(
                f"No two subgroups report a finite '{self.metric}'.",
                groups=sorted(usable),
            )
        details: Dict[str, Any] = {
            "metric": self.metric,
            "gap": round(gap, 6),
            "best_group": best,
            "best_value": round(usable[best], 6),
            "worst_group": worst,
            "worst_value": round(usable[worst], 6),
            "groups": {k: round(v, 6) for k, v in sorted(usable.items())},
            "excluded_small_groups": sorted(small),
            "warn_gap": self.warn_gap,
            "fail_gap": self.fail_gap,
        }
        if gap >= self.fail_gap:
            return self._result(
                Status.FAIL,
                f"{self.metric} differs by {gap:.3f} between '{worst}' "
                f"({usable[worst]:.3f}) and '{best}' ({usable[best]:.3f}).",
                details=details,
                recommendation=(
                    "Investigate the underperforming subgroup before release: "
                    "check its sample size, label quality and feature coverage, "
                    "and state the disparity in the model card."
                ),
            )
        if gap >= self.warn_gap:
            return self._result(
                Status.WARN,
                f"{self.metric} gap of {gap:.3f} between '{worst}' and '{best}'.",
                details=details,
                recommendation="Document the disparity in the model card.",
            )
        return self._result(
            Status.PASS,
            f"Largest {self.metric} gap across {len(usable)} subgroups is "
            f"{gap:.3f}.",
            details=details,
        )


__all__ = ["MetricsPresentCheck", "SubgroupGapCheck", "REQUIRED_METRICS"]
