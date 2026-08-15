"""Leakage screens: membership inference and parameter-norm outliers.

Neither check is a privacy guarantee, and both are written to say so.

The membership-inference screen implements the simplest attack in the
literature -- a global threshold on the model's confidence (Yeom et al., CSF
2018; Shokri et al., IEEE S&P 2017 for the stronger shadow-model version). If a
single threshold on confidence separates training members from non-members, the
model is memorizing, and stronger attacks will do better. The converse does not
hold: a passing score means *this* attack failed, not that the model is private.

The parameter-norm screen looks for a submitted weight vector that is
structurally anomalous -- non-finite entries, an enormous norm, or a few
coordinates carrying implausible weight. That is a cheap tripwire for a poisoned
or backdoored submission, not a detector: a competent attacker keeps the norm
inside the honest range (which is exactly what
:mod:`trustfed.attack.adaptive` does).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from trustfed.metrics import roc_auc
from trustfed.quality.bundle import BundleView, QualityEvidence
from trustfed.quality.checks.base import Check
from trustfed.quality.errors import CheckConfigError
from trustfed.quality.report import CheckResult, Status


def threshold_attack_advantage(
    member_scores: np.ndarray, nonmember_scores: np.ndarray
) -> Tuple[float, float, float]:
    """Run a global-threshold membership-inference attack.

    Parameters
    ----------
    member_scores, nonmember_scores:
        Model confidence values for training members and non-members. Any
        monotone confidence signal works (predicted probability of the true
        class, negative loss, ...), as long as both arrays use the same one.

    Returns
    -------
    (attack_auc, advantage, best_threshold)
        ``attack_auc`` is the AUC of the confidence signal for distinguishing
        members (0.5 = no signal). ``advantage`` is the maximum of
        ``|TPR - FPR|`` over all thresholds: the absolute value is deliberate,
        because an attacker is free to invert its decision rule, so a model
        whose members are *less* confident than non-members leaks just as much.
        ``best_threshold`` is where that maximum occurs.
    """
    m = np.asarray(member_scores, dtype=float).ravel()
    nm = np.asarray(nonmember_scores, dtype=float).ravel()
    labels = np.concatenate([np.ones_like(m), np.zeros_like(nm)])
    scores = np.concatenate([m, nm])
    auc = roc_auc(labels, scores)

    candidates = np.unique(scores)
    best_adv, best_thr = 0.0, float(candidates[0]) if candidates.size else 0.0
    for thr in candidates:
        tpr = float(np.mean(m >= thr)) if m.size else 0.0
        fpr = float(np.mean(nm >= thr)) if nm.size else 0.0
        adv = abs(tpr - fpr)
        if adv > best_adv:
            best_adv, best_thr = adv, float(thr)
    return float(auc), float(best_adv), best_thr


class MembershipInferenceCheck(Check):
    """Threshold membership-inference screen over confidence scores.

    Parameters
    ----------
    warn_advantage, fail_advantage:
        Membership-advantage thresholds. The defaults (0.10 / 0.20) are
        conventional screening values, not derived from a privacy calculus; a
        deployment with a formal budget should set them from that budget.
    min_samples:
        Minimum size of each group; below it the check skips rather than
        reporting a number dominated by noise.

    Requires ``evidence.member_scores`` and ``evidence.nonmember_scores``, or a
    precomputed ``metrics["membership_inference"]`` block containing
    ``member_scores``/``nonmember_scores``. Skips with an explanation otherwise.

    What it does not do: it does not run shadow-model or per-example
    calibrated attacks, which are strictly stronger, and it cannot certify
    anything. Treat a ``PASS`` as "the cheapest attack found nothing".
    """

    check_id = "membership_inference"
    title = "Membership-inference leakage screen"

    def __init__(
        self,
        *,
        warn_advantage: float = 0.10,
        fail_advantage: float = 0.20,
        min_samples: int = 30,
    ):
        if not 0.0 <= warn_advantage <= fail_advantage <= 1.0:
            raise CheckConfigError(
                "require 0 <= warn_advantage <= fail_advantage <= 1"
            )
        self.warn_advantage = float(warn_advantage)
        self.fail_advantage = float(fail_advantage)
        self.min_samples = int(min_samples)

    def _scores(
        self, bundle: BundleView, evidence: QualityEvidence
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Locate member/non-member confidence arrays in evidence or bundle."""
        if evidence.member_scores is not None and evidence.nonmember_scores is not None:
            return (
                np.asarray(evidence.member_scores, dtype=float),
                np.asarray(evidence.nonmember_scores, dtype=float),
            )
        block = bundle.metrics.get("membership_inference")
        if isinstance(block, dict):
            m = block.get("member_scores")
            nm = block.get("nonmember_scores")
            if m is not None and nm is not None:
                return (
                    np.asarray(m, dtype=float).ravel(),
                    np.asarray(nm, dtype=float).ravel(),
                )
        return None, None

    def run(self, bundle: BundleView, evidence: QualityEvidence) -> CheckResult:
        """Run the threshold attack and grade the membership advantage."""
        m, nm = self._scores(bundle, evidence)
        if m is None or nm is None:
            return self._skip(
                "No member/non-member confidence scores supplied; leakage was "
                "not screened.",
                needs=["evidence.member_scores", "evidence.nonmember_scores"],
            )
        if m.size < self.min_samples or nm.size < self.min_samples:
            return self._skip(
                f"Need at least {self.min_samples} scores per group; got "
                f"{m.size} member and {nm.size} non-member.",
                n_member=int(m.size),
                n_nonmember=int(nm.size),
            )
        if not (np.all(np.isfinite(m)) and np.all(np.isfinite(nm))):
            return self._result(
                Status.ERROR,
                "Confidence scores contain non-finite values.",
                details={"n_member": int(m.size), "n_nonmember": int(nm.size)},
                recommendation="Regenerate the confidence arrays.",
            )

        auc, advantage, threshold = threshold_attack_advantage(m, nm)
        details: Dict[str, Any] = {
            "attack_auc": round(auc, 6),
            "membership_advantage": round(advantage, 6),
            "best_threshold": round(threshold, 6),
            "n_member": int(m.size),
            "n_nonmember": int(nm.size),
            "mean_member_confidence": round(float(np.mean(m)), 6),
            "mean_nonmember_confidence": round(float(np.mean(nm)), 6),
            "warn_advantage": self.warn_advantage,
            "fail_advantage": self.fail_advantage,
            "attack": "global confidence threshold (Yeom et al., CSF 2018)",
        }
        if advantage >= self.fail_advantage:
            return self._result(
                Status.FAIL,
                f"A single confidence threshold separates training members from "
                f"non-members with advantage {advantage:.3f} "
                f"(attack AUC {auc:.3f}).",
                details=details,
                recommendation=(
                    "The model memorizes its training set. Add regularization, "
                    "early stopping, or train with differential privacy before "
                    "release, and do not publish per-example confidences."
                ),
            )
        if advantage >= self.warn_advantage:
            return self._result(
                Status.WARN,
                f"Membership advantage {advantage:.3f} (attack AUC {auc:.3f}) "
                "indicates measurable memorization.",
                details=details,
                recommendation=(
                    "Review the train/holdout confidence gap; consider stronger "
                    "regularization."
                ),
            )
        return self._result(
            Status.PASS,
            f"Threshold attack achieved advantage {advantage:.3f} "
            f"(attack AUC {auc:.3f}); no leakage detected by this screen.",
            details=details,
        )


class ParameterNormCheck(Check):
    """Structural screen over the submitted weight vector.

    Flags non-finite entries, an overall norm far outside the expected range,
    and individual coordinates whose magnitude is an extreme outlier relative to
    the rest of the vector (robust z-score on the median absolute deviation).

    Parameters
    ----------
    max_norm:
        Absolute L2 norm above which the vector is failed outright.
    warn_z, fail_z:
        Robust z-score thresholds for individual coordinates.
    max_outlier_fraction:
        Fraction of coordinates allowed to exceed ``warn_z`` before the check
        escalates.
    min_params_for_z:
        Below this parameter count the coordinate-level z-score is not computed
        at all. A robust z-score estimated from a handful of coordinates is
        noise, and reporting it produces false alarms on small models -- only
        the finiteness and total-norm screens apply there.

    Weights are read from ``evidence.weights`` if present, otherwise loaded from
    ``bundle.weights_path`` (``.npy``/``.npz`` via numpy). A missing or
    unreadable artifact is a ``SKIP``, not a crash -- and so is a remote
    locator such as ``s3://bucket/model.npz``, which is reported back verbatim
    because this check fetches nothing over a network.

    This is a tripwire, not a poisoning detector: an attacker who keeps its
    update inside the honest norm range passes it trivially.
    """

    check_id = "parameter_norm"
    title = "Parameter-norm outlier screen"

    def __init__(
        self,
        *,
        max_norm: float = 1e4,
        warn_z: float = 6.0,
        fail_z: float = 12.0,
        max_outlier_fraction: float = 0.05,
        min_params_for_z: int = 32,
    ):
        if not 0.0 < warn_z <= fail_z:
            raise CheckConfigError("require 0 < warn_z <= fail_z")
        if max_norm <= 0:
            raise CheckConfigError("max_norm must be positive")
        if min_params_for_z < 2:
            raise CheckConfigError("min_params_for_z must be >= 2")
        self.max_norm = float(max_norm)
        self.warn_z = float(warn_z)
        self.fail_z = float(fail_z)
        self.max_outlier_fraction = float(max_outlier_fraction)
        self.min_params_for_z = int(min_params_for_z)

    def _load(
        self, bundle: BundleView, evidence: QualityEvidence
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Return ``(weights, error)``; exactly one is not ``None``."""
        if evidence.weights is not None:
            return np.asarray(evidence.weights, dtype=float).ravel(), None
        declared = bundle.weights_path
        if declared is None:
            return None, "bundle declares no weights_path"
        if not isinstance(declared, Path):
            # A locator with a URI scheme (the registry's usual form). Report it
            # verbatim: rewriting it as a Path would quote something the
            # submitter never wrote.
            return None, (
                f"weights are declared at the locator '{declared}', which this "
                "check cannot fetch; it reads a local .npy/.npz file"
            )
        path = declared
        if not path.exists():
            return None, f"weights artifact not found at {path}"
        if path.suffix not in (".npy", ".npz"):
            return None, (
                f"unsupported weights format '{path.suffix}'; this check reads "
                ".npy/.npz or an in-memory array"
            )
        try:
            loaded = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            return None, f"could not read {path}: {exc}"
        if isinstance(loaded, np.lib.npyio.NpzFile):
            arrays = [np.asarray(loaded[k], dtype=float).ravel() for k in loaded.files]
            loaded.close()
            if not arrays:
                return None, "npz archive contains no arrays"
            return np.concatenate(arrays), None
        return np.asarray(loaded, dtype=float).ravel(), None

    def run(self, bundle: BundleView, evidence: QualityEvidence) -> CheckResult:
        """Screen the weight vector for structural anomalies."""
        weights, error = self._load(bundle, evidence)
        if weights is None:
            return self._skip(
                f"Weights could not be inspected: {error}.",
                needs=["evidence.weights", "bundle.weights_path (.npy/.npz)"],
            )
        if weights.size == 0:
            return self._result(
                Status.FAIL,
                "Weight vector is empty.",
                details={"n_params": 0},
                recommendation="Attach the trained parameters to the bundle.",
            )

        n_nonfinite = int(np.sum(~np.isfinite(weights)))
        if n_nonfinite:
            return self._result(
                Status.FAIL,
                f"{n_nonfinite} of {weights.size} parameters are not finite.",
                details={"n_params": int(weights.size), "n_nonfinite": n_nonfinite},
                recommendation="Training diverged; do not publish this model.",
            )

        norm = float(np.linalg.norm(weights))
        absw = np.abs(weights)
        median = float(np.median(absw))
        mad = float(np.median(np.abs(absw - median)))
        # 1.4826 scales the MAD to a standard-deviation equivalent for normals.
        # The mean-magnitude floor stops a near-zero MAD (common when most
        # coordinates are tiny) from inflating every z-score.
        scale = max(1.4826 * mad, 0.1 * float(np.mean(absw)))
        z_meaningful = weights.size >= self.min_params_for_z and scale > 0.0
        if z_meaningful:
            z = (absw - median) / scale
        else:
            z = np.zeros_like(absw)
        max_z = float(np.max(z)) if z.size else 0.0
        n_warn = int(np.sum(z >= self.warn_z))
        outlier_fraction = n_warn / float(weights.size)

        details: Dict[str, Any] = {
            "n_params": int(weights.size),
            "l2_norm": round(norm, 6),
            "max_abs": round(float(np.max(absw)), 6),
            "median_abs": round(median, 6),
            "max_robust_z": round(max_z, 4) if z_meaningful else None,
            "coordinate_screen_applied": bool(z_meaningful),
            "n_outliers": n_warn,
            "outlier_fraction": round(outlier_fraction, 4),
            "max_norm": self.max_norm,
            "warn_z": self.warn_z,
            "fail_z": self.fail_z,
        }
        if evidence.reference_weights is not None:
            ref = np.asarray(evidence.reference_weights, dtype=float).ravel()
            if ref.shape == weights.shape:
                details["distance_to_reference"] = round(
                    float(np.linalg.norm(weights - ref)), 6
                )

        if norm > self.max_norm:
            return self._result(
                Status.FAIL,
                f"Parameter L2 norm {norm:.3g} exceeds the {self.max_norm:.3g} limit.",
                details=details,
                recommendation=(
                    "An enormous parameter norm indicates divergence or a scaling "
                    "attack; retrain or investigate the contributing sites."
                ),
            )
        if max_z >= self.fail_z:
            return self._result(
                Status.FAIL,
                f"A parameter is {max_z:.1f} robust standard deviations from the "
                "rest of the vector.",
                details=details,
                recommendation=(
                    "Inspect the outlying coordinates; a small number of extreme "
                    "weights is a common signature of a backdoored submission."
                ),
            )
        if outlier_fraction > self.max_outlier_fraction or max_z >= self.warn_z:
            return self._result(
                Status.WARN,
                f"{n_warn} parameter(s) exceed {self.warn_z} robust SDs "
                f"(max {max_z:.1f}).",
                details=details,
                recommendation="Confirm the outlying weights are expected.",
            )
        if not z_meaningful:
            return self._result(
                Status.PASS,
                f"Parameter vector is finite with L2 norm {norm:.3g}; the "
                f"coordinate outlier screen needs at least "
                f"{self.min_params_for_z} parameters and was not applied.",
                details=details,
            )
        return self._result(
            Status.PASS,
            f"Parameter vector looks structurally normal "
            f"(L2 {norm:.3g}, max robust z {max_z:.1f}).",
            details=details,
        )


__all__ = [
    "MembershipInferenceCheck",
    "ParameterNormCheck",
    "threshold_attack_advantage",
]
