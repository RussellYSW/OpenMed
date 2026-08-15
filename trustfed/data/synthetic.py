"""Deterministic synthetic multi-site clinical cohort generator.

**No patient data ships with TrustFed.** Everything produced by this module is
simulated in-process from an explicit seed: features are drawn from Gaussians,
labels from a logistic link over a random coefficient vector. The variable names
("years since diagnosis", "UPDRS-III") are *evocative placeholders* chosen so the
demo reads like a clinical prognosis task; they carry no empirical relationship
to real Parkinson's-disease measurements and must not be interpreted as such.
See ``docs/DATA.md`` for how to plug in a real, controlled-access cohort.

What the generator models, and why it matters for federated evaluation:

* **Feature shift (covariate shift)** — each site's feature means are drawn
  around the population mean, standing in for scanner/assay/protocol drift
  between hospitals.
* **Label skew (prior shift)** — each site has its own outcome prevalence,
  standing in for referral bias between a community clinic and a tertiary
  movement-disorder center.
* **Site-size skew** — sites hold different numbers of patients.
* **A subgroup axis** — a binary stratum with extra label noise, so that
  subgroup-gap detection (Component 5) has something real to find.

Together these make the cohort non-IID across sites, which is the regime where
robust aggregation is actually hard: honest updates already disagree, so an
aggregator cannot treat "far from the mean" as synonymous with "malicious".
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from trustfed.data.cohort import (
    FEATURE_NAMES,
    SUBGROUP_FEATURE,
    SUBGROUP_NAMES,
    Cohort,
    CohortSpec,
    SiteData,
)
from trustfed.data.errors import DataError

# Distinct RNG stream id for the held-out test set, so that changing the number
# of sites does not perturb the test draw.
_TEST_STREAM = 999


def _solve_bias(logits: np.ndarray, target: float) -> float:
    """Find ``b`` such that ``mean(sigmoid(logits + b)) ~= target``.

    Bisection on a monotone function; 80 iterations over ``[-25, 25]`` is well
    past float64 resolution for this range, so the result is deterministic.
    """
    lo, hi = -25.0, 25.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        p = float(np.mean(1.0 / (1.0 + np.exp(-np.clip(logits + mid, -30.0, 30.0)))))
        if p < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _site_sizes(spec: CohortSpec, rng: np.random.Generator) -> np.ndarray:
    """Draw per-site sample counts with the configured size skew."""
    if spec.site_size_skew <= 0.0:
        return np.full(spec.n_sites, spec.samples_per_site, dtype=int)
    lo = 1.0 - spec.site_size_skew
    hi = 1.0 + spec.site_size_skew
    factors = rng.uniform(lo, hi, size=spec.n_sites)
    sizes = np.maximum(8, np.round(factors * spec.samples_per_site)).astype(int)
    return sizes


def _site_prevalences(spec: CohortSpec) -> np.ndarray:
    """Deterministic per-site target prevalences, symmetric about the base rate.

    With ``label_skew == 0`` every site matches ``base_prevalence``; with
    ``label_skew == 1`` prevalences fan out to +/-0.25 around it (clipped into a
    range that keeps both classes present at every site).
    """
    if spec.n_sites == 1:
        return np.array([spec.base_prevalence], dtype=float)
    spread = 0.25 * spec.label_skew
    offsets = np.linspace(-spread, spread, spec.n_sites)
    return np.clip(spec.base_prevalence + offsets, 0.06, 0.94)


def _subgroups(X: np.ndarray) -> np.ndarray:
    """Binary subgroup index derived from the first feature (median split)."""
    col = X[:, SUBGROUP_FEATURE] if X.shape[1] > SUBGROUP_FEATURE else X[:, 0]
    return (col > 0.0).astype(int)


def _draw_labels(
    logits: np.ndarray,
    groups: np.ndarray,
    subgroup_noise: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample Bernoulli labels, adding extra flip noise to subgroup 1."""
    p = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    y = (rng.uniform(size=p.shape[0]) < p).astype(float)
    if subgroup_noise > 0.0:
        mask = groups == 1
        flip = mask & (rng.uniform(size=y.shape[0]) < subgroup_noise)
        y[flip] = 1.0 - y[flip]
    return y


def make_cohort(spec: Optional[CohortSpec] = None, **overrides: object) -> Cohort:
    """Generate a synthetic non-IID federated cohort from a :class:`CohortSpec`.

    Parameters
    ----------
    spec:
        The cohort specification. If omitted, defaults are used.
    **overrides:
        Field overrides applied on top of ``spec`` (e.g. ``seed=3``).

    Returns
    -------
    Cohort
        Sites, a global test set, subgroup labels, and the ground-truth
        coefficient vector (useful for sanity checks; a real deployment has no
        such oracle).

    Raises
    ------
    DataError
        If the resulting spec is invalid.

    Notes
    -----
    The data is entirely synthetic. It is suitable for testing federated
    protocols and defenses; it is **not** suitable for any clinical claim.
    """
    if spec is None:
        spec = CohortSpec()
    if overrides:
        unknown = set(overrides) - set(spec.__dataclass_fields__)
        if unknown:
            raise DataError(f"unknown CohortSpec fields: {sorted(unknown)}")
        spec = replace(spec, **overrides)  # type: ignore[arg-type]
    spec.validate()

    root = np.random.default_rng(spec.seed)
    d = spec.n_features

    beta = np.zeros(d, dtype=float)
    beta[: spec.n_informative] = root.normal(
        0.0, spec.signal_strength, size=spec.n_informative
    )

    sizes = _site_sizes(spec, root)
    prevalences = _site_prevalences(spec)
    shifts = root.normal(0.0, spec.feature_shift, size=(spec.n_sites, d))

    sites: List[SiteData] = []
    for i in range(spec.n_sites):
        rng = np.random.default_rng([spec.seed, i, 0xC0FFEE])
        n = int(sizes[i])
        X = rng.normal(0.0, 1.0, size=(n, d)) + shifts[i][None, :]
        groups = _subgroups(X)
        raw = X @ beta
        bias = _solve_bias(raw, float(prevalences[i]))
        y = _draw_labels(raw + bias, groups, spec.subgroup_noise, rng)
        sites.append(
            SiteData(
                site_id=f"site_{i:02d}",
                X=X,
                y=y,
                groups=groups,
                prevalence=float(np.mean(y)),
                mean_shift=shifts[i].copy(),
            )
        )

    test_rng = np.random.default_rng([spec.seed, _TEST_STREAM, 7])
    which = test_rng.integers(0, spec.n_sites, size=spec.test_size)
    X_test = test_rng.normal(0.0, 1.0, size=(spec.test_size, d)) + shifts[which]
    test_groups = _subgroups(X_test)
    raw_test = X_test @ beta
    bias_test = _solve_bias(raw_test, spec.base_prevalence)
    y_test = _draw_labels(
        raw_test + bias_test, test_groups, spec.subgroup_noise, test_rng
    )

    return Cohort(
        sites=sites,
        X_test=X_test,
        y_test=y_test,
        test_groups=test_groups,
        feature_names=FEATURE_NAMES[:d],
        spec=spec,
        coefficients=beta,
    )


def make_federated_parkinson(
    n_sites: int = 6,
    samples_per_site: int = 220,
    test_size: int = 1200,
    *,
    seed: int = 0,
    **spec_overrides: object,
) -> Tuple[List[SiteData], np.ndarray, np.ndarray]:
    """Convenience wrapper returning ``(sites, X_test, y_test)``.

    This is the entry point used by the tests and the ``parkinson_decline``
    example. The cohort is **synthetic**; the disease framing is a narrative
    device (see the module docstring).
    """
    cohort = make_cohort(
        CohortSpec(
            n_sites=n_sites,
            samples_per_site=samples_per_site,
            test_size=test_size,
            seed=seed,
        ),
        **spec_overrides,
    )
    return cohort.as_tuple()


def describe_cohort(cohort: Cohort) -> Dict[str, object]:
    """Summarize a cohort for logging / provenance (JSON-serializable)."""
    return {
        "spec": cohort.spec.to_dict(),
        "n_sites": len(cohort.sites),
        "site_sizes": [s.n_samples for s in cohort.sites],
        "site_prevalences": [round(p, 4) for p in cohort.site_prevalences()],
        "test_size": int(cohort.X_test.shape[0]),
        "test_prevalence": round(float(np.mean(cohort.y_test)), 4),
        "feature_names": list(cohort.feature_names),
        "subgroup_names": list(SUBGROUP_NAMES),
        "provenance": "synthetic; generated in-process from seed",
    }


__all__ = [
    "FEATURE_NAMES",
    "SUBGROUP_NAMES",
    "CohortSpec",
    "SiteData",
    "Cohort",
    "make_cohort",
    "make_federated_parkinson",
    "describe_cohort",
]
