"""Records describing a synthetic federated cohort.

Split out from :mod:`trustfed.data.synthetic` so the declarative spec can be
imported (e.g. by the benchmark harness) without pulling in the generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from trustfed.data.errors import DataError

# Placeholder feature names. These label the columns of the synthetic design
# matrix; they are *not* derived from, or validated against, real measurements.
FEATURE_NAMES: Tuple[str, ...] = (
    "age_z",
    "years_since_diagnosis",
    "updrs_iii_baseline",
    "updrs_iii_slope",
    "moca_baseline",
    "levodopa_equivalent_dose",
    "rbd_score",
    "gait_speed",
    "tremor_index",
    "rigidity_index",
    "site_noise_a",
    "site_noise_b",
)

#: Index of the feature that the (synthetic) subgroup axis is derived from.
SUBGROUP_FEATURE = 0
SUBGROUP_NAMES: Tuple[str, str] = ("age_le_median", "age_gt_median")


@dataclass(frozen=True)
class CohortSpec:
    """Declarative description of a synthetic federated cohort.

    All randomness is a pure function of :attr:`seed`, so two runs with the same
    spec produce bit-identical arrays.

    Parameters
    ----------
    n_sites:
        Number of simulated hospitals.
    samples_per_site:
        Mean number of patients per site (before ``site_size_skew``).
    test_size:
        Size of the held-out population test set. It is drawn from the *global*
        distribution, not from any single site, so it measures how well the
        federation generalizes rather than how well it fits one silo.
    n_features:
        Total feature count; must not exceed ``len(FEATURE_NAMES)``.
    n_informative:
        How many features actually carry signal. The remainder are pure noise
        columns, which is what makes the parameter-norm screen in
        :mod:`trustfed.quality` non-trivial.
    label_skew:
        ``0.0`` gives every site the same outcome prevalence; ``1.0`` spreads
        site prevalences across roughly ``[0.10, 0.65]``.
    feature_shift:
        Standard deviation of each site's per-feature mean offset. ``0.0`` is
        IID; larger values make honest updates diverge.
    site_size_skew:
        ``0.0`` gives every site ``samples_per_site`` patients; ``1.0`` spreads
        site sizes over roughly half to 1.5x that.
    base_prevalence:
        Population outcome prevalence.
    signal_strength:
        Scale of the true coefficient vector; larger means an easier task.
    subgroup_noise:
        Extra label-flip probability applied to the *second* subgroup only.
        This is what produces a detectable, deliberate subgroup performance gap.
    seed:
        Master seed.
    """

    n_sites: int = 6
    samples_per_site: int = 220
    test_size: int = 1200
    n_features: int = 12
    n_informative: int = 8
    label_skew: float = 0.6
    feature_shift: float = 0.35
    site_size_skew: float = 0.4
    base_prevalence: float = 0.35
    signal_strength: float = 0.9
    subgroup_noise: float = 0.12
    seed: int = 0

    def validate(self) -> None:
        """Raise :class:`DataError` if the spec cannot produce a usable cohort."""
        if self.n_sites < 1:
            raise DataError(f"n_sites must be >= 1, got {self.n_sites}")
        if self.samples_per_site < 2:
            raise DataError(
                f"samples_per_site must be >= 2, got {self.samples_per_site}"
            )
        if self.test_size < 1:
            raise DataError(f"test_size must be >= 1, got {self.test_size}")
        if not 1 <= self.n_features <= len(FEATURE_NAMES):
            raise DataError(
                f"n_features must be in [1, {len(FEATURE_NAMES)}], "
                f"got {self.n_features}"
            )
        if not 1 <= self.n_informative <= self.n_features:
            raise DataError(
                f"n_informative must be in [1, n_features={self.n_features}], "
                f"got {self.n_informative}"
            )
        if not 0.0 <= self.label_skew <= 1.0:
            raise DataError(f"label_skew must be in [0, 1], got {self.label_skew}")
        if self.feature_shift < 0.0:
            raise DataError(f"feature_shift must be >= 0, got {self.feature_shift}")
        if not 0.0 <= self.site_size_skew < 1.0:
            raise DataError(
                f"site_size_skew must be in [0, 1), got {self.site_size_skew}"
            )
        if not 0.0 < self.base_prevalence < 1.0:
            raise DataError(
                f"base_prevalence must be in (0, 1), got {self.base_prevalence}"
            )
        if self.signal_strength <= 0.0:
            raise DataError(
                f"signal_strength must be > 0, got {self.signal_strength}"
            )
        if not 0.0 <= self.subgroup_noise < 0.5:
            raise DataError(
                f"subgroup_noise must be in [0, 0.5), got {self.subgroup_noise}"
            )

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable view of the spec (for provenance records)."""
        return {
            "n_sites": self.n_sites,
            "samples_per_site": self.samples_per_site,
            "test_size": self.test_size,
            "n_features": self.n_features,
            "n_informative": self.n_informative,
            "label_skew": self.label_skew,
            "feature_shift": self.feature_shift,
            "site_size_skew": self.site_size_skew,
            "base_prevalence": self.base_prevalence,
            "signal_strength": self.signal_strength,
            "subgroup_noise": self.subgroup_noise,
            "seed": self.seed,
            "synthetic": True,
        }


@dataclass(frozen=True)
class SiteData:
    """One simulated hospital's private cohort.

    Attributes
    ----------
    site_id:
        Stable identifier, ``"site_00"``-style.
    X:
        ``(n, d)`` feature matrix.
    y:
        ``(n,)`` binary outcome vector of 0/1 floats.
    groups:
        ``(n,)`` integer subgroup index (see :data:`SUBGROUP_NAMES`).
    prevalence:
        Realized outcome prevalence at this site.
    mean_shift:
        ``(d,)`` per-feature mean offset applied to this site, i.e. the ground
        truth of the covariate shift the federation has to cope with.
    """

    site_id: str
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    prevalence: float
    mean_shift: np.ndarray

    @property
    def n_samples(self) -> int:
        """Number of patients held by this site."""
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        """Feature dimensionality."""
        return int(self.X.shape[1])


@dataclass(frozen=True)
class Cohort:
    """A generated federation: per-site training data plus a shared test set."""

    sites: List[SiteData]
    X_test: np.ndarray
    y_test: np.ndarray
    test_groups: np.ndarray
    feature_names: Tuple[str, ...]
    spec: CohortSpec
    coefficients: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))

    @property
    def n_features(self) -> int:
        """Feature dimensionality of the cohort."""
        return int(self.X_test.shape[1])

    def as_tuple(self) -> Tuple[List[SiteData], np.ndarray, np.ndarray]:
        """Return ``(sites, X_test, y_test)`` for the simple training loop."""
        return self.sites, self.X_test, self.y_test

    def site_prevalences(self) -> List[float]:
        """Realized outcome prevalence per site, in site order."""
        return [s.prevalence for s in self.sites]



__all__ = [
    "FEATURE_NAMES",
    "SUBGROUP_FEATURE",
    "SUBGROUP_NAMES",
    "CohortSpec",
    "SiteData",
    "Cohort",
]
