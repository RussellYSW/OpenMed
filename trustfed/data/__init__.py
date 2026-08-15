"""Synthetic cohort generation for TrustFed.

No real or patient-derived data ships with this project. This package generates
non-IID multi-site clinical-style cohorts in-process from an explicit seed; see
:mod:`trustfed.data.synthetic` and ``docs/DATA.md``.
"""

from __future__ import annotations

from trustfed.data.errors import DataError
from trustfed.data.cohort import (
    FEATURE_NAMES,
    SUBGROUP_NAMES,
    Cohort,
    CohortSpec,
    SiteData,
)
from trustfed.data.synthetic import (
    describe_cohort,
    make_cohort,
    make_federated_parkinson,
)

__all__ = [
    "DataError",
    "FEATURE_NAMES",
    "SUBGROUP_NAMES",
    "Cohort",
    "CohortSpec",
    "SiteData",
    "make_cohort",
    "make_federated_parkinson",
    "describe_cohort",
]
