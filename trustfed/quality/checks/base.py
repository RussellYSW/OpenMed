"""The ``Check`` interface every quality check implements.

Checks are deliberately independent: each one reads what it needs from the
bundle view and the evidence, and returns a single
:class:`~trustfed.quality.report.CheckResult`. They never raise for a *missing
field* -- that is a finding -- and the analyzer converts any unexpected
exception into an ``ERROR`` result so one broken check cannot take down a
review.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from trustfed.quality.bundle import BundleView, QualityEvidence
from trustfed.quality.report import CheckResult, Status


class Check(ABC):
    """One independent quality check.

    Subclasses set :attr:`check_id` and :attr:`title` and implement
    :meth:`run`. Thresholds belong on the instance so a deployment can tune them
    without editing code.
    """

    check_id: str = "check"
    title: str = "Check"

    @abstractmethod
    def run(self, bundle: BundleView, evidence: QualityEvidence) -> CheckResult:
        """Evaluate the bundle and return a result."""

    def describe(self) -> Dict[str, Any]:
        """JSON-serializable description of this check's configuration."""
        return {
            "check_id": self.check_id,
            "title": self.title,
            "parameters": {
                k: v
                for k, v in vars(self).items()
                if not k.startswith("_") and isinstance(v, (int, float, str, bool))
            },
        }

    # -- helpers for subclasses --------------------------------------------

    def _result(
        self,
        status: Status,
        summary: str,
        *,
        details: Optional[Dict[str, Any]] = None,
        recommendation: Optional[str] = None,
    ) -> CheckResult:
        """Build a :class:`CheckResult` carrying this check's identity."""
        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status=status,
            summary=summary,
            details=details or {},
            recommendation=recommendation,
        )

    def _skip(self, reason: str, **details: Any) -> CheckResult:
        """Build a ``SKIP`` result explaining exactly what was missing."""
        return self._result(
            Status.SKIP,
            reason,
            details=dict(details),
            recommendation="Supply the missing evidence and re-run the analysis.",
        )


__all__ = ["Check"]
