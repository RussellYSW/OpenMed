"""The quality analyzer: run every check, roll up, narrate.

Design intent (Component 5 of the proposal): a submission to the commons should
get the same automated read every time, locally, with no network service and no
required language model. The analyzer is therefore a thin, boring orchestrator:
it owns no judgement of its own, it just runs independent
:class:`~trustfed.quality.checks.base.Check` objects over a bundle and collects
their verdicts.

Robustness rule: a check that raises is recorded as an ``ERROR`` result, not
propagated. One malformed field must not deny a reviewer the other six findings.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

from trustfed.quality.bundle import BundleView, QualityEvidence
from trustfed.quality.checks import default_checks
from trustfed.quality.checks.base import Check
from trustfed.quality.errors import BundleFormatError
from trustfed.quality.report import CheckResult, QualityReport, Status
from trustfed.quality.summarize import LLMSummarizer, TemplateSummarizer

ANALYZER_VERSION = "0.1.0"


class QualityAnalyzer:
    """Runs a pipeline of checks over a submitted model bundle.

    Parameters
    ----------
    checks:
        The checks to run. Defaults to
        :func:`trustfed.quality.checks.default_checks`. Pass your own list to
        add a domain-specific check or to tighten a threshold.
    summarizer:
        Narrative generator implementing
        :class:`~trustfed.quality.summarize.LLMSummarizer`. Defaults to the
        deterministic :class:`~trustfed.quality.summarize.TemplateSummarizer`,
        so no model is required.
    fail_fast:
        Stop at the first ``FAIL``. Off by default: a submitter benefits more
        from the full list of problems than from the first one.

    Notes
    -----
    The analyzer does not import the registry package, and reads the bundle
    through :class:`~trustfed.quality.bundle.BundleView`, so it works with any
    object exposing some subset of ``model_card``, ``metrics``, ``attestation``,
    ``lineage`` and ``weights_path`` -- including a plain dict.

    The report is a triage aid. It does not certify a model.
    """

    def __init__(
        self,
        checks: Optional[Sequence[Check]] = None,
        *,
        summarizer: Optional[LLMSummarizer] = None,
        fail_fast: bool = False,
    ):
        self.checks: List[Check] = list(checks) if checks is not None else default_checks()
        self.summarizer: LLMSummarizer = summarizer or TemplateSummarizer()
        self.fail_fast = bool(fail_fast)

    def analyze(
        self,
        bundle: Any,
        evidence: Optional[QualityEvidence] = None,
        *,
        narrate: bool = True,
    ) -> QualityReport:
        """Analyze one bundle and return a :class:`QualityReport`.

        Parameters
        ----------
        bundle:
            The submitted bundle, or anything duck-typed like one.
        evidence:
            Optional extra evidence (confidence scores, subgroup metrics,
            weights). Checks that need evidence they were not given return
            ``SKIP``.
        narrate:
            Whether to run the summarizer. Turn it off for bulk processing.

        Raises
        ------
        BundleFormatError
            Only if the bundle is not inspectable at all (``None`` or a scalar).
            Missing fields are reported as findings instead.
        """
        view = bundle if isinstance(bundle, BundleView) else BundleView(bundle)
        ev = evidence if evidence is not None else QualityEvidence()

        started = time.perf_counter()
        results: List[CheckResult] = []
        for check in self.checks:
            results.append(self._run_check(check, view, ev))
            if self.fail_fast and results[-1].status is Status.FAIL:
                break

        report = QualityReport(
            bundle_id=view.bundle_id,
            results=results,
            analyzer_version=ANALYZER_VERSION,
            context=self._context(view, ev, started, len(results)),
        )
        if narrate:
            report.narrative = self.summarizer.summarize(report)
        return report

    def _run_check(
        self, check: Check, view: BundleView, evidence: QualityEvidence
    ) -> CheckResult:
        """Run one check, converting an unexpected exception into ``ERROR``."""
        try:
            result = check.run(view, evidence)
        except Exception as exc:  # noqa: BLE001 - one bad check must not stop the review
            return CheckResult(
                check_id=getattr(check, "check_id", check.__class__.__name__),
                title=getattr(check, "title", check.__class__.__name__),
                status=Status.ERROR,
                summary=f"Check raised {type(exc).__name__}: {exc}",
                details={"exception": type(exc).__name__},
                recommendation="Report this as a bug in the check implementation.",
            )
        if not isinstance(result, CheckResult):
            return CheckResult(
                check_id=getattr(check, "check_id", check.__class__.__name__),
                title=getattr(check, "title", check.__class__.__name__),
                status=Status.ERROR,
                summary=(
                    "Check returned "
                    f"{type(result).__name__}, expected a CheckResult."
                ),
                details={},
                recommendation="Report this as a bug in the check implementation.",
            )
        return result

    def _context(
        self,
        view: BundleView,
        evidence: QualityEvidence,
        started: float,
        n_run: int,
    ) -> Dict[str, Any]:
        """Provenance block recorded alongside the results."""
        return {
            "checks_configured": [c.describe() for c in self.checks],
            "checks_run": n_run,
            "summarizer": getattr(self.summarizer, "name", type(self.summarizer).__name__),
            "bundle_fields_present": view.present_fields(),
            "bundle_fields_missing": view.missing_fields(),
            "evidence_supplied": sorted(
                name
                for name in (
                    "member_scores",
                    "nonmember_scores",
                    "subgroup_metrics",
                    "weights",
                    "reference_weights",
                )
                if getattr(evidence, name) is not None
            ),
            "evidence_notes": dict(evidence.notes),
            "elapsed_s": round(time.perf_counter() - started, 4),
        }


def analyze_bundle(
    bundle: Any,
    evidence: Optional[QualityEvidence] = None,
    **kwargs: Any,
) -> QualityReport:
    """One-shot convenience wrapper around :class:`QualityAnalyzer`."""
    return QualityAnalyzer(**kwargs).analyze(bundle, evidence)


__all__ = ["QualityAnalyzer", "analyze_bundle", "ANALYZER_VERSION", "BundleFormatError"]
