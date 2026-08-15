"""The structured output of a quality analysis: statuses, findings, renderings.

A :class:`QualityReport` is the artifact a reviewer reads and a registry stores.
It is deliberately conservative in what it claims: each check reports its own
status and the evidence behind it, and the report's overall verdict is a
mechanical roll-up of those statuses. Nothing in here decides that a model is
"safe", "validated" or "fit for clinical use" -- it decides whether a submission
is *reviewable*, and surfaces what a human reviewer should look at first.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class Status(str, Enum):
    """Outcome of a single check.

    ``SKIP`` is a first-class outcome: a check that could not run because
    evidence was missing must not be silently counted as a pass.
    """

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"

    @property
    def is_blocking(self) -> bool:
        """Whether this status should block automatic acceptance."""
        return self in (Status.FAIL, Status.ERROR)


#: Ordering used to roll individual statuses up into an overall verdict.
_SEVERITY_ORDER = {
    Status.PASS: 0,
    Status.SKIP: 1,
    Status.WARN: 2,
    Status.FAIL: 3,
    Status.ERROR: 4,
}


@dataclass(frozen=True)
class CheckResult:
    """What one check concluded.

    Attributes
    ----------
    check_id:
        Stable machine-readable identifier.
    title:
        Human-readable name.
    status:
        See :class:`Status`.
    summary:
        One sentence a reviewer can read without context.
    details:
        JSON-serializable evidence (measured values, thresholds, field names).
    recommendation:
        What the submitter should do about it, when there is something to do.
    """

    check_id: str
    title: str
    status: Status
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommendation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable view of this result."""
        return {
            "check_id": self.check_id,
            "title": self.title,
            "status": self.status.value,
            "summary": self.summary,
            "details": _jsonable(self.details),
            "recommendation": self.recommendation,
        }


def _jsonable(value: Any) -> Any:
    """Recursively convert numpy scalars/arrays and Paths into JSON types."""
    import numpy as np

    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, float) and (value != value):  # nan
        return None
    if isinstance(value, Status):
        return value.value
    return value


@dataclass
class QualityReport:
    """Aggregated result of a :class:`~trustfed.quality.analyzer.QualityAnalyzer` run."""

    bundle_id: str
    results: List[CheckResult] = field(default_factory=list)
    narrative: Optional[str] = None
    analyzer_version: str = "0.1.0"
    context: Dict[str, Any] = field(default_factory=dict)

    @property
    def overall_status(self) -> Status:
        """The most severe status among the checks (``PASS`` when empty)."""
        if not self.results:
            return Status.PASS
        return max((r.status for r in self.results), key=lambda s: _SEVERITY_ORDER[s])

    @property
    def counts(self) -> Dict[str, int]:
        """Number of results per status, including zero entries."""
        out = {s.value: 0 for s in Status}
        for r in self.results:
            out[r.status.value] += 1
        return out

    @property
    def blocking(self) -> List[CheckResult]:
        """Results that should block automatic acceptance."""
        return [r for r in self.results if r.status.is_blocking]

    def result(self, check_id: str) -> Optional[CheckResult]:
        """Look up one result by check id."""
        for r in self.results:
            if r.check_id == check_id:
                return r
        return None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable view of the whole report."""
        return {
            "bundle_id": self.bundle_id,
            "analyzer_version": self.analyzer_version,
            "overall_status": self.overall_status.value,
            "counts": self.counts,
            "context": _jsonable(self.context),
            "narrative": self.narrative,
            "results": [r.to_dict() for r in self.results],
            "disclaimer": DISCLAIMER,
        }

    def to_json(self) -> str:
        """Render as UTF-8 JSON, indent 2, sorted keys, newline-terminated."""
        return (
            json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)
            + "\n"
        )

    def to_markdown(self) -> str:
        """Render as a Markdown review document."""
        icon = {
            Status.PASS: "PASS",
            Status.WARN: "WARN",
            Status.FAIL: "FAIL",
            Status.SKIP: "SKIP",
            Status.ERROR: "ERROR",
        }
        lines = [
            f"# Quality report: {self.bundle_id}",
            "",
            f"**Overall: {self.overall_status.value.upper()}** "
            f"(pass {self.counts['pass']}, warn {self.counts['warn']}, "
            f"fail {self.counts['fail']}, skip {self.counts['skip']}, "
            f"error {self.counts['error']})",
            "",
        ]
        if self.narrative:
            lines += ["## Summary", "", self.narrative.strip(), ""]
        lines += ["## Checks", "", "| check | status | finding |", "|---|---|---|"]
        for r in self.results:
            summary = r.summary.replace("|", "\\|")
            lines.append(f"| {r.title} | {icon[r.status]} | {summary} |")

        actions = [r for r in self.results if r.recommendation]
        if actions:
            lines += ["", "## Recommended actions", ""]
            for r in actions:
                lines.append(f"- **{r.title}**: {r.recommendation}")

        lines += ["", "## Evidence", ""]
        for r in self.results:
            if not r.details:
                continue
            lines.append(f"### {r.title} (`{r.check_id}`)")
            lines.append("")
            for key, value in sorted(_jsonable(r.details).items()):
                lines.append(f"- `{key}`: {value}")
            lines.append("")

        lines += ["---", "", DISCLAIMER, ""]
        return "\n".join(lines) + "\n"

    def write(self, directory: Path, *, stem: str = "quality_report") -> Dict[str, Path]:
        """Write ``<stem>.json`` and ``<stem>.md``; return the paths written."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        json_path = directory / f"{stem}.json"
        md_path = directory / f"{stem}.md"
        json_path.write_text(self.to_json(), encoding="utf-8")
        md_path.write_text(self.to_markdown(), encoding="utf-8")
        return {"json": json_path, "markdown": md_path}


DISCLAIMER = (
    "This report is produced by automated static and statistical checks over a "
    "submitted bundle. It is a triage aid for human reviewers: passing every "
    "check does not establish that a model is safe, generalizable, or suitable "
    "for clinical use, and the membership-inference screen is a lower bound on "
    "leakage, not a privacy guarantee."
)

__all__ = ["Status", "CheckResult", "QualityReport", "DISCLAIMER"]
