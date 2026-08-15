"""Turning a structured report into prose, with or without a language model.

The proposal's Component 5 calls for a local LLM to draft the technical quality
report. The design constraint that matters more is that the system must work
with **no model present**: a reviewer running ``pip install trustfed`` on a
laptop with numpy and nothing else must still get a readable report.

So the narrative layer is a protocol with two implementations:

* :class:`TemplateSummarizer` -- the default. Deterministic, dependency-free,
  and derived mechanically from the check results, so it can never assert
  something the checks did not find.
* :class:`LocalLLMSummarizer` -- optional. Posts the structured report to a
  *locally configured* OpenAI-compatible endpoint (llama.cpp, Ollama, vLLM). It
  is never constructed automatically, never contacts anything unless a caller
  passes an endpoint, and falls back to the template on any failure.

Both are constrained to summarize; neither is given authority to change a
status. The statuses in the report come from the checks alone.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - typing convenience only
    from typing import Protocol, runtime_checkable
except ImportError:  # pragma: no cover
    Protocol = object  # type: ignore[assignment]

    def runtime_checkable(cls: object) -> object:  # type: ignore[no-redef]
        """No-op stand-in for Python versions without ``runtime_checkable``."""
        return cls


from trustfed.quality.report import QualityReport, Status

SYSTEM_PROMPT = (
    "You are summarizing an automated quality report for a clinical machine-"
    "learning model submitted to an open commons. Summarize only what the "
    "check results state. Do not add findings, do not speculate about clinical "
    "utility, and do not claim the model is validated or safe. Three short "
    "paragraphs: what was checked, what failed or warned, what the submitter "
    "should do next."
)


@runtime_checkable
class LLMSummarizer(Protocol):
    """Protocol for anything that turns a report into a narrative paragraph."""

    def summarize(self, report: QualityReport) -> str:
        """Return prose describing the report."""
        ...


class TemplateSummarizer:
    """Deterministic, model-free narrative generator (the default).

    Produces the same text for the same report every time, which makes reports
    diffable and reviews reproducible. Every sentence is derived from a check
    result; the summarizer never introduces a claim of its own.
    """

    name = "template"

    def __init__(self, *, max_items: int = 5):
        self.max_items = int(max_items)

    def summarize(self, report: QualityReport) -> str:
        """Render the report as three short paragraphs."""
        counts = report.counts
        n = len(report.results)
        overall = report.overall_status

        head = (
            f"Automated analysis of bundle '{report.bundle_id}' ran {n} "
            f"independent checks: {counts['pass']} passed, {counts['warn']} "
            f"raised a warning, {counts['fail']} failed, {counts['skip']} were "
            f"skipped for missing evidence, and {counts['error']} errored. "
            f"Overall status: {overall.value.upper()}."
        )

        rank = {Status.ERROR: 0, Status.FAIL: 1, Status.WARN: 2}
        flagged = [r for r in report.results if r.status in rank]
        flagged.sort(key=lambda r: rank[r.status])
        problems: List[str] = [
            f"{r.title} ({r.status.value}): {r.summary}" for r in flagged
        ]
        if problems:
            shown = problems[: self.max_items]
            more = len(problems) - len(shown)
            body = "Findings, most severe first: " + " ".join(shown)
            if more > 0:
                body += f" ...and {more} further finding(s) in the full report."
        else:
            body = (
                "No check reported a failure or a warning. Note that a clean "
                "automated report establishes reviewability, not clinical "
                "validity."
            )

        skipped = [r.title for r in report.results if r.status is Status.SKIP]
        actions = [
            f"{r.title}: {r.recommendation}"
            for r in report.results
            if r.recommendation and r.status is not Status.PASS
        ]
        tail_parts = []
        if actions:
            tail_parts.append("Recommended next steps -- " + " ".join(actions[: self.max_items]))
        if skipped:
            tail_parts.append(
                "Checks skipped for lack of evidence: " + ", ".join(skipped) + "."
            )
        if not tail_parts:
            tail_parts.append("No action is required from the submitter.")
        tail = " ".join(tail_parts)

        return "\n\n".join([head, body, tail])


class LocalLLMSummarizer:
    """Optional narrative generator backed by a locally-hosted LLM.

    Parameters
    ----------
    endpoint:
        Full URL of an OpenAI-compatible chat-completions endpoint on a host the
        operator controls (e.g. ``http://127.0.0.1:11434/v1/chat/completions``).
        There is no default: nothing is contacted unless a caller supplies one.
    model:
        Model name to request from that endpoint.
    timeout:
        Request timeout in seconds.
    fallback:
        Summarizer used when the endpoint is unreachable, slow or returns
        something unusable. Defaults to :class:`TemplateSummarizer`, so the
        report is always produced.
    allow_remote:
        Guard rail. By default only loopback endpoints are permitted, so a
        misconfiguration cannot send a submission's metadata to a third party.
        Set ``True`` deliberately if your model is on another host you control.

    Notes
    -----
    Optional by design: no part of TrustFed constructs this class for you, and
    the package has no LLM dependency. The model sees the structured report
    (statuses, summaries, numeric details) -- never the weights or any data.
    """

    name = "local_llm"

    def __init__(
        self,
        endpoint: str,
        *,
        model: str = "local-model",
        timeout: float = 30.0,
        fallback: Optional[LLMSummarizer] = None,
        allow_remote: bool = False,
        temperature: float = 0.0,
    ):
        self.endpoint = str(endpoint)
        self.model = str(model)
        self.timeout = float(timeout)
        self.fallback: LLMSummarizer = fallback or TemplateSummarizer()
        self.allow_remote = bool(allow_remote)
        self.temperature = float(temperature)
        self.last_error: Optional[str] = None

    def _is_local(self) -> bool:
        """Whether the configured endpoint points at the loopback interface."""
        from urllib.parse import urlparse

        host = (urlparse(self.endpoint).hostname or "").lower()
        return host in ("localhost", "127.0.0.1", "::1", "0.0.0.0")

    def _payload(self, report: QualityReport) -> Dict[str, Any]:
        """Build the chat-completions request body."""
        compact = {
            "bundle_id": report.bundle_id,
            "overall_status": report.overall_status.value,
            "counts": report.counts,
            "results": [
                {
                    "title": r.title,
                    "status": r.status.value,
                    "summary": r.summary,
                    "recommendation": r.recommendation,
                }
                for r in report.results
            ],
        }
        return {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(compact, sort_keys=True)},
            ],
        }

    def summarize(self, report: QualityReport) -> str:
        """Ask the local model for a narrative; fall back on any problem.

        Never raises: a summarizer failure must not lose a completed analysis.
        """
        self.last_error = None
        if not self.allow_remote and not self._is_local():
            self.last_error = (
                f"endpoint {self.endpoint!r} is not loopback and allow_remote is "
                "False; refusing to send bundle metadata off-host"
            )
            return self.fallback.summarize(report)
        try:
            text = self._request(self._payload(report))
        except Exception as exc:  # noqa: BLE001 - any failure must degrade, not raise
            self.last_error = f"{type(exc).__name__}: {exc}"
            return self.fallback.summarize(report)
        if not text or not text.strip():
            self.last_error = "empty response from endpoint"
            return self.fallback.summarize(report)
        return text.strip()

    def _request(self, payload: Dict[str, Any]) -> str:
        """POST the payload and extract the assistant message."""
        import urllib.request

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        choices = body.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return str(message.get("content") or "")


__all__ = ["LLMSummarizer", "TemplateSummarizer", "LocalLLMSummarizer", "SYSTEM_PROMPT"]
