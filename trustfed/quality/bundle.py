"""Defensive, duck-typed access to a submitted model bundle.

The bundle schema is owned by the registry component (``trustfed.registry``).
The quality analyzer must not depend on it: a submission arriving from an older
client, a partially-populated draft, or a plain dictionary in a notebook all
have to be analyzable, and a missing field is a *finding*, not a crash.

:class:`BundleView` therefore wraps whatever it is given -- an object with
attributes, a mapping, or a mix -- and exposes the five fields the analyzer
cares about (``model_card``, ``metrics``, ``attestation``, ``lineage``,
``weights_path``) with ``None`` for anything absent. Nothing in this module
imports the registry package.

Two vocabularies are understood for the same facts, because the registry's
``ModelBundle`` names them differently from the analyzer's own dict form:

===============  ==========================================================
analyzer field   also read from
===============  ==========================================================
``metrics``      ``evaluation.metrics``; ``metrics["subgroups"]`` is filled
                 from ``evaluation.subgroup_metrics`` and ``metrics["n_eval"]``
                 from ``evaluation.n_samples``
``lineage``      ``parents`` (the bundle ids a model was derived from)
``weights_path`` ``weights.uri`` (a locator, not necessarily a local path)
===============  ==========================================================

The mapping is duck-typed by name, so it works for a registry object, for that
object's ``to_dict()`` output, and for a hand-built dict, without importing the
registry.

Two consequences of that translation are load-bearing and easy to get wrong:

* An empty ``parents`` tuple means "trained independently" (a *root* release),
  not "provenance withheld". :attr:`BundleView.declares_lineage_field` keeps the
  two distinguishable so a check can pass a root submission.
* ``weights.uri`` may carry a URI scheme. It is returned unchanged as a string
  in that case, because ``Path("s3://b/x")`` silently becomes ``s3:/b/x`` and a
  report must quote the locator the submitter actually declared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np

from trustfed.quality.errors import BundleFormatError

#: The bundle fields the analyzer knows how to read.
BUNDLE_FIELDS = ("model_card", "metrics", "attestation", "lineage", "weights_path")


def _get(obj: Any, name: str) -> Any:
    """Read ``name`` from an object or a mapping; ``None`` when absent."""
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def as_mapping(value: Any) -> Dict[str, Any]:
    """Best-effort conversion of a bundle sub-object into a plain dict.

    Handles mappings, dataclass-like objects with ``__dict__``, and objects
    exposing ``to_dict()``. Returns ``{}`` for anything else, so callers can
    treat "not a structured field" as "no fields present".
    """
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(k): v for k, v in value.items()}
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            result = to_dict()
        except Exception:  # pragma: no cover - defensive
            return {}
        if isinstance(result, Mapping):
            return {str(k): v for k, v in result.items()}
    data = getattr(value, "__dict__", None)
    if isinstance(data, Mapping):
        return {str(k): v for k, v in data.items() if not str(k).startswith("_")}
    return {}


class BundleView:
    """Read-only, failure-tolerant view over a submitted model bundle.

    Parameters
    ----------
    bundle:
        Any object or mapping exposing some subset of :data:`BUNDLE_FIELDS`.

    Raises
    ------
    BundleFormatError
        If ``bundle`` is ``None`` or a scalar, i.e. not inspectable at all.
    """

    def __init__(self, bundle: Any):
        if bundle is None or isinstance(bundle, (str, bytes, int, float, bool)):
            raise BundleFormatError(
                f"bundle must be an object or mapping, got {type(bundle).__name__}"
            )
        self._bundle = bundle

    @property
    def raw(self) -> Any:
        """The wrapped object, for checks that need something unusual."""
        return self._bundle

    @property
    def bundle_id(self) -> str:
        """Best-effort identifier for reporting; falls back to a placeholder."""
        for name in ("bundle_id", "model_id", "id", "name"):
            value = _get(self._bundle, name)
            if isinstance(value, str) and value:
                return value
        card = self.model_card
        for name in ("model_id", "name", "title"):
            value = card.get(name)
            if isinstance(value, str) and value:
                return value
        return "unidentified-bundle"

    @property
    def model_card(self) -> Dict[str, Any]:
        """The model card as a dict (``{}`` when missing or unreadable)."""
        return as_mapping(_get(self._bundle, "model_card"))

    @property
    def metrics(self) -> Dict[str, Any]:
        """Reported evaluation metrics as a dict.

        Falls back to the registry vocabulary: ``evaluation.metrics``, with
        ``subgroups`` synthesised from ``evaluation.subgroup_metrics`` and
        ``n_eval`` from ``evaluation.n_samples``. The returned dict is a copy;
        mutating it does not touch the bundle.
        """
        metrics = as_mapping(_get(self._bundle, "metrics"))
        evaluation = as_mapping(_get(self._bundle, "evaluation"))
        if not evaluation:
            return metrics
        if not metrics:
            metrics = dict(as_mapping(evaluation.get("metrics")))
        if "subgroups" not in metrics:
            subgroups = as_mapping(evaluation.get("subgroup_metrics"))
            if subgroups:
                metrics["subgroups"] = subgroups
        if "n_eval" not in metrics and "n_samples" not in metrics:
            n_samples = evaluation.get("n_samples")
            if n_samples is not None:
                metrics["n_eval"] = n_samples
        return metrics

    @property
    def attestation(self) -> Dict[str, Any]:
        """Attestation record as a dict."""
        return as_mapping(_get(self._bundle, "attestation"))

    @property
    def lineage(self) -> List[Any]:
        """Lineage entries as a list (a single record is wrapped in a list).

        Falls back to the registry's ``parents`` tuple of bundle ids, which is
        that vocabulary's statement of "what this model was derived from".
        """
        entries = self._as_entries(_get(self._bundle, "lineage"))
        if entries:
            return entries
        return self._as_entries(_get(self._bundle, "parents"))

    @property
    def declares_lineage_field(self) -> bool:
        """Whether the bundle *has* a lineage/parents field, empty or not.

        This distinguishes the two cases a provenance check must treat
        differently: a bundle that never mentions provenance at all, and a
        registry bundle whose ``parents`` tuple is deliberately empty because
        the model was trained independently (a root release, which is what the
        first submission of every model line looks like).
        """
        for name in ("lineage", "parents"):
            if _get(self._bundle, name) is not None:
                return True
        return False

    @staticmethod
    def _as_entries(value: Any) -> List[Any]:
        """Normalise a lineage-ish field into a list of entries."""
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        if isinstance(value, Mapping):
            entries = value.get("entries")
            if isinstance(entries, (list, tuple)):
                return list(entries)
            return [value]
        entries = getattr(value, "entries", None)
        if isinstance(entries, (list, tuple)):
            return list(entries)
        return [value]

    @property
    def weights_path(self) -> Optional[Union[Path, str]]:
        """Path to -- or locator of -- the weights artifact, if one is declared.

        Reads ``weights_path`` first, then the registry's ``weights.uri``. The
        registry stores a *locator* plus a digest rather than a filesystem path,
        and a locator carrying a URI scheme (``s3://bucket/model.npz``) is
        returned verbatim as a string: ``Path`` collapses ``://`` to ``:/`` and
        would put a locator in the report that does not match the bundle.
        Values that look like filesystem paths are returned as :class:`Path`.

        Either way the artifact may not be readable locally -- the
        parameter-norm check treats that as a ``SKIP``, not a failure.
        """
        value = _get(self._bundle, "weights_path")
        if value is None:
            value = as_mapping(_get(self._bundle, "weights")).get("uri")
        if value is None:
            return None
        if isinstance(value, Path):
            return value
        if isinstance(value, str):
            if not value.strip():
                return None
            if "://" in value:
                return value
            return Path(value)
        try:
            return Path(value)
        except TypeError:
            return None

    def resolved_fields(self) -> Dict[str, Any]:
        """Return the resolved value of every entry in :data:`BUNDLE_FIELDS`.

        "Resolved" means after the registry-vocabulary fallbacks above, so a
        registry ``ModelBundle`` and a hand-built analyzer dict produce the same
        five keys.
        """
        return {
            "model_card": self.model_card,
            "metrics": self.metrics,
            "attestation": self.attestation,
            "lineage": self.lineage,
            "weights_path": self.weights_path,
        }

    def present_fields(self) -> List[str]:
        """Which of :data:`BUNDLE_FIELDS` the bundle actually supplies."""
        resolved = self.resolved_fields()
        out = []
        for name in BUNDLE_FIELDS:
            value = resolved[name]
            if value is None:
                continue
            if isinstance(value, (Mapping, list, tuple)) and len(value) == 0:
                continue
            out.append(name)
        return out

    def missing_fields(self) -> List[str]:
        """Which of :data:`BUNDLE_FIELDS` are absent or empty."""
        present = set(self.present_fields())
        return [f for f in BUNDLE_FIELDS if f not in present]


@dataclass
class QualityEvidence:
    """Optional evidence a submitter can supply alongside the bundle.

    None of it is required: every check that needs evidence degrades to a
    ``SKIP`` result explaining what was missing, so the analyzer still produces a
    usable report for a bare bundle.

    Attributes
    ----------
    member_scores:
        Model confidence on examples that **were** in the training set.
    nonmember_scores:
        Model confidence on examples that were **not**. Together these drive the
        membership-inference screen.
    subgroup_metrics:
        ``group -> metric -> value``. If absent, the analyzer looks for
        ``metrics["subgroups"]`` in the bundle.
    weights:
        Parameter vector, if the caller already has it in memory. Otherwise the
        parameter-norm check tries ``weights_path``.
    reference_weights:
        Optional comparison model (e.g. the previous release) for the
        parameter-norm screen.
    notes:
        Free-form provenance notes echoed into the report.
    """

    member_scores: Optional[np.ndarray] = None
    nonmember_scores: Optional[np.ndarray] = None
    subgroup_metrics: Optional[Dict[str, Dict[str, float]]] = None
    weights: Optional[np.ndarray] = None
    reference_weights: Optional[np.ndarray] = None
    notes: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_predictions(
        cls,
        *,
        train_scores: Optional[Iterable[float]] = None,
        holdout_scores: Optional[Iterable[float]] = None,
        y_true: Optional[Iterable[int]] = None,
        y_score: Optional[Iterable[float]] = None,
        groups: Optional[Sequence[object]] = None,
        weights: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> "QualityEvidence":
        """Build evidence from raw predictions.

        ``train_scores``/``holdout_scores`` become the membership-inference
        inputs (converted to *confidence in the predicted class*, which is what
        the threshold attack uses). ``y_true``/``y_score``/``groups`` are reduced
        to subgroup metrics via :func:`trustfed.metrics.subgroup_metrics`.
        """
        from trustfed.metrics import subgroup_metrics as _subgroup_metrics

        sub = None
        if y_true is not None and y_score is not None and groups is not None:
            sub = _subgroup_metrics(
                np.asarray(list(y_true)), np.asarray(list(y_score), dtype=float), groups
            )
        return cls(
            member_scores=(
                None if train_scores is None else np.asarray(list(train_scores), dtype=float)
            ),
            nonmember_scores=(
                None
                if holdout_scores is None
                else np.asarray(list(holdout_scores), dtype=float)
            ),
            subgroup_metrics=sub,
            weights=weights,
            **kwargs,
        )


__all__ = ["BundleView", "QualityEvidence", "BUNDLE_FIELDS", "as_mapping"]
