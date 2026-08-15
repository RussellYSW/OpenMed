"""Versioned model bundles and their content-addressed identifiers.

A bundle is everything a receiving site needs to decide whether to trust and
reuse a model: a *reference* to the weights (never the weights themselves), a
model card, the attestation of the pipeline that produced it, an evaluation
report, an optional clinician-readable fine-tuning manual, and the parent
bundles it was derived from.

The bundle id is the SHA-256 of the canonical JSON of that content, so the id
*is* the integrity check: recomputing it detects any field that changed. The id
deliberately excludes wall-clock publication time, which means republishing
byte-identical content yields the same id (and the registry refuses it as a
duplicate).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Sequence, Tuple

from trustfed.attestation.attestor import AttestationResult, Attestor, Quote
from trustfed.ledger.crypto import canonical_json
from trustfed.registry.errors import ValidationError
from trustfed.registry.model_card import ModelCard

if TYPE_CHECKING:  # pragma: no cover - avoids a runtime import cycle
    from trustfed.certification.manual import ManualValidation

#: Prefix on every bundle id, so ids are self-describing in logs and ledgers.
BUNDLE_ID_PREFIX = "openmed:bundle:sha256:"


@dataclass(frozen=True)
class WeightsRef:
    """A reference to model weights held outside the registry.

    The registry stores no weights: it stores a URI plus the digest a consumer
    must check after fetching. ``uri`` may be any locator the deployment
    understands (file path, object-store key, DOI-style identifier).
    """

    uri: str
    sha256: str
    size_bytes: int = 0
    format: str = "npz"

    def __post_init__(self) -> None:
        if not self.uri:
            raise ValidationError("weights uri must not be empty")
        if len(self.sha256) != 64 or not all(
            c in "0123456789abcdef" for c in self.sha256.lower()
        ):
            raise ValidationError("weights sha256 must be 64 hex characters")
        if self.size_bytes < 0:
            raise ValidationError("size_bytes must be >= 0")

    @classmethod
    def from_bytes(cls, uri: str, blob: bytes, *, fmt: str = "npz") -> "WeightsRef":
        """Build a reference by hashing an in-memory weights blob."""
        return cls(
            uri=uri,
            sha256=hashlib.sha256(blob).hexdigest(),
            size_bytes=len(blob),
            format=fmt,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "uri": self.uri,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "format": self.format,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WeightsRef":
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            uri=str(data["uri"]),
            sha256=str(data["sha256"]),
            size_bytes=int(data.get("size_bytes", 0)),
            format=str(data.get("format", "npz")),
        )


@dataclass(frozen=True)
class EvaluationReport:
    """Metrics for one bundle on one named evaluation dataset.

    ``subgroup_metrics`` maps a subgroup label to its own metric mapping, which
    is what makes performance-gap screening possible downstream. Nothing here
    verifies that the evaluation was actually run on the data claimed.
    """

    dataset_id: str
    n_samples: int
    metrics: Mapping[str, float] = field(default_factory=dict)
    subgroup_metrics: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    protocol: str = ""
    evaluated_by: str = ""
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ValidationError("evaluation dataset_id must not be empty")
        if self.n_samples <= 0:
            raise ValidationError("evaluation n_samples must be positive")
        if not self.metrics:
            raise ValidationError("evaluation report needs at least one metric")

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "dataset_id": self.dataset_id,
            "n_samples": self.n_samples,
            "metrics": {k: float(v) for k, v in self.metrics.items()},
            "subgroup_metrics": {
                g: {k: float(v) for k, v in m.items()}
                for g, m in self.subgroup_metrics.items()
            },
            "protocol": self.protocol,
            "evaluated_by": self.evaluated_by,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationReport":
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            dataset_id=str(data["dataset_id"]),
            n_samples=int(data["n_samples"]),
            metrics=dict(data.get("metrics", {})),
            subgroup_metrics={
                g: dict(m) for g, m in dict(data.get("subgroup_metrics", {})).items()
            },
            protocol=str(data.get("protocol", "")),
            evaluated_by=str(data.get("evaluated_by", "")),
            seed=data.get("seed"),
        )


@dataclass(frozen=True)
class PipelineAttestation:
    """Record of the attestation quote that covered the producing pipeline.

    Stores the quote itself plus the verdict a verifier reached, so a later
    reader can see both what was claimed and who accepted it. ``verified=True``
    means *some* verifier accepted the quote at publication time; with the mock
    attestor that is not evidence about the code that actually ran.
    """

    client_id: str
    measurement: str
    config_hash: str
    attestor_id: str
    signature: str
    nonce: str = ""
    verified: bool = False
    verifier_reason: str = ""

    @classmethod
    def from_quote(
        cls,
        quote: Quote,
        *,
        attestor: Optional[Attestor] = None,
        result: Optional[AttestationResult] = None,
        **check_kwargs: Any,
    ) -> "PipelineAttestation":
        """Build a record from a quote, optionally verifying it right now.

        Pass ``attestor`` to have the quote checked (recommended), or ``result``
        if the check already happened. With neither, ``verified`` stays False.
        """
        if result is None and attestor is not None:
            result = attestor.check(quote, **check_kwargs)
        return cls(
            client_id=quote.client_id,
            measurement=quote.measurement,
            config_hash=quote.config_hash,
            attestor_id=quote.attestor_id,
            signature=quote.signature,
            nonce=quote.nonce,
            verified=bool(result.ok) if result is not None else False,
            verifier_reason=result.reason_code if result is not None else "unchecked",
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "client_id": self.client_id,
            "measurement": self.measurement,
            "config_hash": self.config_hash,
            "attestor_id": self.attestor_id,
            "signature": self.signature,
            "nonce": self.nonce,
            "verified": self.verified,
            "verifier_reason": self.verifier_reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PipelineAttestation":
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            client_id=str(data["client_id"]),
            measurement=str(data["measurement"]),
            config_hash=str(data["config_hash"]),
            attestor_id=str(data.get("attestor_id", "")),
            signature=str(data.get("signature", "")),
            nonce=str(data.get("nonce", "")),
            verified=bool(data.get("verified", False)),
            verifier_reason=str(data.get("verifier_reason", "")),
        )


@dataclass(frozen=True)
class ModelBundle:
    """An immutable, content-addressed, versioned model release.

    Use :meth:`create` rather than the constructor: it computes the id. The
    ``parents`` tuple is what turns the registry into a lineage graph -- a
    derived model names every bundle it was built from.
    """

    bundle_id: str
    name: str
    version: str
    weights: WeightsRef
    model_card: ModelCard
    evaluation: EvaluationReport
    attestation: PipelineAttestation
    published_by: str
    parents: Tuple[str, ...] = ()
    fine_tuning_manual: Optional[Mapping[str, Any]] = None
    tags: Tuple[str, ...] = ()

    @staticmethod
    def content_of(
        *,
        name: str,
        version: str,
        weights: WeightsRef,
        model_card: ModelCard,
        evaluation: EvaluationReport,
        attestation: PipelineAttestation,
        published_by: str,
        parents: Sequence[str] = (),
        fine_tuning_manual: Optional[Mapping[str, Any]] = None,
        tags: Sequence[str] = (),
    ) -> Dict[str, Any]:
        """Return the exact content dict the bundle id is computed over."""
        return {
            "name": name,
            "version": version,
            "weights": weights.to_dict(),
            "model_card": model_card.to_dict(),
            "evaluation": evaluation.to_dict(),
            "attestation": attestation.to_dict(),
            "published_by": published_by,
            "parents": list(parents),
            "fine_tuning_manual": dict(fine_tuning_manual or {}) or None,
            "tags": sorted(tags),
        }

    @classmethod
    def create(cls, **content: Any) -> "ModelBundle":
        """Build a bundle and derive its content-addressed id.

        Accepts the same keyword arguments as :meth:`content_of`.
        """
        if not content.get("name"):
            raise ValidationError("bundle name must not be empty")
        if not content.get("version"):
            raise ValidationError("bundle version must not be empty")
        if not content.get("published_by"):
            raise ValidationError("bundle published_by must not be empty")
        body = cls.content_of(**content)
        digest = hashlib.sha256(canonical_json(body)).hexdigest()
        return cls(
            bundle_id=BUNDLE_ID_PREFIX + digest,
            name=content["name"],
            version=content["version"],
            weights=content["weights"],
            model_card=content["model_card"],
            evaluation=content["evaluation"],
            attestation=content["attestation"],
            published_by=content["published_by"],
            parents=tuple(content.get("parents", ())),
            fine_tuning_manual=(
                dict(content["fine_tuning_manual"])
                if content.get("fine_tuning_manual")
                else None
            ),
            tags=tuple(sorted(content.get("tags", ()))),
        )

    def content(self) -> Dict[str, Any]:
        """Return this bundle's content dict (without the id)."""
        return self.content_of(
            name=self.name,
            version=self.version,
            weights=self.weights,
            model_card=self.model_card,
            evaluation=self.evaluation,
            attestation=self.attestation,
            published_by=self.published_by,
            parents=self.parents,
            fine_tuning_manual=self.fine_tuning_manual,
            tags=self.tags,
        )

    def computed_id(self) -> str:
        """Recompute the content-addressed id from the stored content."""
        return BUNDLE_ID_PREFIX + hashlib.sha256(canonical_json(self.content())).hexdigest()

    def id_is_intact(self) -> bool:
        """Return ``True`` iff the stored id matches the stored content."""
        return self.bundle_id == self.computed_id()

    def short_id(self, length: int = 12) -> str:
        """Return a shortened id for display (not for equality checks)."""
        return self.bundle_id[len(BUNDLE_ID_PREFIX) :][:length]

    def validate_manual(self) -> Optional["ManualValidation"]:
        """Validate the attached fine-tuning manual, if there is one.

        Returns ``None`` when no manual is attached, otherwise a
        :class:`trustfed.certification.manual.ManualValidation`.
        """
        if not self.fine_tuning_manual:
            return None
        from trustfed.certification.manual import FineTuningManual

        return FineTuningManual.from_dict(self.fine_tuning_manual).validate()

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict including the id."""
        body = self.content()
        body["bundle_id"] = self.bundle_id
        return body

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelBundle":
        """Rebuild a bundle from :meth:`to_dict` output (id preserved)."""
        bundle = cls.create(
            name=str(data["name"]),
            version=str(data["version"]),
            weights=WeightsRef.from_dict(data["weights"]),
            model_card=ModelCard.from_dict(data["model_card"]),
            evaluation=EvaluationReport.from_dict(data["evaluation"]),
            attestation=PipelineAttestation.from_dict(data["attestation"]),
            published_by=str(data["published_by"]),
            parents=tuple(data.get("parents", ())),
            fine_tuning_manual=data.get("fine_tuning_manual"),
            tags=tuple(data.get("tags", ())),
        )
        stored_id = data.get("bundle_id")
        if stored_id is None or stored_id == bundle.bundle_id:
            return bundle
        # Preserve the recorded id so callers can detect the mismatch.
        return cls(
            bundle_id=str(stored_id),
            name=bundle.name,
            version=bundle.version,
            weights=bundle.weights,
            model_card=bundle.model_card,
            evaluation=bundle.evaluation,
            attestation=bundle.attestation,
            published_by=bundle.published_by,
            parents=bundle.parents,
            fine_tuning_manual=bundle.fine_tuning_manual,
            tags=bundle.tags,
        )


__all__ = [
    "BUNDLE_ID_PREFIX",
    "EvaluationReport",
    "ModelBundle",
    "PipelineAttestation",
    "WeightsRef",
]
