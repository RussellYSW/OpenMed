"""Wiring of the ``trustfed`` trust plane to durable storage.

One :class:`TrustPlane` per hub process holds the three hash-chained ledgers
(registry, certification decisions, credit), the model registry rebuilt from
its ledger, the certification authority rebuilt from its decision log, the
credit ledger, the attestation gate, and the content-addressed blob store for
weights. Reviewer keys and institution service keys are (re)registered from
the database at startup by :meth:`TrustPlane.load_from_db`.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from trustfed.attestation import (
    AttestationPolicy,
    MockSoftwareAttestor,
    NodeKeyAttestor,
    NonceStore,
    Quote,
    measure_code,
)
from trustfed.certification import CertificationAuthority, Reviewer, ReviewerKeyring, ThresholdPolicy
from trustfed.certification.case import CertificationCase
from trustfed.certification.decisionlog import EVENT_CERTIFICATION
from trustfed.certification.keys import ReviewSignature
from trustfed.certification.policy import SubmissionFacts
from trustfed.certification.states import CertificationState
from trustfed.incentives import CounterpartyRegistry, CreditLedger, ReciprocityPolicy
from trustfed.ledger import FileLedger
from trustfed.ledger.crypto import HAVE_CRYPTOGRAPHY, Signer, Verifier, default_signer
from trustfed.registry import ModelRegistry, PipelineAttestation

from openmed_hub.config import HubSettings
from openmed_hub.db import ApprovedMeasurement, Institution, User

ATTESTOR_ID = "openmed-hub-software-attestor"


class BlobStore:
    """Content-addressed store for weights files: ``<root>/<aa>/<sha256>``."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, sha256: str) -> Path:
        return self.root / sha256[:2] / sha256

    def put(self, data: bytes) -> str:
        digest = hashlib.sha256(data).hexdigest()
        target = self.path(digest)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(target)
        return digest

    def exists(self, sha256: str) -> bool:
        return self.path(sha256).exists()

    def get(self, sha256: str) -> bytes:
        return self.path(sha256).read_bytes()

    @staticmethod
    def uri(sha256: str) -> str:
        return f"hub://blobs/sha256/{sha256}"


def signer_from_seed(seed_hex: str) -> Signer:
    """Deterministic signer from a stored seed (Ed25519 when available)."""
    return default_signer(bytes.fromhex(seed_hex))


def verifier_from_public_hex(public_hex: str) -> Verifier:
    """Ed25519 verifier for a client-held public key."""
    if not HAVE_CRYPTOGRAPHY:
        raise RuntimeError("client-held keys need the 'cryptography' package on the hub")
    from trustfed.ledger.crypto import Ed25519Verifier

    return Ed25519Verifier.from_public_hex(public_hex)


class TrustPlane:
    """The trustfed components behind one hub, on file-backed ledgers."""

    def __init__(self, settings: HubSettings) -> None:
        self.settings = settings
        self.lock = threading.RLock()
        ledger_dir = settings.ledger_dir
        ledger_dir.mkdir(parents=True, exist_ok=True)

        secret = settings.secret_key.encode("utf-8")
        self.registry_ledger = FileLedger(
            ledger_dir / "registry.jsonl", signer=default_signer(secret + b"|ledger:registry")
        )
        self.registry = ModelRegistry.rebuild_from_ledger(self.registry_ledger)

        self.keyring = ReviewerKeyring(seed=settings.secret_key.encode("utf-8"))
        self.certification_ledger = FileLedger(
            ledger_dir / "certification.jsonl",
            signer=default_signer(secret + b"|ledger:certification"),
        )
        self.authority = CertificationAuthority(
            self.keyring,
            self.certification_ledger,
            policy=ThresholdPolicy(
                k=settings.certification_k,
                min_institutions=settings.certification_min_institutions,
            ),
        )
        self._replay_certification()

        self.counterparties = CounterpartyRegistry()
        self.credit_ledger = FileLedger(
            ledger_dir / "credit.jsonl", signer=default_signer(secret + b"|ledger:credit")
        )
        self.credit = CreditLedger(self.credit_ledger, counterparties=self.counterparties)

        self.nonces = NonceStore()
        self.attestor = MockSoftwareAttestor(
            settings.attestation_root_key,
            policy=AttestationPolicy(
                approved_measurements=frozenset(
                    measure_code(identity) for identity in settings.approved_code_identities
                ),
                require_nonce=True,
                max_age_seconds=settings.attestation_max_age_seconds,
            ),
            nonce_store=self.nonces,
            attestor_id=ATTESTOR_ID,
        )
        # Software-measurement attestation signed at the site with its node
        # key: the hub holds only public keys here.
        self._approved_measurements = set(m.lower() for m in settings.approved_measurements)
        self.node_attestor = NodeKeyAttestor(
            policy=self._node_policy(),
            nonce_store=NonceStore(ttl_seconds=settings.attestation_max_age_seconds),
        )
        self.blobs = BlobStore(settings.blob_dir)
        self._institution_signers: Dict[str, Signer] = {}

    def _node_policy(self) -> AttestationPolicy:
        return AttestationPolicy(
            approved_measurements=frozenset(self._approved_measurements),
            require_nonce=True,
            max_age_seconds=self.settings.attestation_max_age_seconds,
        )

    # ------------------------------------------------------------ persistence

    def _replay_certification(self) -> None:
        """Rebuild in-memory certification cases from the decision log.

        The authority keeps cases in memory and treats the ledger as the
        history; a restarted hub replays that history so the cases come back
        exactly as the log recorded them.
        """
        cases: Dict[str, CertificationCase] = self.authority._cases
        for block in self.certification_ledger:
            payload = block.payload
            if payload.get("event") != EVENT_CERTIFICATION:
                continue
            bundle_id = str(payload.get("bundle_id", ""))
            action = str(payload.get("action", ""))
            if action == "submitted":
                facts_raw = payload.get("facts") or {}
                facts = SubmissionFacts(
                    bundle_id=bundle_id,
                    owner_institution=str(facts_raw.get("owner_institution", "")),
                    contributor_institutions=tuple(facts_raw.get("contributor_institutions", ())),
                    contributor_ids=tuple(facts_raw.get("contributor_ids", ())),
                    self_asserted=bool(facts_raw.get("self_asserted", True)),
                )
                cases[bundle_id] = CertificationCase(
                    bundle_id=bundle_id,
                    facts=facts,
                    state=CertificationState.SUBMITTED,
                    submitted_by=str(payload.get("actor", "")),
                    manual_ok=payload.get("manual_ok"),
                    verify=self.keyring.verify,
                )
                continue
            case = cases.get(bundle_id)
            if case is None:
                continue
            if action == "review_signed" and isinstance(payload.get("signature"), dict):
                case.signatures.append(ReviewSignature.from_dict(payload["signature"]))
            if action == "remediated" and payload.get("cleared_signatures"):
                case.signatures = []
            to_state = payload.get("to_state")
            if to_state:
                case.state = CertificationState(to_state)

    def load_from_db(self, db: Session) -> None:
        """Register reviewer, service and node keys, and approved measurements."""
        for institution in db.scalars(select(Institution)):
            self.register_institution(institution)
            self.register_node_key(institution)
        for user in db.scalars(select(User).where(User.reviewer_id.is_not(None))):
            self.register_reviewer(user)
        for row in db.scalars(select(ApprovedMeasurement).where(ApprovedMeasurement.revoked_at.is_(None))):
            self._approved_measurements.add(row.measurement.lower())
        self.node_attestor.set_policy(self._node_policy())

    # ------------------------------------------------------------- identities

    def register_institution(self, institution: Institution) -> None:
        """Bind the institution's hub-held service key as a counterparty."""
        if institution.service_key_seed:
            signer = signer_from_seed(institution.service_key_seed)
            self._institution_signers[institution.slug] = signer
            self.counterparties.register(institution.slug, signer.verifier())

    def register_node_key(self, institution: Institution) -> None:
        """Bind the institution's *verified* node key as its quote signer."""
        if institution.node_public_key and institution.node_verified_at is not None:
            self.node_attestor.register_client(
                institution.slug, verifier_from_public_hex(institution.node_public_key)
            )
        else:
            self.node_attestor.revoke_client(institution.slug)

    def approve_measurement(self, measurement: str) -> None:
        self._approved_measurements.add(measurement.lower())
        self.node_attestor.set_policy(self._node_policy())

    def revoke_measurement(self, measurement: str) -> None:
        self._approved_measurements.discard(measurement.lower())
        self.node_attestor.set_policy(self._node_policy())

    def approved_measurements(self) -> Tuple[str, ...]:
        return tuple(sorted(self._approved_measurements))

    def issue_challenge(self, client_id: str) -> str:
        """Fresh single-use nonce for a site about to sign a quote."""
        return self.node_attestor.issue_nonce(client_id)

    def verify_node_quote(self, quote: Quote) -> PipelineAttestation:
        """Check a site-signed quote; the nonce it carries must be the one issued."""
        return PipelineAttestation.from_quote(quote, attestor=self.node_attestor, expected_nonce=quote.nonce)

    def institution_signer(self, slug: str) -> Signer:
        try:
            return self._institution_signers[slug]
        except KeyError:
            raise KeyError(f"institution {slug!r} has no service key registered")

    def register_reviewer(self, user: User) -> Reviewer:
        """(Re)register a user's reviewer identity in the certification keyring."""
        if not user.reviewer_id:
            raise ValueError("user has no reviewer_id")
        reviewer = Reviewer(
            user.reviewer_id,
            user.institution.slug,
            role=user.reviewer_role_label,
        )
        if user.reviewer_public_key:
            self.keyring.register(reviewer, verifier=verifier_from_public_hex(user.reviewer_public_key))
        elif user.reviewer_key_seed:
            self.keyring.register(reviewer, signer_from_seed(user.reviewer_key_seed))
        else:
            raise ValueError("reviewer needs either a hub-held seed or a public key")
        return reviewer

    def revoke_reviewer(self, reviewer_id: str) -> None:
        if reviewer_id in self.keyring:
            self.keyring.revoke(reviewer_id)

    # ------------------------------------------------------------- attestation

    def attest(self, client_id: str, code_identity: str, config: str) -> PipelineAttestation:
        """Run a challenge-response with the hub's (software-mock) attestor.

        The mock signs whatever identity the client declares; what the gate
        enforces is that the declared identity is on the approved list and
        that the nonce is fresh. A real TEE backend implements the same
        ``Attestor`` interface.
        """
        nonce = self.attestor.issue_nonce(client_id)
        quote = self.attestor.generate_quote(client_id, code_identity, config, nonce=nonce)
        return PipelineAttestation.from_quote(quote, attestor=self.attestor, expected_nonce=nonce)

    def approved_code_identities(self) -> Iterable[str]:
        return self.settings.approved_code_identities

    def reciprocity_policy(self, exempt: Iterable[str] = ()) -> ReciprocityPolicy:
        return ReciprocityPolicy(
            min_evaluations_served=self.settings.reciprocity_min_evaluations_served,
            exempt_actors=frozenset(exempt),
            require_attested_service=True,
        )

    # ---------------------------------------------------------------- reports

    def verify_ledgers(self) -> Dict[str, Any]:
        """Verify all three chains against their anchors."""
        out: Dict[str, Any] = {}
        for name, ledger in (
            ("registry", self.registry_ledger),
            ("certification", self.certification_ledger),
            ("credit", self.credit_ledger),
        ):
            verdict = ledger.verify_chain()
            out[name] = {
                "ok": bool(getattr(verdict, "ok", False)),
                "length": len(ledger),
                "head": ledger.head_hash(),
                "detail": _verdict_dict(verdict),
            }
        return out


def _verdict_dict(verdict: Any) -> Any:
    if hasattr(verdict, "to_dict"):
        try:
            return verdict.to_dict()
        except Exception:  # pragma: no cover - defensive
            pass
    return {k: v for k, v in vars(verdict).items() if not k.startswith("_")} if hasattr(verdict, "__dict__") else str(verdict)


__all__ = ["TrustPlane", "BlobStore", "signer_from_seed", "verifier_from_public_hex", "ATTESTOR_ID"]
