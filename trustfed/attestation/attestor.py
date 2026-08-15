"""Remote-attestation abstraction for federated clients.

The idea mirrors hardware TEE attestation (Intel SGX/TDX, AMD SEV-SNP): before a
client's update is accepted, it must present a *quote* proving that
(1) it is running an approved code measurement, (2) the quote was produced by
genuine attestation hardware rooted in a key the verifier trusts, and (3) the
quote is fresh -- it answers a challenge the verifier issued just now.

This reference implementation is a **software mock**: the "measurement" is a hash
of the client's declared code identity, and the "quote" is signed under a key
that stands in for the TEE vendor's root of trust. It is enough to demonstrate
the protocol and to unit-test the accept/reject logic. To use a real TEE,
implement the :class:`Attestor` interface against your platform's quoting and
verification APIs (e.g. DCAP for SGX/TDX) -- the rest of TrustFed is unchanged.

SECURITY NOTE: the mock is NOT a real security boundary. A client declares its
own code identity, so a compromised host can declare an approved one; with the
default HMAC backend the verifier and the prover share a key, so a verifier can
forge quotes. Nothing here measures actual loaded code. Do not rely on it to
protect production federations.
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Iterable, Optional

from trustfed.attestation.errors import AttestationError, NonceError, PolicyError
from trustfed.attestation.policy import AttestationPolicy, NonceStore
from trustfed.ledger.crypto import HmacSigner, Signer, Verifier, canonical_json

#: Machine-readable reason codes returned by :meth:`Attestor.check`.
REASON_OK = "ok"
REASON_BAD_SIGNATURE = "bad_signature"
REASON_UNAPPROVED_MEASUREMENT = "unapproved_measurement"
REASON_UNAPPROVED_CONFIG = "unapproved_config"
REASON_MISSING_NONCE = "missing_nonce"
REASON_NONCE_MISMATCH = "nonce_mismatch"
REASON_NONCE_REJECTED = "nonce_rejected"
REASON_QUOTE_EXPIRED = "quote_expired"
REASON_QUOTE_FROM_FUTURE = "quote_from_future"


def measure_code(identity: str) -> str:
    """Return a stand-in code *measurement* (like an SGX MRENCLAVE).

    In a real TEE this is a hash of the loaded enclave image, produced by
    hardware the client cannot lie to. Here we hash a declared code-identity
    string so tests can simulate approved vs. tampered code; it is a stand-in,
    not a measurement of anything actually executing.
    """
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Quote:
    """An attestation quote presented by a client alongside its update.

    ``nonce`` and ``issued_at`` are optional so that quotes produced by the
    original two-field protocol still construct; a policy with
    ``require_nonce=True`` or ``max_age_seconds`` set will reject them.
    """

    client_id: str
    measurement: str
    config_hash: str
    signature: str
    nonce: str = ""
    issued_at: float = 0.0
    attestor_id: str = "mock-software-attestor"

    def signing_material(self) -> bytes:
        """Return the exact bytes the quote signature is computed over."""
        return canonical_json(
            {
                "client_id": self.client_id,
                "measurement": self.measurement,
                "config_hash": self.config_hash,
                "nonce": self.nonce,
                "issued_at": self.issued_at,
                "attestor_id": self.attestor_id,
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict of the quote."""
        return {
            "client_id": self.client_id,
            "measurement": self.measurement,
            "config_hash": self.config_hash,
            "signature": self.signature,
            "nonce": self.nonce,
            "issued_at": self.issued_at,
            "attestor_id": self.attestor_id,
        }


@dataclass(frozen=True)
class AttestationResult:
    """Structured outcome of verifying one quote."""

    ok: bool
    reason_code: str
    detail: str = ""

    def raise_if_failed(self) -> None:
        """Raise :class:`AttestationError` when the quote was not accepted."""
        if not self.ok:
            raise AttestationError(f"[{self.reason_code}] {self.detail}")

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict of the result."""
        return {"ok": self.ok, "reason_code": self.reason_code, "detail": self.detail}


class Attestor(ABC):
    """Interface for producing and verifying attestation quotes.

    A backend implements :meth:`generate_quote` and :meth:`verify`; overriding
    :meth:`check` as well lets callers see *why* a quote was refused. Backends
    are expected to be stateless apart from their key material and any
    challenge store.
    """

    @abstractmethod
    def generate_quote(
        self,
        client_id: str,
        code_identity: str,
        config: str,
        *,
        nonce: str = "",
    ) -> Quote:
        """Produce a quote for ``client_id`` running ``code_identity``."""

    @abstractmethod
    def verify(self, quote: Quote, **kwargs: Any) -> bool:
        """Return ``True`` iff the quote is acceptable under this backend."""

    def check(self, quote: Quote, **kwargs: Any) -> AttestationResult:
        """Return a structured verdict. Default wraps :meth:`verify`."""
        ok = self.verify(quote, **kwargs)
        return AttestationResult(
            ok=ok,
            reason_code=REASON_OK if ok else "rejected",
            detail="" if ok else "backend gave no reason code",
        )

    def verify_or_raise(self, quote: Quote, **kwargs: Any) -> Quote:
        """Return ``quote`` if acceptable, else raise :class:`AttestationError`."""
        self.check(quote, **kwargs).raise_if_failed()
        return quote

    def issue_nonce(self, client_id: str) -> str:
        """Return a challenge nonce for ``client_id``.

        Backends without a challenge store raise :class:`PolicyError`.
        """
        raise PolicyError(f"{type(self).__name__} does not issue nonces")


class MockSoftwareAttestor(Attestor):
    """Software mock of a TEE attestation service.

    Parameters
    ----------
    root_key:
        Shared secret standing in for the hardware root of trust. Used to build
        the default HMAC backend; ignored when ``signer`` is given.
    approved_measurements:
        Convenience form of the policy allow-list, kept for the original API.
        Mutually exclusive with ``policy``.
    policy:
        Full :class:`~trustfed.attestation.policy.AttestationPolicy`, including
        freshness and nonce requirements.
    signer:
        Pluggable signing backend (:class:`trustfed.ledger.crypto.Signer`). Pass
        an ``Ed25519Signer`` so that verifiers holding only the public key
        cannot forge quotes.
    verifier:
        Verification key. Defaults to ``signer.verifier()``.
    nonce_store:
        Challenge store used by :meth:`issue_nonce` and replay detection.
        **Required** when ``policy.require_nonce`` is True: a verifier that
        cannot issue challenges cannot tell a fresh nonce from one the prover
        invented, so that combination raises
        :class:`~trustfed.attestation.errors.PolicyError`.
    clock:
        Unix-seconds clock, injectable for deterministic tests.

    SECURITY NOTE: this class does not measure code. It signs whatever identity
    the caller declares. It exists to exercise the accept/reject logic that a
    real attestation backend would drive.
    """

    def __init__(
        self,
        root_key: Optional[bytes] = None,
        approved_measurements: Optional[Iterable[str]] = None,
        *,
        policy: Optional[AttestationPolicy] = None,
        signer: Optional[Signer] = None,
        verifier: Optional[Verifier] = None,
        nonce_store: Optional[NonceStore] = None,
        clock: Optional[Callable[[], float]] = None,
        attestor_id: str = "mock-software-attestor",
    ) -> None:
        if policy is not None and approved_measurements is not None:
            raise PolicyError("pass either approved_measurements or policy, not both")
        if policy is None:
            policy = AttestationPolicy(
                approved_measurements=frozenset(approved_measurements or ())
            )
        if signer is None:
            if root_key is None:
                raise PolicyError("provide root_key or an explicit signer")
            signer = HmacSigner(root_key, key_label=f"mock-root:{attestor_id}")
        if policy.require_nonce and nonce_store is None:
            raise PolicyError(
                "policy.require_nonce=True needs a NonceStore: without one this "
                "verifier can only check that *some* nonce is present -- one "
                "the prover picked -- so every quote stays replayable"
            )
        self._policy = policy
        self._signer = signer
        self._verifier = verifier if verifier is not None else signer.verifier()
        self._nonces = nonce_store
        self._clock: Callable[[], float] = clock if clock is not None else time.time
        self._attestor_id = attestor_id
        # Retained for backwards compatibility with the original attribute.
        self._root_key = root_key
        self._approved = set(policy.approved_measurements)

    # ------------------------------------------------------------------ helpers

    @property
    def policy(self) -> AttestationPolicy:
        """The policy this verifier enforces (immutable)."""
        return self._policy

    def set_policy(self, policy: AttestationPolicy) -> None:
        """Replace the policy, e.g. to approve a new code version or revoke one.

        Raises :class:`~trustfed.attestation.errors.PolicyError` if the new
        policy requires a nonce and this attestor has no challenge store.
        """
        if not isinstance(policy, AttestationPolicy):
            raise PolicyError("policy must be an AttestationPolicy")
        if policy.require_nonce and self._nonces is None:
            raise PolicyError(
                "cannot switch to a require_nonce policy: this attestor has no "
                "NonceStore, so it can neither issue nor spend challenges"
            )
        self._policy = policy
        self._approved = set(policy.approved_measurements)

    @staticmethod
    def config_hash(config: str) -> str:
        """Return the hash of a client configuration string."""
        return hashlib.sha256(config.encode("utf-8")).hexdigest()

    def issue_nonce(self, client_id: str) -> str:
        """Return a fresh single-use challenge for ``client_id``."""
        if self._nonces is None:
            raise PolicyError("this attestor was built without a NonceStore")
        return self._nonces.issue(client_id)

    # ------------------------------------------------------------------- quoting

    def generate_quote(
        self,
        client_id: str,
        code_identity: str,
        config: str,
        *,
        nonce: str = "",
    ) -> Quote:
        """Sign a quote over the declared identity, config, nonce and time."""
        unsigned = Quote(
            client_id=client_id,
            measurement=measure_code(code_identity),
            config_hash=self.config_hash(config),
            signature="",
            nonce=nonce,
            issued_at=float(self._clock()),
            attestor_id=self._attestor_id,
        )
        return replace(unsigned, signature=self._signer.sign(unsigned.signing_material()))

    # -------------------------------------------------------------- verification

    def check(
        self,
        quote: Quote,
        *,
        expected_nonce: Optional[str] = None,
        now: Optional[float] = None,
        **_: Any,
    ) -> AttestationResult:
        """Verify signature, freshness, nonce binding and allow-lists.

        Order matters, and it is this:

        1. the signature, so that no policy decision is made on unauthenticated
           fields;
        2. the timestamp -- both directions. A quote dated in the future is
           rejected whenever it carries a timestamp at all, independently of
           ``max_age_seconds``: "is this quote stale?" and "is this clock
           lying?" are different questions and a policy that declines to set a
           maximum age has not thereby agreed to accept quotes from next year;
        3. the nonce *binding* (present, and answering the expected challenge);
        4. the measurement and config allow-lists;
        5. only then, spending the single-use challenge.

        Consuming the challenge last is deliberate. A client whose code version
        was just revoked would otherwise burn a fresh challenge on every
        rejected attempt, and an attacker who observed a victim's outstanding
        challenge could spend it by presenting it with junk. A challenge is
        only consumed by a quote that is acceptable in every other respect.
        """
        if not self._verifier.verify(quote.signing_material(), quote.signature):
            return AttestationResult(
                False, REASON_BAD_SIGNATURE, "quote signature does not verify"
            )

        policy = self._policy
        current = float(now) if now is not None else float(self._clock())
        age = current - quote.issued_at

        if quote.issued_at > 0 and age < -policy.clock_skew_seconds:
            return AttestationResult(
                False,
                REASON_QUOTE_FROM_FUTURE,
                f"quote is timestamped {-age:.1f}s in the future, beyond the "
                f"{policy.clock_skew_seconds:.1f}s skew allowance",
            )
        if policy.max_age_seconds is not None and age > policy.max_age_seconds:
            return AttestationResult(
                False,
                REASON_QUOTE_EXPIRED,
                f"quote is {age:.1f}s old, policy allows "
                f"{policy.max_age_seconds:.1f}s",
            )

        nonce_required = policy.require_nonce or expected_nonce is not None
        if nonce_required:
            if expected_nonce is None and self._nonces is None:
                # Fail closed: accepting a prover-chosen nonce would make the
                # quote a bearer token again. The constructor refuses this
                # combination, so reaching it means the store was removed.
                return AttestationResult(
                    False,
                    REASON_NONCE_REJECTED,
                    "this verifier has no challenge store, so freshness cannot "
                    "be checked",
                )
            if not quote.nonce:
                return AttestationResult(
                    False, REASON_MISSING_NONCE, "policy requires a challenge nonce"
                )
            if expected_nonce is not None and quote.nonce != expected_nonce:
                return AttestationResult(
                    False, REASON_NONCE_MISMATCH, "quote answers a different challenge"
                )

        if not policy.approves_measurement(quote.measurement):
            return AttestationResult(
                False,
                REASON_UNAPPROVED_MEASUREMENT,
                "code measurement is not on the approved allow-list",
            )
        if not policy.approves_config(quote.config_hash):
            return AttestationResult(
                False,
                REASON_UNAPPROVED_CONFIG,
                "configuration hash is not on the approved allow-list",
            )

        if nonce_required and self._nonces is not None:
            try:
                self._nonces.consume(quote.client_id, quote.nonce)
            except NonceError as exc:
                return AttestationResult(False, REASON_NONCE_REJECTED, str(exc))
        return AttestationResult(True, REASON_OK, "quote accepted")

    def verify(self, quote: Quote, **kwargs: Any) -> bool:
        """Return ``True`` iff :meth:`check` accepts the quote."""
        return self.check(quote, **kwargs).ok


__all__ = [
    "AttestationError",
    "AttestationResult",
    "Attestor",
    "MockSoftwareAttestor",
    "Quote",
    "measure_code",
]
