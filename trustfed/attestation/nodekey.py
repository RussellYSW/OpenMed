"""Attestation signed by a registered per-client key over a software measurement.

This is the software counterpart of a TEE quote for deployments without
attestation hardware. The protocol is the same as :class:`MockSoftwareAttestor`
enforces -- challenge nonce, freshness, measurement allow-list -- with one
difference that matters: the quote is signed **at the client** with a key the
verifier does not hold, and the verifier checks it against the key registered
for that client (for OpenMed, the node key that completed the site's
verification handshake).

Trust model, stated plainly:

* The verifier cannot forge a client's quote, so an accepted quote proves that
  the holder of that client's key signed this measurement for this challenge.
* The measurement is computed by software on the client
  (:func:`trustfed.attestation.measure.software_measurement`), so a client
  whose *operator* is dishonest can sign a false measurement. This backend
  authenticates the site, not the hardware. A TEE backend replaces the signing
  key with a hardware root of trust and otherwise fits the same interface.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional, Tuple

from trustfed.attestation.attestor import (
    REASON_BAD_SIGNATURE,
    REASON_MISSING_NONCE,
    REASON_NONCE_MISMATCH,
    REASON_NONCE_REJECTED,
    REASON_OK,
    REASON_QUOTE_EXPIRED,
    REASON_QUOTE_FROM_FUTURE,
    REASON_UNAPPROVED_CONFIG,
    REASON_UNAPPROVED_MEASUREMENT,
    AttestationResult,
    Attestor,
    Quote,
)
from trustfed.attestation.errors import AttestationError, NonceError, PolicyError
from trustfed.attestation.policy import AttestationPolicy, NonceStore
from trustfed.ledger.crypto import Signer, Verifier

REASON_UNKNOWN_CLIENT = "unknown_client"
REASON_WRONG_ATTESTOR = "wrong_attestor"

#: ``attestor_id`` carried by every quote of this scheme.
NODE_KEY_ATTESTOR_ID = "node-key-software-measurement"


class NodeKeyAttestor(Attestor):
    """Verify quotes signed by registered client keys.

    Parameters
    ----------
    policy:
        Allow-list and freshness rules. ``approved_measurements`` should hold
        digests from :func:`~trustfed.attestation.measure.software_measurement`.
    nonce_store:
        Challenge store; required when ``policy.require_nonce`` is set.
    clock:
        Unix-seconds clock, injectable for tests.
    """

    def __init__(
        self,
        *,
        policy: AttestationPolicy,
        nonce_store: Optional[NonceStore] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        if policy.require_nonce and nonce_store is None:
            raise PolicyError("policy.require_nonce=True needs a NonceStore")
        self._policy = policy
        self._nonces = nonce_store
        self._clock: Callable[[], float] = clock if clock is not None else time.time
        self._keys: Dict[str, Verifier] = {}

    # ------------------------------------------------------------ registry

    def register_client(self, client_id: str, verifier: Verifier) -> None:
        """Bind ``client_id`` to the key its quotes must verify under.

        Re-registering replaces the key, which is how rotation is modelled.
        """
        if not client_id:
            raise PolicyError("client_id must not be empty")
        self._keys[client_id] = verifier

    def revoke_client(self, client_id: str) -> None:
        self._keys.pop(client_id, None)

    def clients(self) -> Tuple[str, ...]:
        return tuple(sorted(self._keys))

    def __contains__(self, client_id: object) -> bool:
        return client_id in self._keys

    @property
    def policy(self) -> AttestationPolicy:
        return self._policy

    def set_policy(self, policy: AttestationPolicy) -> None:
        if policy.require_nonce and self._nonces is None:
            raise PolicyError("cannot require a nonce without a NonceStore")
        self._policy = policy

    def issue_nonce(self, client_id: str) -> str:
        if self._nonces is None:
            raise PolicyError("this attestor was built without a NonceStore")
        return self._nonces.issue(client_id)

    # ------------------------------------------------------------- quoting

    def generate_quote(self, client_id: str, code_identity: str, config: str, *, nonce: str = "") -> Quote:
        """Refused: the verifier holds no client keys. Clients call :meth:`sign_quote`."""
        raise AttestationError(
            "a NodeKeyAttestor verifies quotes; the client produces them with "
            "NodeKeyAttestor.sign_quote() and its own key"
        )

    @staticmethod
    def sign_quote(
        signer: Signer,
        *,
        client_id: str,
        measurement: str,
        config_hash: str,
        nonce: str,
        issued_at: Optional[float] = None,
    ) -> Quote:
        """Client side: sign a quote over the measurement and the challenge."""
        unsigned = Quote(
            client_id=client_id,
            measurement=measurement,
            config_hash=config_hash,
            signature="",
            nonce=nonce,
            issued_at=float(issued_at if issued_at is not None else time.time()),
            attestor_id=NODE_KEY_ATTESTOR_ID,
        )
        return Quote(
            client_id=unsigned.client_id,
            measurement=unsigned.measurement,
            config_hash=unsigned.config_hash,
            signature=signer.sign(unsigned.signing_material()),
            nonce=unsigned.nonce,
            issued_at=unsigned.issued_at,
            attestor_id=unsigned.attestor_id,
        )

    # ---------------------------------------------------------------- check

    def check(
        self,
        quote: Quote,
        *,
        expected_nonce: Optional[str] = None,
        now: Optional[float] = None,
        **_: Any,
    ) -> AttestationResult:
        """Same order as the mock: signature, clock, nonce binding, allow-lists, spend."""
        if quote.attestor_id != NODE_KEY_ATTESTOR_ID:
            return AttestationResult(
                False, REASON_WRONG_ATTESTOR, f"quote was not produced for this scheme ({quote.attestor_id!r})"
            )
        verifier = self._keys.get(quote.client_id)
        if verifier is None:
            return AttestationResult(False, REASON_UNKNOWN_CLIENT, f"no key registered for client {quote.client_id!r}")
        if not quote.signature or not verifier.verify(quote.signing_material(), quote.signature):
            return AttestationResult(False, REASON_BAD_SIGNATURE, "quote signature does not verify under the client's registered key")
        policy = self._policy
        current = float(now) if now is not None else float(self._clock())
        age = current - quote.issued_at
        if quote.issued_at > 0 and age < -policy.clock_skew_seconds:
            return AttestationResult(False, REASON_QUOTE_FROM_FUTURE, f"quote is timestamped {-age:.1f}s in the future")
        if policy.max_age_seconds is not None and age > policy.max_age_seconds:
            return AttestationResult(False, REASON_QUOTE_EXPIRED, f"quote is {age:.1f}s old, policy allows {policy.max_age_seconds:.1f}s")
        nonce_required = policy.require_nonce or expected_nonce is not None
        if nonce_required:
            if expected_nonce is None and self._nonces is None:
                return AttestationResult(False, REASON_NONCE_REJECTED, "no challenge store; freshness cannot be checked")
            if not quote.nonce:
                return AttestationResult(False, REASON_MISSING_NONCE, "policy requires a challenge nonce")
            if expected_nonce is not None and quote.nonce != expected_nonce:
                return AttestationResult(False, REASON_NONCE_MISMATCH, "quote answers a different challenge")
        if not policy.approves_measurement(quote.measurement):
            return AttestationResult(False, REASON_UNAPPROVED_MEASUREMENT, "code measurement is not on the approved allow-list")
        if not policy.approves_config(quote.config_hash):
            return AttestationResult(False, REASON_UNAPPROVED_CONFIG, "configuration hash is not on the approved allow-list")
        if nonce_required and self._nonces is not None:
            try:
                self._nonces.consume(quote.client_id, quote.nonce)
            except NonceError as exc:
                return AttestationResult(False, REASON_NONCE_REJECTED, str(exc))
        return AttestationResult(True, REASON_OK, "quote accepted")

    def verify(self, quote: Quote, **kwargs: Any) -> bool:
        return self.check(quote, **kwargs).ok


__all__ = ["NodeKeyAttestor", "NODE_KEY_ATTESTOR_ID", "REASON_UNKNOWN_CLIENT", "REASON_WRONG_ATTESTOR"]
