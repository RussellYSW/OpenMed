"""Remote-attestation abstraction for federated clients.

The idea mirrors hardware TEE attestation (Intel SGX/TDX, AMD SEV-SNP): before a
client's update is accepted, it must present a *quote* proving that
(1) it is running an approved code measurement, and (2) the quote was produced
by genuine attestation hardware rooted in a key the verifier trusts.

This reference implementation is a **software mock**: the "measurement" is a hash
of the client's declared code identity, and the "quote" is an HMAC under a shared
root key that stands in for the TEE vendor's root of trust. It is enough to
demonstrate the protocol and to unit-test the accept/reject logic. To use a real
TEE, implement the :class:`Attestor` interface against your platform's quoting
and verification APIs (e.g. DCAP for SGX/TDX) -- the rest of TrustFed is
unchanged.

SECURITY NOTE: the mock is NOT a real security boundary. Do not rely on it to
protect production federations.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Iterable


class AttestationError(Exception):
    """Raised when a quote cannot be verified."""


def measure_code(identity: str) -> str:
    """Return a stand-in code *measurement* (like an SGX MRENCLAVE).

    In a real TEE this is a hash of the loaded enclave image. Here we hash a
    declared code-identity string so tests can simulate approved vs. tampered
    code.
    """
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Quote:
    """An attestation quote presented by a client alongside its update."""

    client_id: str
    measurement: str
    config_hash: str
    signature: str


class Attestor:
    """Interface for producing and verifying attestation quotes."""

    def generate_quote(self, client_id: str, code_identity: str, config: str) -> Quote:
        raise NotImplementedError

    def verify(self, quote: Quote) -> bool:
        raise NotImplementedError


class MockSoftwareAttestor(Attestor):
    """Software mock of a TEE attestation service.

    Parameters
    ----------
    root_key:
        Shared secret standing in for the hardware root of trust.
    approved_measurements:
        The set of code measurements the verifier will accept. A client whose
        measurement is not in this allow-list is rejected (models code
        tampering / an unapproved client).
    """

    def __init__(self, root_key: bytes, approved_measurements: Iterable[str]):
        self._root_key = root_key
        self._approved = set(approved_measurements)

    @staticmethod
    def config_hash(config: str) -> str:
        return hashlib.sha256(config.encode("utf-8")).hexdigest()

    def _sign(self, client_id: str, measurement: str, config_hash: str) -> str:
        msg = f"{client_id}|{measurement}|{config_hash}".encode("utf-8")
        return hmac.new(self._root_key, msg, hashlib.sha256).hexdigest()

    def generate_quote(self, client_id: str, code_identity: str, config: str) -> Quote:
        measurement = measure_code(code_identity)
        config_hash = self.config_hash(config)
        signature = self._sign(client_id, measurement, config_hash)
        return Quote(client_id, measurement, config_hash, signature)

    def verify(self, quote: Quote) -> bool:
        expected_sig = self._sign(
            quote.client_id, quote.measurement, quote.config_hash
        )
        # Constant-time comparison of the signature, then check the code
        # measurement against the approved allow-list.
        if not hmac.compare_digest(expected_sig, quote.signature):
            return False
        return quote.measurement in self._approved
