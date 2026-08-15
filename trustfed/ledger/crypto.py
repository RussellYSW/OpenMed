"""Signing primitives for the ledger, with a plainly-marked HMAC fallback.

Ed25519 (via the optional ``cryptography`` package) is used when it is
importable. When it is not, the package falls back to an HMAC construction so
that TrustFed still runs on a bare ``numpy`` install.

SECURITY NOTE: the HMAC fallback is **not** a security boundary. It is a
symmetric MAC: anyone who can verify a block can also forge one, because
verification requires the same secret used for signing. It provides tamper
*evidence* against a party that does not hold the key and nothing more. Use
:class:`Ed25519Signer` (or a real HSM/KMS backend behind :class:`Signer`) for
anything where the verifier and the signer are different principals.

This module also defines the canonical JSON encoding used for hashing, so that
two processes hashing the same record always agree.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from abc import ABC, abstractmethod
from typing import Any, Optional

from trustfed.ledger.errors import LedgerStorageError, SignatureError

try:  # pragma: no cover - exercised implicitly by both branches
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        PublicFormat,
    )

    HAVE_CRYPTOGRAPHY = True
except ImportError:  # pragma: no cover - depends on the environment
    HAVE_CRYPTOGRAPHY = False


def _reject_non_json(value: Any) -> Any:
    """Raise rather than coerce a value JSON cannot represent natively.

    Coercing (the old ``default=str``) made the encoding non-injective:
    ``Path("/a")`` and the string ``"/a"`` produced identical bytes, so two
    different payloads committed to the same hash and a payload could change
    Python type across a storage round trip while still verifying.
    """
    raise LedgerStorageError(
        f"payload value of type {type(value).__name__!r} is not JSON-native; "
        "serialise it before appending (e.g. str(path), float(numpy_scalar))"
    )


def canonical_json(obj: Any) -> bytes:
    """Serialise ``obj`` to the canonical byte form used for all hashing.

    Keys are sorted and separators are tight, so the encoding is stable across
    processes and Python versions.

    Raises
    ------
    LedgerStorageError
        If ``obj`` contains a value JSON cannot represent natively. The encoder
        never silently coerces, because a lossy encoder in a content-commitment
        path lets two distinct payloads share one hash.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_reject_non_json,
    ).encode("utf-8")


def hash_payload(payload: Any) -> str:
    """Return the hex SHA-256 of ``payload`` under :func:`canonical_json`."""
    return hashlib.sha256(canonical_json(payload)).hexdigest()


class Verifier(ABC):
    """Something that can check a signature over a message.

    Implementations must be side-effect free and must not raise on a bad
    signature -- they return ``False``.
    """

    #: Short algorithm label recorded in each block (e.g. ``"ed25519"``).
    algorithm: str = "abstract"

    @property
    @abstractmethod
    def key_id(self) -> str:
        """Stable public identifier of the key (hex; safe to publish)."""

    @abstractmethod
    def verify(self, message: bytes, signature: str) -> bool:
        """Return ``True`` iff ``signature`` (hex) is valid for ``message``."""


class Signer(Verifier):
    """A :class:`Verifier` that can also produce signatures."""

    @abstractmethod
    def sign(self, message: bytes) -> str:
        """Return a hex-encoded signature over ``message``."""

    def verifier(self) -> Verifier:
        """Return a verify-only handle for this key.

        For symmetric backends this returns ``self``, which is exactly why the
        symmetric backend is not a security boundary.
        """
        return self


class HmacSigner(Signer):
    """Symmetric HMAC-SHA256 signer -- the dependency-free fallback.

    SECURITY NOTE: not a security boundary. The verifying key *is* the signing
    key, so any verifier can forge blocks. Detects accidental corruption and
    tampering by a party without the key; nothing else.
    """

    algorithm = "hmac-sha256"

    def __init__(self, key: bytes, *, key_label: Optional[str] = None) -> None:
        if not isinstance(key, (bytes, bytearray)) or len(key) == 0:
            raise SignatureError("HMAC key must be non-empty bytes")
        self._key = bytes(key)
        self._label = key_label

    @property
    def key_id(self) -> str:
        """Hash of the secret (not the secret), so it is safe to record."""
        if self._label is not None:
            return self._label
        return "hmac:" + hashlib.sha256(self._key).hexdigest()[:32]

    def sign(self, message: bytes) -> str:
        """Return the hex HMAC-SHA256 tag over ``message``."""
        return hmac.new(self._key, message, hashlib.sha256).hexdigest()

    def verify(self, message: bytes, signature: str) -> bool:
        """Return ``True`` iff the tag matches, compared in constant time."""
        if not isinstance(signature, str):
            return False
        return hmac.compare_digest(self.sign(message), signature)


class Ed25519Verifier(Verifier):
    """Ed25519 public-key verifier. Requires the ``cryptography`` package."""

    algorithm = "ed25519"

    def __init__(self, public_key: "Ed25519PublicKey") -> None:
        if not HAVE_CRYPTOGRAPHY:  # pragma: no cover - guarded by caller
            raise SignatureError("cryptography is not installed")
        self._public = public_key

    @classmethod
    def from_public_hex(cls, public_hex: str) -> "Ed25519Verifier":
        """Build a verifier from a 32-byte hex-encoded public key."""
        if not HAVE_CRYPTOGRAPHY:
            raise SignatureError("cryptography is not installed")
        try:
            raw = bytes.fromhex(public_hex)
        except ValueError as exc:
            raise SignatureError(f"malformed public key hex: {exc}") from exc
        return cls(Ed25519PublicKey.from_public_bytes(raw))

    @property
    def key_id(self) -> str:
        """Return the raw 32-byte public key, hex-encoded."""
        raw = self._public.public_bytes(Encoding.Raw, PublicFormat.Raw)
        return raw.hex()

    def verify(self, message: bytes, signature: str) -> bool:
        """Return ``True`` iff the Ed25519 signature is valid for ``message``."""
        if not isinstance(signature, str):
            return False
        try:
            self._public.verify(bytes.fromhex(signature), message)
        except (InvalidSignature, ValueError):
            return False
        return True


class Ed25519Signer(Signer):
    """Ed25519 signer. Requires the optional ``cryptography`` package.

    Asymmetric, so a verifier holding only :attr:`key_id` cannot forge blocks.
    Key material is held in process memory only -- this class does not implement
    HSM custody, key rotation, or revocation.
    """

    algorithm = "ed25519"

    def __init__(self, private_key: "Ed25519PrivateKey") -> None:
        if not HAVE_CRYPTOGRAPHY:  # pragma: no cover - guarded by caller
            raise SignatureError("cryptography is not installed")
        self._private = private_key

    @classmethod
    def generate(cls) -> "Ed25519Signer":
        """Generate a fresh keypair from the OS CSPRNG."""
        if not HAVE_CRYPTOGRAPHY:
            raise SignatureError("cryptography is not installed")
        return cls(Ed25519PrivateKey.generate())

    @classmethod
    def from_seed(cls, seed: bytes) -> "Ed25519Signer":
        """Derive a *deterministic* key from ``seed`` (tests and demos only).

        SECURITY NOTE: a seed committed to a repository is a published private
        key. Use :meth:`generate` for anything real.
        """
        if not HAVE_CRYPTOGRAPHY:
            raise SignatureError("cryptography is not installed")
        raw = hashlib.sha256(seed).digest()
        return cls(Ed25519PrivateKey.from_private_bytes(raw))

    @property
    def key_id(self) -> str:
        """Return the corresponding public key, hex-encoded."""
        return self.verifier().key_id

    def sign(self, message: bytes) -> str:
        """Return a hex Ed25519 signature over ``message``."""
        return self._private.sign(message).hex()

    def verify(self, message: bytes, signature: str) -> bool:
        """Return ``True`` iff the signature verifies under the public key."""
        return self.verifier().verify(message, signature)

    def verifier(self) -> Verifier:
        """Return a public-key-only verifier that cannot sign."""
        return Ed25519Verifier(self._private.public_key())


def default_signer(seed: bytes = b"trustfed-default-ledger-key") -> Signer:
    """Return the best deterministic signer available in this environment.

    Ed25519 when ``cryptography`` is importable, otherwise the HMAC fallback.
    Deterministic in ``seed`` so demos and tests reproduce exactly; see the
    security notes on both classes before using this outside a demo.
    """
    if HAVE_CRYPTOGRAPHY:
        return Ed25519Signer.from_seed(seed)
    return HmacSigner(hashlib.sha256(seed).digest())


__all__ = [
    "HAVE_CRYPTOGRAPHY",
    "Ed25519Signer",
    "Ed25519Verifier",
    "HmacSigner",
    "Signer",
    "Verifier",
    "canonical_json",
    "default_signer",
    "hash_payload",
]
