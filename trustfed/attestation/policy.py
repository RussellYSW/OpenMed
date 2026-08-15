"""Verifier-side attestation policy and the challenge-nonce store.

The policy is the object a relying party actually configures: which code
measurements it will accept, whether a quote must answer a fresh challenge, and
how old a quote may be. Separating it from the attestor means the same policy
can be evaluated against quotes from a mock attestor today and a real TEE
backend later.

Freshness matters because a quote without a challenge is a bearer token: an
attacker who observes one valid quote can replay it forever. :class:`NonceStore`
issues single-use challenges and refuses the second presentation of one.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, Iterable, Optional

from trustfed.attestation.errors import NonceError, PolicyError


@dataclass(frozen=True)
class AttestationPolicy:
    """What a verifier will accept from a client.

    Parameters
    ----------
    approved_measurements:
        Allow-list of code measurements (see
        :func:`trustfed.attestation.attestor.measure_code`). Empty means the
        verifier accepts nothing, which is deliberate: fail closed.
    approved_config_hashes:
        Optional allow-list of configuration hashes. ``None`` means any
        configuration is acceptable.
    require_nonce:
        Require every quote to carry a challenge nonce issued by this verifier.
        Defaults to ``False`` so the pre-existing two-argument API keeps
        working; turn it on for anything where replay matters.
    max_age_seconds:
        Reject quotes older than this many seconds. ``None`` disables the
        staleness check only; a quote dated in the future is still rejected,
        because ``None`` means "I do not require freshness", not "I accept any
        timestamp whatsoever".
    clock_skew_seconds:
        Tolerance for a quote timestamped slightly in the future. Beyond this,
        a future-dated quote is refused with ``quote_from_future`` regardless
        of ``max_age_seconds``. A quote with ``issued_at == 0`` carries no
        timestamp at all (the original two-field protocol) and is exempt.
    """

    approved_measurements: FrozenSet[str] = frozenset()
    approved_config_hashes: Optional[FrozenSet[str]] = None
    require_nonce: bool = False
    max_age_seconds: Optional[float] = None
    clock_skew_seconds: float = 5.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "approved_measurements", frozenset(self.approved_measurements)
        )
        if self.approved_config_hashes is not None:
            object.__setattr__(
                self, "approved_config_hashes", frozenset(self.approved_config_hashes)
            )
        if self.max_age_seconds is not None and self.max_age_seconds <= 0:
            raise PolicyError("max_age_seconds must be positive or None")
        if self.clock_skew_seconds < 0:
            raise PolicyError("clock_skew_seconds must be >= 0")

    @classmethod
    def from_code_identities(
        cls, identities: Iterable[str], **kwargs: object
    ) -> "AttestationPolicy":
        """Build a policy from declared code-identity strings.

        Convenience wrapper that measures each identity for you.
        """
        from trustfed.attestation.attestor import measure_code

        measurements = frozenset(measure_code(i) for i in identities)
        return cls(approved_measurements=measurements, **kwargs)  # type: ignore[arg-type]

    def approves_measurement(self, measurement: str) -> bool:
        """Return ``True`` iff ``measurement`` is on the allow-list."""
        return measurement in self.approved_measurements

    def approves_config(self, config_hash: str) -> bool:
        """Return ``True`` iff ``config_hash`` is acceptable under this policy."""
        if self.approved_config_hashes is None:
            return True
        return config_hash in self.approved_config_hashes

    def with_measurements(self, *measurements: str) -> "AttestationPolicy":
        """Return a copy with additional approved measurements.

        Policies are immutable; upgrading the approved code version produces a
        new policy object rather than mutating one other components may hold.
        """
        return AttestationPolicy(
            approved_measurements=self.approved_measurements.union(measurements),
            approved_config_hashes=self.approved_config_hashes,
            require_nonce=self.require_nonce,
            max_age_seconds=self.max_age_seconds,
            clock_skew_seconds=self.clock_skew_seconds,
        )

    def without_measurements(self, *measurements: str) -> "AttestationPolicy":
        """Return a copy with the given measurements revoked."""
        return AttestationPolicy(
            approved_measurements=self.approved_measurements.difference(measurements),
            approved_config_hashes=self.approved_config_hashes,
            require_nonce=self.require_nonce,
            max_age_seconds=self.max_age_seconds,
            clock_skew_seconds=self.clock_skew_seconds,
        )


@dataclass
class _Challenge:
    client_id: str
    issued_at: float
    spent: bool = False


class NonceStore:
    """Issues single-use attestation challenges and detects replay.

    Assumes a single verifier process: challenges live in memory only, so a
    multi-process verifier needs a shared store instead. Expired challenges are
    pruned lazily on issue.

    SECURITY NOTE: this stops replay of a captured quote. It does not stop a
    client that is *currently* compromised from obtaining a fresh valid quote --
    only a real TEE measurement can address that, and the mock attestor cannot.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = 300.0,
        clock: Optional[Callable[[], float]] = None,
        nonce_factory: Optional[Callable[[], str]] = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise PolicyError("ttl_seconds must be positive")
        self._ttl = ttl_seconds
        self._clock: Callable[[], float] = clock if clock is not None else time.time
        self._factory: Callable[[], str] = (
            nonce_factory if nonce_factory is not None else (lambda: secrets.token_hex(16))
        )
        self._challenges: Dict[str, _Challenge] = {}

    def issue(self, client_id: str) -> str:
        """Return a fresh single-use challenge nonce bound to ``client_id``."""
        self._prune()
        nonce = self._factory()
        if nonce in self._challenges:
            raise NonceError("nonce factory produced a collision")
        self._challenges[nonce] = _Challenge(client_id, self._clock())
        return nonce

    def consume(self, client_id: str, nonce: str) -> None:
        """Spend ``nonce`` for ``client_id``.

        Raises
        ------
        NonceError
            If the nonce is unknown, bound to a different client, expired, or
            has already been spent (a replay).
        """
        challenge = self._challenges.get(nonce)
        if challenge is None:
            raise NonceError("unknown or already-pruned nonce")
        if challenge.client_id != client_id:
            raise NonceError("nonce was issued to a different client")
        now = self._clock()
        if now - challenge.issued_at > self._ttl:
            del self._challenges[nonce]
            raise NonceError("nonce has expired")
        if challenge.spent:
            raise NonceError("nonce has already been used (replay)")
        challenge.spent = True

    def is_outstanding(self, nonce: str) -> bool:
        """Return ``True`` iff ``nonce`` exists and has not been spent."""
        challenge = self._challenges.get(nonce)
        return challenge is not None and not challenge.spent

    def _prune(self) -> None:
        cutoff = self._clock() - self._ttl
        for nonce in [
            n for n, c in self._challenges.items() if c.issued_at < cutoff
        ]:
            del self._challenges[nonce]

    def __len__(self) -> int:
        """Return the number of challenges currently tracked."""
        return len(self._challenges)


__all__ = ["AttestationPolicy", "NonceStore"]
