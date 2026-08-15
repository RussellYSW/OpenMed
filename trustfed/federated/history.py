"""Per-client behavioural history used by the selection policies.

The history is the memory that makes reputation-weighted selection possible: it
folds each round's observations (was the update accepted? how far did it sit
from the robust aggregate? what local loss did the site report?) into a running
per-client record.

Reputation here is *behavioural evidence*, not proof. Honest sites with unusual
case-mix drift down; a patient attacker that behaves for many rounds drifts back
up. It reduces an attacker's expected exposure, it does not identify attackers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from trustfed.federated.errors import SelectionError


@dataclass
class ClientRecord:
    """Running per-client statistics used by reputation and loss selectors."""

    client_id: str
    rounds_selected: int = 0
    rounds_accepted: int = 0
    rounds_rejected: int = 0
    last_loss: Optional[float] = None
    mean_loss: Optional[float] = None
    last_deviation: Optional[float] = None
    mean_deviation: Optional[float] = None
    reputation: float = 1.0

    def to_dict(self) -> Dict[str, object]:
        """JSON-serializable view of this record."""
        return {
            "client_id": self.client_id,
            "rounds_selected": self.rounds_selected,
            "rounds_accepted": self.rounds_accepted,
            "rounds_rejected": self.rounds_rejected,
            "last_loss": self.last_loss,
            "mean_loss": self.mean_loss,
            "last_deviation": self.last_deviation,
            "mean_deviation": self.mean_deviation,
            "reputation": round(self.reputation, 6),
        }


@dataclass(frozen=True)
class RoundObservation:
    """What the server observed about one client in one round."""

    client_id: str
    accepted: bool
    loss: Optional[float] = None
    deviation: Optional[float] = None


@dataclass
class ClientHistory:
    """Per-client history consumed by the selectors.

    Parameters
    ----------
    alpha:
        EMA weight for reputation updates; ``1.0`` means "only the last round
        matters", small values mean reputation moves slowly (and an attacker
        needs many clean rounds to recover).
    rejection_penalty:
        Reputation score assigned to a round in which the client's update was
        rejected (e.g. failed attestation). ``0.0`` is maximum penalty.

    Notes
    -----
    Reputation is behavioural, not cryptographic: it is evidence about
    *consistency with the consensus*, and honest sites with genuinely unusual
    (non-IID) data will lose some reputation too. Do not treat a low score as
    proof of compromise.
    """

    alpha: float = 0.3
    rejection_penalty: float = 0.0
    records: Dict[str, ClientRecord] = field(default_factory=dict)
    rounds_observed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise SelectionError(f"alpha must be in (0, 1], got {self.alpha}")
        if not 0.0 <= self.rejection_penalty <= 1.0:
            raise SelectionError(
                f"rejection_penalty must be in [0, 1], got {self.rejection_penalty}"
            )

    def record(self, client_id: str) -> ClientRecord:
        """Return (creating if needed) the record for ``client_id``."""
        rec = self.records.get(client_id)
        if rec is None:
            rec = ClientRecord(client_id=client_id)
            self.records[client_id] = rec
        return rec

    def reputation(self, client_id: str) -> float:
        """Current reputation of a client; unseen clients start at 1.0."""
        return self.record(client_id).reputation

    def observe_round(self, observations: Sequence[RoundObservation]) -> None:
        """Fold one round of observations into the history.

        The deviation scale is computed *within* the round (median deviation) so
        that reputation reflects relative, not absolute, distance from the
        aggregate. That keeps the signal meaningful as the model converges and
        all updates shrink.
        """
        if not observations:
            return
        self.rounds_observed += 1
        devs = [
            o.deviation
            for o in observations
            if o.deviation is not None and np.isfinite(o.deviation)
        ]
        scale = float(np.median(devs)) if devs else 0.0
        if scale <= 0.0:
            scale = 1.0

        for obs in observations:
            rec = self.record(obs.client_id)
            rec.rounds_selected += 1
            if obs.accepted:
                rec.rounds_accepted += 1
            else:
                rec.rounds_rejected += 1

            if obs.loss is not None and np.isfinite(obs.loss):
                rec.last_loss = float(obs.loss)
                rec.mean_loss = _running_mean(
                    rec.mean_loss, float(obs.loss), rec.rounds_selected
                )
            if obs.deviation is not None and np.isfinite(obs.deviation):
                rec.last_deviation = float(obs.deviation)
                rec.mean_deviation = _running_mean(
                    rec.mean_deviation, float(obs.deviation), rec.rounds_selected
                )

            if not obs.accepted:
                score = self.rejection_penalty
            elif obs.deviation is None or not np.isfinite(obs.deviation):
                score = 1.0
            else:
                # exp(-d/scale): 1.0 at the origin, ~0.37 at the round median.
                score = float(np.exp(-float(obs.deviation) / scale))
            rec.reputation = (1.0 - self.alpha) * rec.reputation + self.alpha * score

    def to_dict(self) -> Dict[str, object]:
        """JSON-serializable snapshot of the whole history."""
        return {
            "alpha": self.alpha,
            "rounds_observed": self.rounds_observed,
            "clients": {k: v.to_dict() for k, v in sorted(self.records.items())},
        }

    def ranking(self) -> List[str]:
        """Client ids ordered by descending reputation (ties broken by id)."""
        return [
            r.client_id
            for r in sorted(
                self.records.values(), key=lambda r: (-r.reputation, r.client_id)
            )
        ]


def _running_mean(current: Optional[float], value: float, count: int) -> float:
    """Incremental mean update; ``count`` is the new sample count."""
    if current is None or count <= 1:
        return value
    return current + (value - current) / float(count)


__all__ = ["ClientRecord", "RoundObservation", "ClientHistory"]
