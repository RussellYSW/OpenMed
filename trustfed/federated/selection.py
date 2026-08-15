"""Client selection: who gets to contribute to a given round.

Selection is a security control as much as an efficiency one. Sampling a subset
of sites per round is standard practice for bandwidth, but *which* subset also
decides how much a compromised site can influence the model over time. Three
policies are provided:

``RandomSelector``
    Uniform sampling. Unbiased and the right default, but an attacker is
    selected in proportion to its share of the federation, every round, forever.
``LossBasedSelector``
    Prefer sites whose local loss is highest -- the "power of choice" heuristic
    (Cho, Wang and Joshi, 2020), which speeds up convergence on non-IID data.
    Note the security tradeoff: an attacker can *self-report* a high loss to be
    picked more often, so this policy should not be used alone under threat.
``ReputationSelector``
    Weight sampling by an exponentially-smoothed reputation built from observed
    behaviour: attestation outcomes and how far each site's update sat from the
    round's robust aggregate. Sites that repeatedly land far from the consensus
    lose selection probability. This is a *soft* control -- it reduces exposure,
    it does not prove anyone is malicious, and a patient attacker that behaves
    for many rounds regains full reputation.

All selectors are deterministic given ``(seed, round_index)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from trustfed.federated.errors import SelectionError
from trustfed.federated.history import ClientHistory, ClientRecord, RoundObservation


def _weighted_sample_without_replacement(
    ids: Sequence[str], weights: np.ndarray, k: int, rng: np.random.Generator
) -> List[str]:
    """Efraimidis-Spirakis weighted sampling without replacement.

    Draws ``key_i = u_i ** (1 / w_i)`` and takes the ``k`` largest keys, which
    yields a sample whose inclusion probabilities are proportional to ``w`` for
    ``k = 1`` and a well-behaved generalization above that.
    """
    w = np.asarray(weights, dtype=float)
    w = np.clip(w, 1e-9, None)
    u = rng.uniform(size=w.shape[0])
    keys = np.power(u, 1.0 / w)
    order = np.argsort(-keys, kind="stable")[:k]
    return [ids[i] for i in sorted(order)]


class Selector(ABC):
    """Chooses which clients participate in a round.

    Implementations must be deterministic given ``(self.seed, round_index)`` and
    must return a non-empty subset of the offered ids whenever at least one id
    is offered.
    """

    name: str = "selector"

    def __init__(self, fraction: float = 1.0, *, seed: int = 0, min_clients: int = 1):
        if not 0.0 < fraction <= 1.0:
            raise SelectionError(f"fraction must be in (0, 1], got {fraction}")
        if min_clients < 1:
            raise SelectionError(f"min_clients must be >= 1, got {min_clients}")
        self.fraction = float(fraction)
        self.seed = int(seed)
        self.min_clients = int(min_clients)

    def _k(self, n: int) -> int:
        """Number of clients to select out of ``n`` offered."""
        return int(min(n, max(self.min_clients, round(self.fraction * n))))

    def _rng(self, round_index: int) -> np.random.Generator:
        """Round-specific generator, so selection is reproducible per round."""
        return np.random.default_rng([self.seed, int(round_index)])

    @abstractmethod
    def select(
        self,
        client_ids: Sequence[str],
        *,
        round_index: int,
        history: Optional[ClientHistory] = None,
    ) -> List[str]:
        """Return the ids selected for ``round_index``."""

    def describe(self) -> Dict[str, object]:
        """JSON-serializable description of this policy."""
        return {
            "name": self.name,
            "fraction": self.fraction,
            "seed": self.seed,
            "min_clients": self.min_clients,
        }


class AllClientsSelector(Selector):
    """Select every client every round (the classic cross-silo setting)."""

    name = "all"

    def __init__(self, **_: object) -> None:
        super().__init__(1.0, seed=0, min_clients=1)

    def select(
        self,
        client_ids: Sequence[str],
        *,
        round_index: int,
        history: Optional[ClientHistory] = None,
    ) -> List[str]:
        """Return all offered ids, unchanged."""
        return list(client_ids)


class RandomSelector(Selector):
    """Uniform sampling without replacement.

    Security note: an attacker controlling a fraction ``p`` of sites is selected
    with probability ``p`` every round; sampling reduces its per-round exposure
    but not its long-run influence.
    """

    name = "random"

    def select(
        self,
        client_ids: Sequence[str],
        *,
        round_index: int,
        history: Optional[ClientHistory] = None,
    ) -> List[str]:
        """Return a uniform random subset of size ``fraction * n``."""
        ids = list(client_ids)
        if not ids:
            raise SelectionError("no clients offered to the selector")
        k = self._k(len(ids))
        rng = self._rng(round_index)
        idx = rng.choice(len(ids), size=k, replace=False)
        return [ids[i] for i in sorted(idx)]


class LossBasedSelector(Selector):
    """Prefer clients with the highest (or lowest) recent local loss.

    Unseen clients are always selected first, so every site is measured once
    before the policy starts biasing. Selection among seen clients uses a
    softmax over loss with temperature ``temperature``; as ``temperature ->
    inf`` this degrades to uniform sampling.

    Security note: the loss is *self-reported*. A malicious site can inflate it
    to buy more participation, so pair this with attestation and robust
    aggregation, or with :class:`ReputationSelector`.
    """

    name = "loss"

    def __init__(
        self,
        fraction: float = 1.0,
        *,
        seed: int = 0,
        min_clients: int = 1,
        temperature: float = 1.0,
        prefer: str = "high",
    ):
        super().__init__(fraction, seed=seed, min_clients=min_clients)
        if temperature <= 0.0:
            raise SelectionError(f"temperature must be > 0, got {temperature}")
        if prefer not in ("high", "low"):
            raise SelectionError(f"prefer must be 'high' or 'low', got {prefer!r}")
        self.temperature = float(temperature)
        self.prefer = prefer

    def select(
        self,
        client_ids: Sequence[str],
        *,
        round_index: int,
        history: Optional[ClientHistory] = None,
    ) -> List[str]:
        """Return the subset chosen by the loss-softmax policy."""
        ids = list(client_ids)
        if not ids:
            raise SelectionError("no clients offered to the selector")
        k = self._k(len(ids))
        rng = self._rng(round_index)
        if history is None:
            idx = rng.choice(len(ids), size=k, replace=False)
            return [ids[i] for i in sorted(idx)]

        unseen = [c for c in ids if history.records.get(c, None) is None]
        if len(unseen) >= k:
            return sorted(unseen)[:k]

        seen = [c for c in ids if c not in set(unseen)]
        losses = np.array(
            [history.record(c).last_loss or 0.0 for c in seen], dtype=float
        )
        sign = 1.0 if self.prefer == "high" else -1.0
        centered = sign * (losses - losses.mean()) / self.temperature
        weights = np.exp(centered - centered.max())
        chosen = _weighted_sample_without_replacement(
            seen, weights, k - len(unseen), rng
        )
        return sorted(set(unseen) | set(chosen))


class ReputationSelector(Selector):
    """Reputation-weighted sampling driven by :class:`ClientHistory`.

    Selection weight is ``max(reputation, floor) ** power`` plus an
    ``exploration`` term that keeps every site reachable, so a site that once
    looked anomalous can be re-measured instead of being permanently excluded.

    What this does *not* do: it is not an attestation substitute and it is not
    evidence of misbehaviour. A site with genuinely distinct case-mix will drift
    down; a patient attacker that submits honest updates for many rounds will
    drift back up. Use it to limit exposure, not to make accusations.
    """

    name = "reputation"

    def __init__(
        self,
        fraction: float = 1.0,
        *,
        seed: int = 0,
        min_clients: int = 1,
        power: float = 2.0,
        exploration: float = 0.1,
        floor: float = 0.0,
    ):
        super().__init__(fraction, seed=seed, min_clients=min_clients)
        if power < 0.0:
            raise SelectionError(f"power must be >= 0, got {power}")
        if not 0.0 <= exploration <= 1.0:
            raise SelectionError(f"exploration must be in [0, 1], got {exploration}")
        if not 0.0 <= floor <= 1.0:
            raise SelectionError(f"floor must be in [0, 1], got {floor}")
        self.power = float(power)
        self.exploration = float(exploration)
        self.floor = float(floor)

    def weights(
        self, client_ids: Iterable[str], history: Optional[ClientHistory]
    ) -> np.ndarray:
        """Selection weight per client (exposed for inspection and tests)."""
        ids = list(client_ids)
        if history is None:
            return np.ones(len(ids), dtype=float)
        reps = np.array(
            [max(history.reputation(c), self.floor) for c in ids], dtype=float
        )
        return np.power(np.clip(reps, 0.0, None), self.power) + self.exploration

    def select(
        self,
        client_ids: Sequence[str],
        *,
        round_index: int,
        history: Optional[ClientHistory] = None,
    ) -> List[str]:
        """Return a reputation-weighted subset."""
        ids = list(client_ids)
        if not ids:
            raise SelectionError("no clients offered to the selector")
        k = self._k(len(ids))
        rng = self._rng(round_index)
        w = self.weights(ids, history)
        return _weighted_sample_without_replacement(ids, w, k, rng)

    def describe(self) -> Dict[str, object]:
        """JSON-serializable description of this policy."""
        base = super().describe()
        base.update(
            {"power": self.power, "exploration": self.exploration, "floor": self.floor}
        )
        return base


SELECTORS = {
    "all": AllClientsSelector,
    "random": RandomSelector,
    "loss": LossBasedSelector,
    "reputation": ReputationSelector,
}


def get_selector(name: str, **kwargs: object) -> Selector:
    """Construct a selector by registry name.

    Raises
    ------
    SelectionError
        If ``name`` is not a registered policy.
    """
    try:
        cls = SELECTORS[name]
    except KeyError:
        raise SelectionError(
            f"unknown selector '{name}'. options: {sorted(SELECTORS)}"
        ) from None
    return cls(**kwargs)  # type: ignore[arg-type]


__all__ = [
    "ClientHistory",
    "ClientRecord",
    "RoundObservation",
    "Selector",
    "AllClientsSelector",
    "RandomSelector",
    "LossBasedSelector",
    "ReputationSelector",
    "SELECTORS",
    "get_selector",
]
