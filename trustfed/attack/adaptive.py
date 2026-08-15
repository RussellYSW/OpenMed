"""Adaptive, aggregator-aware Byzantine attacks.

A defense evaluated only against attacks that ignore it is not evaluated at all.
The attacks in :mod:`trustfed.attack.poison` are static: they perturb one
client's vector without looking at anything else. The adversary here is the
harder threat model that Byzantine-robustness papers are actually judged on:

* it **colludes** -- all ``f`` compromised sites submit a coordinated set of
  vectors chosen together;
* it is **omniscient about the honest updates** for the current round (the
  standard worst-case assumption: the attacker has compromised a site that can
  observe the broadcast, or simply has a good model of its peers);
* it **knows which aggregation rule is running** and picks a strategy for it.

Strategies implemented
----------------------
``fedavg``
    Unbounded shift: choose the malicious vectors so the resulting mean is
    approximately the negation of the honest mean.
``median`` / ``trimmed_mean``
    ``min-max`` (Shejwalkar and Houmansadr, NDSS 2021): displace along the
    negated honest mean by the largest step whose distance to every honest
    update still fits inside the largest honest-to-honest distance. The
    malicious vectors are therefore indistinguishable from a legitimately
    unusual site by any distance test, but still drag the order statistic.
    "A Little Is Enough" (Baruch et al., NeurIPS 2019) is also available as the
    ``alie`` strategy; note that its theoretical deviation ``z`` is exactly zero
    for small federations with a low attacker fraction (e.g. n=8, f=2), which is
    a true statement about the median's strength, not a bug.
``krum`` / ``multi_krum``
    Fang-style directed deviation (Fang et al., USENIX Security 2020): all
    attackers submit one identical vector displaced along ``-sign(mean)``, with
    the displacement magnitude found by search as the largest value that Krum
    still selects. Identical copies make the attackers each other's nearest
    neighbours, which is what Krum's score rewards.
``norm_clip`` / ``centered_clipping``
    Sit exactly on the clipping boundary in the most harmful direction. Clipping
    caps the damage per round but cannot remove it, so the attacker extracts the
    maximum the bound allows.

Scope: this code runs entirely in-process against arrays owned by the
simulation. It is a defense-evaluation harness, not an exploitation tool.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from trustfed.attack.errors import AttackError
from trustfed.attack.stats import alie_z, pairwise_distances

_STRATEGY_FOR = {
    "fedavg": "mean_shift",
    "median": "min_max",
    "trimmed_mean": "min_max",
    "krum": "krum_search",
    "multi_krum": "krum_search",
    "norm_clip": "clip_boundary",
    "centered_clipping": "clip_boundary",
}

_DEFAULT_STRATEGY = "alie"


@dataclass
class AdaptiveAdversary:
    """A colluding adversary that tailors its updates to the aggregation rule.

    Parameters
    ----------
    target_aggregator:
        Registry name of the rule being attacked (see
        :data:`trustfed.aggregation.AGGREGATORS`). An unrecognized name falls
        back to the "A Little Is Enough" strategy.
    strength:
        Multiplier on the chosen displacement. ``1.0`` is the textbook attack;
        larger is more aggressive and easier to detect.
    seed:
        Seed for the tiny amount of randomness used to break ties.
    strategy:
        Force a strategy instead of deriving it from ``target_aggregator``.

    Notes
    -----
    Stateless across rounds by design, apart from ``round_index`` bookkeeping,
    so a benchmark run is reproducible from its seed.
    """

    target_aggregator: str = "fedavg"
    strength: float = 1.0
    seed: int = 0
    strategy: Optional[str] = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.strength <= 0.0 or not math.isfinite(self.strength):
            raise AttackError(f"strength must be positive and finite, got {self.strength}")
        self._rng = np.random.default_rng(self.seed)

    @property
    def resolved_strategy(self) -> str:
        """The strategy this adversary will actually use."""
        if self.strategy is not None:
            return self.strategy
        return _STRATEGY_FOR.get(self.target_aggregator, _DEFAULT_STRATEGY)

    def craft(
        self,
        honest_updates: Sequence[np.ndarray],
        n_malicious: int,
        *,
        global_theta: Optional[np.ndarray] = None,
        n_byzantine_assumed: Optional[int] = None,
        round_index: int = 0,
    ) -> List[np.ndarray]:
        """Return ``n_malicious`` coordinated malicious parameter vectors.

        Parameters
        ----------
        honest_updates:
            The honest clients' vectors for this round (the attacker's view).
        n_malicious:
            How many vectors to produce.
        global_theta:
            The current global model, used by the clipping strategy as the
            natural reference point.
        n_byzantine_assumed:
            The ``f`` the *defender* configured, which the Krum search needs to
            reproduce the defender's scoring. Defaults to ``n_malicious``.
        round_index:
            Round number, for callers that want round-varying behaviour; the
            built-in strategies are round-invariant.

        Raises
        ------
        AttackError
            If ``n_malicious`` is negative or there are no honest updates to
            observe.
        """
        f = int(n_malicious)
        if f < 0:
            raise AttackError(f"n_malicious must be >= 0, got {f}")
        if f == 0:
            return []
        H = [np.asarray(u, dtype=float) for u in honest_updates]
        if not H:
            raise AttackError("adaptive attack needs at least one honest update to observe")
        U = np.stack(H, axis=0)
        mu = U.mean(axis=0)
        sigma = U.std(axis=0) if U.shape[0] > 1 else np.abs(mu) * 0.1 + 1e-6
        n_total = U.shape[0] + f
        assumed = f if n_byzantine_assumed is None else int(n_byzantine_assumed)

        strategy = self.resolved_strategy
        if strategy == "mean_shift":
            mal = self._mean_shift(mu, n_total, f)
        elif strategy == "krum_search":
            mal = self._krum_search(U, mu, sigma, f, assumed)
        elif strategy == "clip_boundary":
            mal = self._clip_boundary(U, mu, global_theta)
        elif strategy == "alie":
            mal = self._alie(mu, sigma, n_total, f)
        elif strategy == "min_max":
            mal = self._min_max(U, mu)
        else:
            raise AttackError(
                f"unknown adaptive strategy '{strategy}'. options: "
                "mean_shift, alie, min_max, krum_search, clip_boundary"
            )
        return [mal.copy() for _ in range(f)]

    # -- strategies ---------------------------------------------------------

    def _mean_shift(self, mu: np.ndarray, n_total: int, f: int) -> np.ndarray:
        """Drive the plain mean to roughly ``-mu``.

        Solving ``((n-f)*mu + f*x)/n = -mu`` gives ``x = -(2n-f)/f * mu``.
        """
        factor = self.strength * (2.0 * n_total - f) / max(f, 1)
        return -factor * mu

    def _alie(
        self, mu: np.ndarray, sigma: np.ndarray, n_total: int, f: int
    ) -> np.ndarray:
        """"A Little Is Enough": shift by ``z`` honest standard deviations."""
        z = alie_z(n_total, f) * self.strength
        return mu - z * sigma

    def _min_max(self, U: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Largest displacement that stays within the honest spread.

        Implements the ``min-max`` (a.k.a. AGR-agnostic) attack of Shejwalkar
        and Houmansadr (NDSS 2021): choose ``x = mu + gamma * d`` with
        ``d = -mu/||mu||`` and ``gamma`` the largest value satisfying
        ``max_i ||x - x_i|| <= max_{i,j} ||x_i - x_j||``. Any defense that
        rejects points by distance to the other updates must then either accept
        the malicious vector or reject an honest one.

        The bound is found by doubling then bisection, both deterministic.
        """
        n = U.shape[0]
        if n < 2:
            return mu.copy()
        norm = float(np.linalg.norm(mu))
        direction = -mu / norm if norm > 0 else -np.ones_like(mu) / math.sqrt(mu.size)
        budget = float(np.max(pairwise_distances(U)))
        if budget <= 0.0:
            return mu.copy()

        def feasible(gamma: float) -> bool:
            """Does the displaced vector stay inside the honest distance budget?"""
            x = mu + gamma * direction
            return float(np.max(np.linalg.norm(U - x[None, :], axis=1))) <= budget

        lo, hi = 0.0, 1.0
        for _ in range(60):
            if feasible(hi):
                lo, hi = hi, hi * 2.0
            else:
                break
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if feasible(mid):
                lo = mid
            else:
                hi = mid
        return mu + self.strength * lo * direction

    def _krum_search(
        self,
        U: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        f: int,
        assumed: int,
    ) -> np.ndarray:
        """Largest ``-sign(mu)`` displacement that Krum still selects.

        Halving search from a large initial displacement, exactly as in Fang et
        al.: propose ``mu + lam * d``, ask the defender's own Krum whether it
        would be chosen, and halve ``lam`` until it is. If no displacement is
        accepted (e.g. the precondition ``n > 2f+2`` fails and Krum degenerates
        to a median), fall back to the ALIE displacement, which is a valid
        attack on the median.
        """
        from trustfed.aggregation.robust import krum  # local import: avoids a cycle

        direction = -np.sign(mu)
        direction[direction == 0.0] = -1.0
        scale = float(np.mean(np.abs(mu))) or 1.0
        lam = 10.0 * scale * self.strength

        for _ in range(30):
            candidate = mu + lam * direction
            proposal = np.concatenate(
                [U, np.repeat(candidate[None, :], f, axis=0)], axis=0
            )
            selected = krum(list(proposal), n_byzantine=max(assumed, 1))
            if np.allclose(selected, candidate, rtol=1e-9, atol=1e-12):
                return candidate
            lam *= 0.5
            if lam < 1e-9:
                break
        return self._alie(mu, sigma, U.shape[0] + f, f)

    def _clip_boundary(
        self, U: np.ndarray, mu: np.ndarray, global_theta: Optional[np.ndarray]
    ) -> np.ndarray:
        """Sit on the clipping radius, pointing away from the honest update.

        The defender's adaptive radius is the median deviation norm, so the
        attacker estimates it the same way and submits a vector just outside it:
        clipping then projects the vector back to exactly the radius, which is
        the most damage the bound permits.
        """
        ref = np.median(U, axis=0) if global_theta is None else np.asarray(
            global_theta, dtype=float
        )
        deltas = U - ref[None, :]
        tau = float(np.median(np.linalg.norm(deltas, axis=1)))
        if tau <= 0.0:
            tau = float(np.linalg.norm(mu)) or 1.0
        drift = mu - ref
        norm = float(np.linalg.norm(drift))
        direction = -drift / norm if norm > 0 else -np.ones_like(mu) / math.sqrt(mu.size)
        # 1.5x the radius: clipping pulls it back to exactly tau, and an
        # unclipped rule (or a larger radius) suffers more.
        return ref + 1.5 * self.strength * tau * direction


def adaptive_updates(
    honest_updates: Sequence[np.ndarray],
    n_malicious: int,
    *,
    aggregator: str = "fedavg",
    strength: float = 1.0,
    seed: int = 0,
    global_theta: Optional[np.ndarray] = None,
    n_byzantine_assumed: Optional[int] = None,
) -> List[np.ndarray]:
    """Functional shortcut around :class:`AdaptiveAdversary`."""
    adversary = AdaptiveAdversary(
        target_aggregator=aggregator, strength=strength, seed=seed
    )
    return adversary.craft(
        honest_updates,
        n_malicious,
        global_theta=global_theta,
        n_byzantine_assumed=n_byzantine_assumed,
    )


def strategy_map() -> Dict[str, str]:
    """Copy of the aggregator-name -> strategy-name mapping."""
    return dict(_STRATEGY_FOR)


__all__ = [
    "AdaptiveAdversary",
    "adaptive_updates",
    "alie_z",
    "strategy_map",
]
