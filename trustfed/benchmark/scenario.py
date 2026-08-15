"""Declarative scenarios for the shared defense benchmark.

A :class:`Scenario` is a complete, hashable description of one federated run:
the cohort, who is compromised, what they do, which aggregation rule defends,
and the seed. Everything the runner needs is in the record, so a result can
always be traced back to -- and reproduced from -- the scenario that produced it.

The point of a shared harness is comparability. A defense contributed by one
group and a defense contributed by another should be measured on the same
cohorts, the same attacks and the same seeds, and their results should sit in
the same table.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from trustfed.benchmark.errors import BenchmarkError

#: Aggregators exercised by the default grid.
DEFAULT_AGGREGATORS: Tuple[str, ...] = (
    "fedavg",
    "median",
    "trimmed_mean",
    "krum",
    "multi_krum",
    "norm_clip",
    "centered_clipping",
)

#: Attacks exercised by the default grid, cheapest threat model first.
DEFAULT_ATTACKS: Tuple[str, ...] = ("none", "label_flip", "sign_flip", "adaptive")

#: Name reserved for the aggregator-aware colluding adversary.
ADAPTIVE = "adaptive"


@dataclass(frozen=True)
class Scenario:
    """One benchmark configuration.

    Parameters
    ----------
    n_sites:
        Number of simulated sites in the federation.
    n_byzantine:
        Number of compromised sites (always the last ``n_byzantine`` sites, so
        the assignment is deterministic).
    attack:
        ``"none"``, an update-space attack from
        :data:`trustfed.attack.ATTACKS`, a data-space attack from
        :data:`trustfed.attack.DATA_ATTACKS`, or ``"adaptive"`` for the
        aggregator-aware colluding adversary.
    aggregator:
        Registry name of the defending aggregation rule.
    rounds:
        Federated rounds to run.
    seed:
        Master seed; controls the cohort, the client attack randomness, the
        selector and the adversary.
    samples_per_site, test_size, feature_shift, label_skew:
        Cohort knobs forwarded to :class:`trustfed.data.CohortSpec`.
    local_epochs:
        Local gradient steps per round.
    selector:
        Client-selection policy name (``all``, ``random``, ``loss``,
        ``reputation``).
    selection_fraction:
        Fraction of sites sampled per round by the selector.
    attestation:
        Whether the server verifies attestation quotes.
    tamper_code:
        Whether the compromised sites declare an unapproved code identity (only
        meaningful together with ``attestation``).
    attack_kwargs:
        Extra keyword arguments passed to the attack.
    label:
        Optional human-readable label; :attr:`name` falls back to a derived one.
    """

    n_sites: int = 8
    n_byzantine: int = 2
    attack: str = "none"
    aggregator: str = "fedavg"
    rounds: int = 15
    seed: int = 0
    samples_per_site: int = 150
    test_size: int = 800
    feature_shift: float = 0.35
    label_skew: float = 0.6
    local_epochs: int = 5
    selector: str = "all"
    selection_fraction: float = 1.0
    attestation: bool = False
    tamper_code: bool = False
    attack_kwargs: Dict[str, Any] = field(default_factory=dict)
    label: Optional[str] = None

    def __post_init__(self) -> None:
        if self.n_sites < 2:
            raise BenchmarkError(f"n_sites must be >= 2, got {self.n_sites}")
        if not 0 <= self.n_byzantine < self.n_sites:
            raise BenchmarkError(
                f"n_byzantine must be in [0, n_sites), got {self.n_byzantine}"
            )
        if self.rounds < 1:
            raise BenchmarkError(f"rounds must be >= 1, got {self.rounds}")
        if self.n_byzantine == 0 and self.attack != "none":
            raise BenchmarkError(
                f"scenario declares attack '{self.attack}' but no Byzantine sites"
            )

    @property
    def name(self) -> str:
        """Stable identifier used as the row key in reports."""
        if self.label:
            return self.label
        return (
            f"{self.aggregator}/{self.attack}/n{self.n_sites}"
            f"f{self.n_byzantine}/s{self.seed}"
        )

    @property
    def byzantine_fraction(self) -> float:
        """Fraction of the federation that is compromised."""
        return self.n_byzantine / float(self.n_sites)

    def clean_twin(self) -> "Scenario":
        """The same configuration with no attack, used as the reference row."""
        return replace(self, attack="none", n_byzantine=0, tamper_code=False, label=None)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable view of the scenario."""
        return {
            "name": self.name,
            "n_sites": self.n_sites,
            "n_byzantine": self.n_byzantine,
            "byzantine_fraction": round(self.byzantine_fraction, 4),
            "attack": self.attack,
            "aggregator": self.aggregator,
            "rounds": self.rounds,
            "seed": self.seed,
            "samples_per_site": self.samples_per_site,
            "test_size": self.test_size,
            "feature_shift": self.feature_shift,
            "label_skew": self.label_skew,
            "local_epochs": self.local_epochs,
            "selector": self.selector,
            "selection_fraction": self.selection_fraction,
            "attestation": self.attestation,
            "tamper_code": self.tamper_code,
            "attack_kwargs": dict(self.attack_kwargs),
        }


def grid(
    *,
    aggregators: Sequence[str] = DEFAULT_AGGREGATORS,
    attacks: Sequence[str] = DEFAULT_ATTACKS,
    n_byzantine: Sequence[int] = (2,),
    seeds: Sequence[int] = (0,),
    **scenario_kwargs: Any,
) -> List[Scenario]:
    """Build the cross product of aggregators, attacks, attacker counts and seeds.

    Clean (``attack="none"``) rows are emitted once per aggregator and seed with
    ``n_byzantine=0``, so every attacked row has a reference to be compared
    against.
    """
    out: List[Scenario] = []
    for seed in seeds:
        for agg in aggregators:
            if "none" in attacks:
                out.append(
                    Scenario(
                        aggregator=agg,
                        attack="none",
                        n_byzantine=0,
                        seed=seed,
                        **scenario_kwargs,
                    )
                )
            for atk in attacks:
                if atk == "none":
                    continue
                for f in n_byzantine:
                    if f <= 0:
                        continue
                    out.append(
                        Scenario(
                            aggregator=agg,
                            attack=atk,
                            n_byzantine=f,
                            seed=seed,
                            **scenario_kwargs,
                        )
                    )
    return out


def default_grid(seed: int = 0) -> List[Scenario]:
    """The standard grid: every aggregator against every attack at f=2 and f=3.

    Sized to run in a few seconds on a laptop CPU so that it can sit in CI and
    in a contributor's edit-test loop. ``n_sites=8`` with ``f in {2, 3}`` spans
    the interesting region: 25% attackers is inside every rule's documented
    tolerance, 37.5% is close to the edge for the threshold rules.
    """
    return grid(n_byzantine=(2, 3), seeds=(seed,))


__all__ = [
    "Scenario",
    "grid",
    "default_grid",
    "DEFAULT_AGGREGATORS",
    "DEFAULT_ATTACKS",
    "ADAPTIVE",
]
