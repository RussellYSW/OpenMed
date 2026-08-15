"""Documented Byzantine-tolerance bounds for each aggregation rule.

The numbers here are the *assumed* bounds from the source papers, restated for
this implementation. They are analytical statements about the aggregation
operator, not measured results: they say how many arbitrarily-corrupted updates
a rule can absorb before its output is no longer controlled by the honest
majority. Whether a real federation meets the papers' assumptions (bounded
honest variance, i.i.d.-ish gradients) is a separate, empirical question --
which is what :mod:`trustfed.benchmark` exists to probe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from trustfed.aggregation.errors import UnknownAggregatorError


@dataclass(frozen=True)
class ToleranceBound:
    """Analytical Byzantine tolerance of one aggregation rule.

    Attributes
    ----------
    name:
        Registry key of the rule.
    breakdown_fraction:
        Largest attacker fraction ``f/n`` for which the rule's guarantee holds.
        ``0.0`` means "no guarantee at any positive f".
    exact_threshold:
        ``True`` when the rule has a hard threshold (below it the honest signal
        dominates, above it the output can be arbitrary); ``False`` when the
        guarantee is a graceful bound on the attacker's influence instead.
    condition:
        The precondition on ``n`` and ``f``, in words.
    note:
        What the rule does and does not protect against.
    """

    name: str
    breakdown_fraction: float
    exact_threshold: bool
    condition: str
    note: str

    def max_byzantine(self, n_clients: int) -> int:
        """Largest ``f`` for which this rule still *recovers* the honest signal.

        This is the exact-recovery bound only. A rule whose guarantee is
        bounded influence rather than exact recovery (``exact_threshold`` is
        ``False``) returns ``0`` here, because at any ``f >= 1`` its output is
        still biased by the attacker -- it is merely bounded. Use
        :func:`bounded_influence_bound` for how large that residual bias may be.
        FedAvg, which has no guarantee at all, also returns ``0``.
        """
        n = int(n_clients)
        if n <= 0:
            return 0
        if not self.exact_threshold:
            return 0
        if self.name == "krum" or self.name == "multi_krum":
            # Krum requires n > 2f + 2, i.e. f < (n - 2) / 2.
            return max(0, (n - 3) // 2)
        f = int((self.breakdown_fraction * n) - 1e-9)
        if f * 2 >= n:
            f = (n - 1) // 2
        return max(0, f)

    def influence_bound(
        self, n_clients: int, n_byzantine: int, *, tau: float = 1.0
    ) -> Optional[float]:
        """Worst-case displacement of the output, for bounded-influence rules.

        Returns ``2 * tau * f / n``, the distance a clipped-mean rule's output
        can be moved by ``f`` attackers that always submit at the clip radius
        ``tau`` around the reference point (uniform weights). ``None`` for rules
        whose guarantee is a threshold rather than a bound on influence: below
        their threshold the displacement is zero, above it unbounded, so a
        single number would misdescribe them.
        """
        n, f = int(n_clients), int(n_byzantine)
        if self.exact_threshold or n <= 0:
            return None
        f = max(0, min(f, n))
        return 2.0 * float(tau) * f / float(n)


_BOUNDS: Dict[str, ToleranceBound] = {
    "fedavg": ToleranceBound(
        name="fedavg",
        breakdown_fraction=0.0,
        exact_threshold=True,
        condition="f = 0",
        note=(
            "Zero Byzantine tolerance. The mean has breakdown point 0: one client "
            "submitting an unbounded vector moves the output arbitrarily far. "
            "Included as the baseline an attack must beat."
        ),
    ),
    "median": ToleranceBound(
        name="median",
        breakdown_fraction=0.5,
        exact_threshold=True,
        condition="f < n/2 (strict majority honest, per coordinate)",
        note=(
            "Coordinate-wise median; breakdown point 1/2 (Yin et al., ICML 2018). "
            "Does not protect against a colluding majority, and because it works "
            "per coordinate it can return a point no client proposed -- which "
            "matters when honest updates are strongly non-IID."
        ),
    ),
    "trimmed_mean": ToleranceBound(
        name="trimmed_mean",
        breakdown_fraction=0.5,
        exact_threshold=True,
        condition="beta*n >= f and 2*beta < 1",
        note=(
            "Per-coordinate beta-trimmed mean (Yin et al., ICML 2018). The trim "
            "fraction must be chosen at least as large as the true attacker "
            "fraction: an under-set beta leaves malicious values in the average, "
            "an over-set beta discards honest tail sites."
        ),
    ),
    "krum": ToleranceBound(
        name="krum",
        breakdown_fraction=0.5,
        exact_threshold=True,
        condition="n > 2f + 2",
        note=(
            "Krum (Blanchard et al., NeurIPS 2017) selects the single update with "
            "the smallest sum of squared distances to its n-f-2 nearest "
            "neighbours. Requires an explicit f. Selecting one update discards "
            "the data of every other site, so it converges more slowly, and it is "
            "vulnerable to colluders that submit mutually-close updates."
        ),
    ),
    "multi_krum": ToleranceBound(
        name="multi_krum",
        breakdown_fraction=0.5,
        exact_threshold=True,
        condition="n > 2f + 2",
        note=(
            "Multi-Krum averages the m best-scoring updates (default m = n - f). "
            "Same requirement and same collusion weakness as Krum, but retains "
            "more honest data per round."
        ),
    ),
    "norm_clip": ToleranceBound(
        name="norm_clip",
        breakdown_fraction=1.0,
        exact_threshold=False,
        condition="f < n (bounded influence, no exact recovery)",
        note=(
            "Norm-clipped mean around a robust reference. Bounded influence: the "
            "output moves by at most 2*tau*f/n under uniform weights. It does not "
            "remove the attacker's effect, only caps it, and an attacker who "
            "always submits at the clip radius still biases every round."
        ),
    ),
    "centered_clipping": ToleranceBound(
        name="centered_clipping",
        breakdown_fraction=0.5,
        exact_threshold=True,
        condition="f/n < 1/2 with a reference near the honest mean",
        note=(
            "Centered clipping (Karimireddy et al., ICML 2021). Needs no explicit "
            "f, but the guarantee is conditional on the reference point still "
            "being close to the honest cluster; a slow drift attack that moves "
            "the reference degrades it."
        ),
    ),
}


def tolerance_bound(name: str) -> ToleranceBound:
    """Return the documented tolerance bound for a registered aggregator.

    Raises
    ------
    UnknownAggregatorError
        If ``name`` is not a registered rule.
    """
    try:
        return _BOUNDS[name]
    except KeyError:
        raise UnknownAggregatorError(
            f"no tolerance bound recorded for aggregator '{name}'. "
            f"known: {sorted(_BOUNDS)}"
        ) from None


def max_byzantine(name: str, n_clients: int) -> int:
    """Largest ``f`` at which ``name`` still recovers the honest signal exactly.

    Zero for FedAvg (no guarantee) *and* for bounded-influence rules such as
    ``norm_clip``, which never remove an attacker's contribution -- see
    :func:`bounded_influence_bound`.
    """
    return tolerance_bound(name).max_byzantine(n_clients)


def bounded_influence_bound(
    name: str, n_clients: int, n_byzantine: Optional[int] = None, *, tau: float = 1.0
) -> Optional[float]:
    """Worst-case output displacement for a bounded-influence rule.

    Parameters
    ----------
    name:
        Registry key of the rule.
    n_clients:
        Cohort size ``n``.
    n_byzantine:
        Attacker count ``f``; defaults to the worst case ``n - 1``.
    tau:
        Clip radius, in the units of the parameter vector. The default of 1.0
        makes the return value the coefficient ``2 * f / n`` rather than an
        absolute distance.

    Returns
    -------
    float or None
        ``2 * tau * f / n`` for rules whose guarantee is bounded influence;
        ``None`` for threshold rules (and for FedAvg, which bounds nothing).

    This is an analytical bound restated from the rule's definition, not a
    measurement: it says how far the output *can* be pushed, not how far it was
    pushed in any particular run.
    """
    bound = tolerance_bound(name)
    f = (int(n_clients) - 1) if n_byzantine is None else int(n_byzantine)
    return bound.influence_bound(n_clients, f, tau=tau)


def tolerance_table() -> str:
    """Render every documented bound as a Markdown table.

    The guarantee kind is printed next to the breakdown fraction on purpose: a
    fraction of 1.00 on a bounded-influence rule means "the output stays
    finite", not "tolerates everything".
    """
    lines = [
        "| rule | condition | breakdown fraction | guarantee |",
        "|---|---|---|---|",
    ]
    for key in sorted(_BOUNDS):
        b = _BOUNDS[key]
        kind = (
            "threshold: exact recovery below the fraction"
            if b.exact_threshold
            else "bounded influence: output stays finite, bias grows as 2*tau*f/n"
        )
        short = "threshold" if b.exact_threshold else "bounded influence"
        lines.append(
            f"| `{b.name}` | {b.condition} | {b.breakdown_fraction:.2f} "
            f"({short}) | {kind} |"
        )
    return "\n".join(lines) + "\n"


def all_bounds() -> Dict[str, ToleranceBound]:
    """Return a copy of the full bound registry."""
    return dict(_BOUNDS)


def recommended_trim_beta(n_clients: int, n_byzantine: int) -> float:
    """Smallest safe ``beta`` for :func:`trustfed.aggregation.trimmed_mean`.

    Trims just over ``n_byzantine`` values from each end, capped below 0.5 so at
    least one value survives.
    """
    if n_clients <= 0:
        return 0.0
    beta = (int(n_byzantine) + 0.5) / int(n_clients)
    return float(min(0.49, max(0.0, beta)))


def describe(name: str, n_clients: Optional[int] = None) -> str:
    """One-paragraph human-readable description of a rule's tolerance.

    States which *kind* of guarantee the number is: exact recovery up to some
    ``f``, or bounded influence with a residual bias that never reaches zero.
    """
    b = tolerance_bound(name)
    head = f"{b.name}: {b.condition}."
    if n_clients is not None:
        if b.exact_threshold:
            head += (
                f" With n={n_clients}, recovers the honest signal for "
                f"f<={b.max_byzantine(n_clients)}."
            )
        else:
            worst = b.influence_bound(n_clients, max(0, int(n_clients) - 1))
            head += (
                f" With n={n_clients}, exact recovery for f=0 only; for f>=1 the "
                f"output stays bounded but is biased by up to 2*tau*f/n "
                f"(at f={max(0, int(n_clients) - 1)}, {worst:.2f}*tau)."
            )
    return f"{head} {b.note}"


__all__ = [
    "ToleranceBound",
    "tolerance_bound",
    "max_byzantine",
    "bounded_influence_bound",
    "tolerance_table",
    "all_bounds",
    "recommended_trim_beta",
    "describe",
]
