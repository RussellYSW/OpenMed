"""Contributor standing tiers and the governance weight they map to.

Standing converts recorded contribution into a governance voting weight, so that
sites which carry the commons' costs (data, compute, evaluation, review) have
proportionate say. The thresholds and weights here are *defaults for a
deployment to replace*: they encode a governance choice, not an empirical
finding, and nothing in this package validates that a particular ladder is fair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class StandingTier:
    """One rung of the standing ladder."""

    name: str
    min_credit: float
    voting_weight: int

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "name": self.name,
            "min_credit": self.min_credit,
            "voting_weight": self.voting_weight,
        }


#: Default ladder. Deployments are expected to set their own in governance.
DEFAULT_TIERS: Tuple[StandingTier, ...] = (
    StandingTier("observer", 0.0, 0),
    StandingTier("participant", 5.0, 1),
    StandingTier("contributor", 20.0, 2),
    StandingTier("steward", 50.0, 3),
)

#: Returned by :func:`tier_for` when no rung of the ladder is reached. Carries
#: voting weight 0, so an unqualified contributor is never handed governance
#: weight by default.
UNRANKED_TIER = StandingTier("unranked", 0.0, 0)


def tier_for(credit: float, tiers: Sequence[StandingTier] = DEFAULT_TIERS) -> StandingTier:
    """Return the highest tier whose ``min_credit`` is met by ``credit``.

    Only tiers the contributor actually qualifies for are considered. When the
    ladder's lowest rung sits above ``credit`` -- which is legitimate, since
    :data:`DEFAULT_TIERS` is a default a deployment is expected to replace and a
    replacement need not start at zero -- no tier is met and
    :data:`UNRANKED_TIER` is returned. That fallback carries voting weight 0, so
    a site with no recorded contribution is not handed the bottom rung's weight.

    Ties are broken by ladder order: with two qualifying tiers at the same
    ``min_credit``, the first one listed wins.
    """
    eligible = [t for t in tiers if credit >= t.min_credit]
    if not eligible:
        return UNRANKED_TIER
    best = eligible[0]
    for tier in eligible[1:]:
        if tier.min_credit > best.min_credit:
            best = tier
    return best


@dataclass(frozen=True)
class ContributorStanding:
    """One contributor's standing at a point in time."""

    actor: str
    institution: str
    credit: float
    tier: StandingTier
    event_counts: Mapping[str, int] = field(default_factory=dict)

    @property
    def voting_weight(self) -> int:
        """Governance voting weight implied by the tier."""
        return self.tier.voting_weight

    def count(self, kind: str) -> int:
        """Return how many events of ``kind`` this contributor has."""
        return int(self.event_counts.get(kind, 0))

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "actor": self.actor,
            "institution": self.institution,
            "credit": round(self.credit, 4),
            "tier": self.tier.name,
            "voting_weight": self.voting_weight,
            "event_counts": dict(sorted(self.event_counts.items())),
        }


@dataclass(frozen=True)
class StandingReport:
    """A snapshot of every contributor's standing."""

    rows: Tuple[ContributorStanding, ...] = ()
    generated_at: str = ""
    ledger_verified: bool = False
    n_events: int = 0

    def by_actor(self, actor: str) -> ContributorStanding:
        """Return one contributor's row.

        Raises
        ------
        KeyError
            If the actor is not in the report.
        """
        for row in self.rows:
            if row.actor == actor:
                return row
        raise KeyError(actor)

    def total_voting_weight(self) -> int:
        """Return the sum of voting weights across contributors."""
        return sum(row.voting_weight for row in self.rows)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict of the report."""
        return {
            "generated_at": self.generated_at,
            "ledger_verified": self.ledger_verified,
            "n_events": self.n_events,
            "total_voting_weight": self.total_voting_weight(),
            "contributors": [row.to_dict() for row in self.rows],
        }

    def to_markdown(self) -> str:
        """Render the report as a Markdown table."""
        lines = [
            "# Contributor standing",
            "",
            f"- events recorded: {self.n_events}",
            f"- ledger verified: {'yes' if self.ledger_verified else 'NO'}",
            f"- generated at: {self.generated_at or 'n/a'}",
            "",
            "| contributor | institution | credit | tier | voting weight |",
            "|---|---|---:|---|---:|",
        ]
        for row in self.rows:
            lines.append(
                f"| {row.actor} | {row.institution or '-'} | {row.credit:.1f} | "
                f"{row.tier.name} | {row.voting_weight} |"
            )
        lines.append("")
        lines.append(
            "Tiers and weights are governance defaults shipped with the package, "
            "not empirical findings."
        )
        return "\n".join(lines) + "\n"


__all__ = [
    "DEFAULT_TIERS",
    "UNRANKED_TIER",
    "ContributorStanding",
    "StandingReport",
    "StandingTier",
    "tier_for",
]
