"""Tests for credit recording, citable identifiers and standing (Component 6).

Reciprocity and counterparty attestation live in
``test_incentives_reciprocity.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trustfed.incentives import (
    DEFAULT_TIERS,
    UNRANKED_TIER,
    Attribution,
    CitableIdentifier,
    CreditLedger,
    IdentifierError,
    ReleaseAttribution,
    StandingTier,
    UnknownContributorError,
    UnknownCreditKindError,
    tier_for,
)
from trustfed.ledger import FileLedger, HmacSigner, InMemoryLedger

BUNDLE = "openmed:bundle:sha256:" + "cd" * 32


def _clock():
    state = {"t": 0}

    def tick() -> str:
        state["t"] += 1
        return f"2026-05-01T00:00:{state['t']:02d}.000000Z"

    return tick


def make_credit(ledger=None) -> CreditLedger:
    if ledger is None:
        ledger = InMemoryLedger(signer=HmacSigner(b"credit"))
    return CreditLedger(ledger, clock=_clock())

# ------------------------------------------------------------------- recording


def test_credit_events_are_appended_to_the_ledger():
    credit = make_credit()
    event = credit.record(
        "site_a", "data_contribution", institution="INST_A", ref=BUNDLE
    )
    assert event.credit == pytest.approx(3.0)
    assert len(credit) == 1
    assert len(credit.ledger) == 1
    assert credit.ledger.get(0).payload["event"] == "credit_entry"
    assert credit.verify().ok is True


def test_quantity_scales_credit_and_must_be_positive():
    credit = make_credit()
    event = credit.record("site_a", "training_round", quantity=4)
    assert event.credit == pytest.approx(4.0)
    with pytest.raises(ValueError):
        credit.record("site_a", "training_round", quantity=0)


def test_unweighted_kind_is_refused_rather_than_scoring_zero():
    credit = make_credit()
    with pytest.raises(UnknownCreditKindError):
        credit.record("site_a", "vibes")


def test_unknown_contributor_raises():
    credit = make_credit()
    with pytest.raises(UnknownContributorError):
        credit.standing("nobody")
    assert credit.credit("nobody") == 0.0


def test_events_can_be_filtered_by_actor_and_kind():
    credit = make_credit()
    credit.record("site_a", "training_round")
    credit.record("site_b", "evaluation_served")
    credit.record("site_b", "evaluation_served")
    assert len(credit.events("site_b")) == 2
    assert len(credit.events(kind="training_round")) == 1
    assert credit.served_evaluations("site_b") == 2
    assert credit.contributors() == ("site_a", "site_b")


# --------------------------------------------------------------- identifiers


def test_minted_identifiers_are_deterministic_and_marked_unregistered():
    first = CitableIdentifier.mint(BUNDLE)
    second = CitableIdentifier.mint(BUNDLE)
    assert first == second
    assert first.registered is False
    assert first.uri.startswith("oid:openmed.local/")
    assert CitableIdentifier.mint("other-bundle") != first


def test_registered_doi_is_distinguished_from_a_local_identifier():
    registered = CitableIdentifier.for_registered_doi("10.5281", "zenodo.000000")
    assert registered.registered is True
    assert registered.uri == "doi:10.5281/zenodo.000000"
    assert CitableIdentifier.parse("doi:10.5281/zenodo.000000") == registered
    assert CitableIdentifier.parse("openmed.local/abc").registered is False


def test_malformed_identifiers_are_rejected():
    with pytest.raises(IdentifierError):
        CitableIdentifier(prefix="", suffix="x")
    with pytest.raises(IdentifierError):
        CitableIdentifier(prefix="a/b", suffix="x")
    with pytest.raises(IdentifierError):
        CitableIdentifier.parse("no-slash-here")
    with pytest.raises(IdentifierError):
        CitableIdentifier.mint("")


def test_release_attribution_citation_states_that_the_id_is_unregistered():
    release = ReleaseAttribution.for_bundle(
        BUNDLE,
        title="PD decline classifier",
        year=2026,
        version="1.0",
        contributors=[
            Attribution("site_a", "training", "INST_A", share=0.5),
            Attribution("site_b", "evaluation", "INST_B", share=0.5),
        ],
    )
    citation = release.to_citation()
    assert "site_a, site_b" in citation
    assert "locally minted, unregistered" in citation
    assert "oid:openmed.local/" in citation
    assert ReleaseAttribution.from_dict(release.to_dict()) == release


def test_release_credit_is_attributed_to_each_contributor():
    credit = make_credit()
    release = ReleaseAttribution.for_bundle(
        BUNDLE,
        title="PD decline classifier",
        year=2026,
        contributors=[
            Attribution("site_a", "training", "INST_A"),
            Attribution("site_b", "evaluation", "INST_B"),
        ],
    )
    events = credit.record_release(release)
    assert len(events) == 2
    assert {e.actor for e in events} == {"site_a", "site_b"}
    assert all(e.ref == release.identifier.uri for e in events)
    assert credit.credit("site_a") == pytest.approx(3.0)


# ------------------------------------------------------------------- standing


def test_tiers_map_credit_to_governance_voting_weight():
    assert tier_for(0.0).name == "observer"
    assert tier_for(0.0).voting_weight == 0
    assert tier_for(5.0).name == "participant"
    assert tier_for(21.0).name == "contributor"
    assert tier_for(500.0).name == "steward"
    assert tier_for(500.0).voting_weight == 3
    assert DEFAULT_TIERS[0].to_dict()["name"] == "observer"


def test_a_ladder_whose_lowest_rung_is_above_zero_grants_nothing_below_it():
    """standing.py tells deployments to replace the ladder, so it may not
    assume the bottom rung sits at zero. A site with no contribution must not
    be handed the lowest tier's voting weight by default."""
    ladder = (
        StandingTier("bronze", 10.0, 1),
        StandingTier("silver", 50.0, 2),
    )
    assert tier_for(0.0, ladder) is UNRANKED_TIER
    assert tier_for(0.0, ladder).voting_weight == 0
    assert tier_for(9.99, ladder).name == "unranked"
    assert tier_for(10.0, ladder).name == "bronze"
    assert tier_for(60.0, ladder).name == "silver"

    # And the tier a contributor is actually given follows the same rule.
    credit = CreditLedger(
        InMemoryLedger(signer=HmacSigner(b"credit")), clock=_clock(), tiers=ladder
    )
    credit.record("site_a", "training_round", institution="INST_A")
    standing = credit.standing("site_a")
    assert standing.credit == pytest.approx(1.0)
    assert standing.tier.name == "unranked"
    assert standing.voting_weight == 0
    assert credit.standing_report().total_voting_weight() == 0


def test_standing_report_ranks_contributors_and_reports_ledger_health():
    credit = make_credit()
    for _ in range(8):
        credit.record("site_a", "data_contribution", institution="INST_A")
    credit.record("site_b", "evaluation_served", institution="INST_B")
    credit.record("site_c", "training_round", institution="INST_C")

    report = credit.standing_report()
    assert [row.actor for row in report.rows] == ["site_a", "site_b", "site_c"]
    assert report.by_actor("site_a").credit == pytest.approx(24.0)
    assert report.by_actor("site_a").tier.name == "contributor"
    assert report.by_actor("site_a").voting_weight == 2
    assert report.by_actor("site_a").count("data_contribution") == 8
    assert report.total_voting_weight() == 2
    assert report.ledger_verified is True
    assert report.n_events == 10

    text = report.to_markdown()
    assert "| site_a | INST_A |" in text
    assert "governance defaults" in text
    assert report.to_dict()["contributors"][0]["tier"] == "contributor"
    with pytest.raises(KeyError):
        report.by_actor("site_z")


def test_standing_report_flags_a_tampered_credit_ledger(tmp_path: Path):
    """Inflating your own credit on disk is visible in the standing report."""
    path = tmp_path / "credit.jsonl"
    credit = make_credit(FileLedger(path, signer=HmacSigner(b"credit")))
    credit.record("site_a", "training_round", institution="INST_A")
    credit.record("site_b", "evaluation_served", institution="INST_B")
    assert credit.standing_report().ledger_verified is True

    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["payload"]["credit"] = 999.0
    record["payload"]["quantity"] = 999.0
    lines[0] = json.dumps(record, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = credit.standing_report()
    assert report.ledger_verified is False
    assert "NO" in report.to_markdown()
    assert credit.verify().ok is False
