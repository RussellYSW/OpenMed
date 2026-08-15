"""Tests for the reciprocity rule and counterparty attestation (Component 6).

The rule's whole value depends on where the served-evaluation count comes from,
so both the self-asserted path and the attested one are exercised here. Credit
recording, identifiers and standing live in ``test_incentives_credit.py``.
"""

from __future__ import annotations

import pytest

from trustfed.incentives import (
    AttestationRejectedError,
    CounterpartyRegistry,
    CreditLedger,
    EvaluationRequest,
    ReciprocityPolicy,
    ReciprocityRefused,
    ServiceAttestation,
)
from trustfed.ledger import HmacSigner, InMemoryLedger

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

# ---------------------------------------------------------------- reciprocity


def test_request_is_refused_until_the_site_has_served_as_evaluator():
    credit = make_credit()
    policy = ReciprocityPolicy(min_evaluations_served=2)
    request = EvaluationRequest("site_a", "site_b", BUNDLE)

    refusal = policy.decide(request, credit)
    assert refusal.allowed is False
    assert refusal.reason_code == "insufficient_reciprocity"
    assert refusal.observed == 0 and refusal.required == 2
    assert "served as evaluator 0 time(s)" in refusal.detail
    with pytest.raises(ReciprocityRefused) as exc:
        refusal.raise_if_refused()
    assert exc.value.reason_code == "insufficient_reciprocity"

    credit.record("site_a", "evaluation_served", institution="INST_A")
    assert policy.decide(request, credit).allowed is False
    credit.record("site_a", "evaluation_served", institution="INST_A")

    granted = policy.decide(request, credit)
    assert granted.allowed is True
    assert granted.reason_code == "granted"
    assert granted.observed == 2
    granted.raise_if_refused()  # no-op when allowed


def test_self_evaluation_is_refused():
    credit = make_credit()
    for _ in range(5):
        credit.record("site_a", "evaluation_served")
    policy = ReciprocityPolicy(min_evaluations_served=2)
    decision = policy.decide(EvaluationRequest("site_a", "site_a", BUNDLE), credit)
    assert decision.allowed is False
    assert decision.reason_code == "self_evaluation_refused"

    permissive = ReciprocityPolicy(min_evaluations_served=2, allow_self_evaluation=True)
    assert permissive.decide(
        EvaluationRequest("site_a", "site_a", BUNDLE), credit
    ).allowed is True


def test_exempt_actors_bypass_the_rule():
    credit = make_credit()
    policy = ReciprocityPolicy(min_evaluations_served=3, exempt_actors={"site_new"})
    decision = policy.decide(EvaluationRequest("site_new", "site_b", BUNDLE), credit)
    assert decision.allowed is True
    assert decision.reason_code == "exempt"


def test_standing_floor_can_be_required_in_addition():
    credit = make_credit()
    credit.record("site_a", "evaluation_served", institution="INST_A")
    credit.record("site_a", "evaluation_served", institution="INST_A")
    policy = ReciprocityPolicy(min_evaluations_served=2, min_voting_weight=2)
    decision = policy.decide(EvaluationRequest("site_a", "site_b", BUNDLE), credit)
    assert decision.allowed is False
    assert decision.reason_code == "insufficient_standing"

    for _ in range(8):
        credit.record("site_a", "data_contribution", institution="INST_A")
    assert policy.decide(EvaluationRequest("site_a", "site_b", BUNDLE), credit).allowed


def test_decisions_are_recorded_on_the_ledger():
    credit = make_credit()
    policy = ReciprocityPolicy(min_evaluations_served=1)
    refused = policy.decide_and_record(
        EvaluationRequest("site_a", "site_b", BUNDLE), credit
    )
    credit.record("site_a", "evaluation_served")
    granted = policy.decide_and_record(
        EvaluationRequest("site_a", "site_b", BUNDLE), credit
    )
    assert refused.allowed is False and granted.allowed is True

    decisions = credit.decisions()
    assert [d["reason_code"] for d in decisions] == [
        "insufficient_reciprocity",
        "granted",
    ]
    assert decisions[0]["request"]["requester"] == "site_a"
    assert credit.verify().ok is True
    # Recorded decisions are not credit events.
    assert len(credit.events()) == 1


def test_policy_configuration_is_validated_and_serialisable():
    with pytest.raises(ValueError):
        ReciprocityPolicy(min_evaluations_served=-1)
    policy = ReciprocityPolicy(min_evaluations_served=2, exempt_actors={"site_new"})
    assert policy.to_dict() == {
        "min_evaluations_served": 2,
        "exempt_actors": ["site_new"],
        "min_voting_weight": None,
        "allow_self_evaluation": False,
        "require_attested_service": False,
    }


# ------------------------------------- who may say that a service was performed


def _counterparty_ledger():
    """Return a credit ledger plus the key site_b signs its acknowledgements with."""
    signer = HmacSigner(b"site-b-key", key_label="site_b")
    registry = CounterpartyRegistry()
    registry.register("site_b", signer.verifier())
    credit = CreditLedger(
        InMemoryLedger(signer=HmacSigner(b"credit")),
        clock=_clock(),
        counterparties=registry,
    )
    return credit, signer


def test_a_site_can_append_its_own_evaluation_credit_and_the_chain_agrees():
    """The honest statement of what the hash chain does and does not do.

    Nothing authenticates an append, so a site can credit itself and the chain
    stays valid -- only a retroactive edit breaks it.
    """
    credit = CreditLedger(InMemoryLedger(signer=HmacSigner(b"credit")), clock=_clock())
    for _ in range(3):
        credit.record("site_a", "evaluation_served", institution="INST_A")
    assert credit.served_evaluations("site_a") == 3
    assert credit.verify().ok is True  # the chain does not object

    decision = ReciprocityPolicy(min_evaluations_served=2).decide(
        EvaluationRequest("site_a", "site_b", BUNDLE), credit
    )
    assert decision.allowed is True  # ...and the rule is satisfied by self-assertion
    assert credit.served_evaluations("site_a", attested_only=True) == 0


def test_attested_service_requires_the_counterpartys_signature():
    credit, signer = _counterparty_ledger()
    attestation = ServiceAttestation.create(
        signer,
        actor="site_a",
        counterparty="site_b",
        kind="evaluation_served",
        ref=BUNDLE,
        signed_at="2026-04-01T00:00:00.000000Z",
    )
    event = credit.record_evaluation_served(
        "site_a",
        counterparty="site_b",
        attestation=attestation,
        institution="INST_A",
        ref=BUNDLE,
    )
    assert (event.counterparty, event.attested) == ("site_b", True)
    assert credit.served_evaluations("site_a", attested_only=True) == 1

    unattested = credit.record_evaluation_served(
        "site_a", counterparty="site_b", institution="INST_A"
    )
    assert unattested.attested is False
    assert credit.served_evaluations("site_a") == 2
    assert credit.served_evaluations("site_a", attested_only=True) == 1
    assert credit.verify().ok is True


def test_a_forged_or_mismatched_attestation_is_refused():
    credit, signer = _counterparty_ledger()
    good = ServiceAttestation.create(
        signer,
        actor="site_a",
        counterparty="site_b",
        kind="evaluation_served",
        ref=BUNDLE,
        signed_at="2026-04-01T00:00:00.000000Z",
    )

    forged = ServiceAttestation(
        actor="site_a",
        counterparty="site_b",
        kind="evaluation_served",
        ref=BUNDLE,
        signed_at="2026-04-01T00:00:00.000000Z",
        signature="00" * 32,
    )
    with pytest.raises(AttestationRejectedError):
        credit.record_evaluation_served(
            "site_a", counterparty="site_b", attestation=forged, ref=BUNDLE
        )

    # Genuine, but for a different beneficiary.
    with pytest.raises(AttestationRejectedError):
        credit.record_evaluation_served(
            "site_c", counterparty="site_b", attestation=good, ref=BUNDLE
        )
    # Genuine, but for a different artefact.
    with pytest.raises(AttestationRejectedError):
        credit.record_evaluation_served(
            "site_a", counterparty="site_b", attestation=good, ref="other-bundle"
        )
    # Signed by a site whose key nobody registered.
    stranger = ServiceAttestation.create(
        HmacSigner(b"stranger"),
        actor="site_a",
        counterparty="site_x",
        kind="evaluation_served",
        ref=BUNDLE,
        signed_at="2026-04-01T00:00:00.000000Z",
    )
    with pytest.raises(AttestationRejectedError):
        credit.record_evaluation_served(
            "site_a", counterparty="site_x", attestation=stranger, ref=BUNDLE
        )
    assert len(credit.events()) == 0


def test_the_same_attestation_cannot_be_submitted_twice():
    """One signed acknowledgement must buy exactly one unit of attested service.

    An attestation verifies every time it is presented, so without a replay
    guard a single counterparty signature would manufacture an unbounded
    attested count -- and that count is the whole basis of
    ``require_attested_service=True``.
    """
    credit, signer = _counterparty_ledger()
    attestation = ServiceAttestation.create(
        signer,
        actor="site_a",
        counterparty="site_b",
        kind="evaluation_served",
        ref=BUNDLE,
        signed_at="2026-04-01T00:00:00.000000Z",
    )
    credit.record_evaluation_served(
        "site_a", counterparty="site_b", attestation=attestation, ref=BUNDLE
    )
    assert credit.served_evaluations("site_a", attested_only=True) == 1

    for _ in range(2):
        with pytest.raises(AttestationRejectedError) as exc:
            credit.record_evaluation_served(
                "site_a", counterparty="site_b", attestation=attestation, ref=BUNDLE
            )
        assert "already recorded" in str(exc.value)
    assert credit.served_evaluations("site_a", attested_only=True) == 1
    assert len(credit.events()) == 1

    # The refused submissions must also not have been able to buy a request.
    strict = ReciprocityPolicy(min_evaluations_served=2, require_attested_service=True)
    refusal = strict.decide(EvaluationRequest("site_a", "site_b", BUNDLE), credit)
    assert refusal.allowed is False
    assert refusal.observed == 1

    # A genuinely second evaluation is a second signature, and that is accepted.
    second = ServiceAttestation.create(
        signer,
        actor="site_a",
        counterparty="site_b",
        kind="evaluation_served",
        ref=BUNDLE,
        signed_at="2026-04-02T00:00:00.000000Z",
    )
    credit.record_evaluation_served(
        "site_a", counterparty="site_b", attestation=second, ref=BUNDLE
    )
    assert credit.served_evaluations("site_a", attested_only=True) == 2
    assert strict.decide(
        EvaluationRequest("site_a", "site_b", BUNDLE), credit
    ).allowed is True


def test_a_duplicated_attested_event_on_the_chain_is_not_counted_twice():
    """Defence in depth: the *count* is over distinct attestations, not events.

    The recording path refuses a resubmission, so this state can only arise
    from a chain written by something else. It still must not turn one signed
    acknowledgement into two units of standing.
    """
    credit, signer = _counterparty_ledger()
    attestation = ServiceAttestation.create(
        signer,
        actor="site_a",
        counterparty="site_b",
        kind="evaluation_served",
        ref=BUNDLE,
        signed_at="2026-04-01T00:00:00.000000Z",
    )
    for _ in range(3):
        credit.record(
            "site_a",
            "evaluation_served",
            ref=BUNDLE,
            counterparty="site_b",
            attested=True,
            attestation_signature=attestation.signature,
            attestation_key_id=attestation.key_id,
        )
    assert credit.served_evaluations("site_a") == 3
    assert credit.served_evaluations("site_a", attested_only=True) == 1


def test_evaluation_credit_always_names_a_second_party():
    credit, _ = _counterparty_ledger()
    with pytest.raises(AttestationRejectedError):
        credit.record_evaluation_served("site_a", counterparty="")
    with pytest.raises(AttestationRejectedError):
        credit.record_evaluation_served("site_a", counterparty="site_a")
    with pytest.raises(AttestationRejectedError):
        ServiceAttestation.create(
            HmacSigner(b"k"),
            actor="site_a",
            counterparty="site_a",
            kind="evaluation_served",
            ref=BUNDLE,
        )


def test_require_attestation_closes_the_self_award_path():
    credit, signer = _counterparty_ledger()
    strict = CreditLedger(
        credit.ledger,
        clock=_clock(),
        counterparties=credit.counterparties,
        require_attestation=True,
    )
    with pytest.raises(AttestationRejectedError):
        strict.record_evaluation_served("site_a", counterparty="site_b")

    strict.record_evaluation_served(
        "site_a",
        counterparty="site_b",
        attestation=ServiceAttestation.create(
            signer,
            actor="site_a",
            counterparty="site_b",
            kind="evaluation_served",
            ref=BUNDLE,
            signed_at="2026-04-01T00:00:00.000000Z",
        ),
        ref=BUNDLE,
    )
    assert strict.served_evaluations("site_a", attested_only=True) == 1


def test_reciprocity_can_be_made_to_ignore_self_asserted_service():
    credit, signer = _counterparty_ledger()
    credit.record("site_a", "evaluation_served", institution="INST_A")
    credit.record("site_a", "evaluation_served", institution="INST_A")
    request = EvaluationRequest("site_a", "site_b", BUNDLE)

    lenient = ReciprocityPolicy(min_evaluations_served=2)
    strict = ReciprocityPolicy(min_evaluations_served=2, require_attested_service=True)
    assert lenient.decide(request, credit).allowed is True
    refused = strict.decide(request, credit)
    assert refused.allowed is False
    assert refused.reason_code == "insufficient_reciprocity"
    assert refused.observed == 0

    for i in range(2):
        credit.record_evaluation_served(
            "site_a",
            counterparty="site_b",
            attestation=ServiceAttestation.create(
                signer,
                actor="site_a",
                counterparty="site_b",
                kind="evaluation_served",
                ref=f"{BUNDLE}#{i}",
                signed_at="2026-04-01T00:00:00.000000Z",
            ),
            ref=f"{BUNDLE}#{i}",
        )
    granted = strict.decide(request, credit)
    assert granted.allowed is True
    assert granted.observed == 2
    assert strict.to_dict()["require_attested_service"] is True


def test_service_attestation_round_trips():
    signer = HmacSigner(b"site-b-key", key_label="site_b")
    attestation = ServiceAttestation.create(
        signer,
        actor="site_a",
        counterparty="site_b",
        kind="evaluation_served",
        ref=BUNDLE,
        signed_at="2026-04-01T00:00:00.000000Z",
    )
    assert ServiceAttestation.from_dict(attestation.to_dict()) == attestation
    registry = CounterpartyRegistry()
    assert registry.verify(attestation) is False  # nobody registered yet
    registry.register("site_b", signer.verifier())
    assert registry.verify(attestation) is True
    assert "site_b" in registry and len(registry) == 1
