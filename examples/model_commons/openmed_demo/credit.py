"""Section 5: credit, citable attribution, standing and the reciprocity rule.

Evaluation credit is recorded through the counterparty path, so each entry names
the site that was evaluated and carries that site's signed acknowledgement. The
reciprocity policy is configured to count only those, which is the difference
between a rule a site can satisfy by asserting things about itself and one that
needs a second party's word.
"""

from __future__ import annotations

from typing import Dict

from trustfed.incentives import (
    Attribution,
    CreditLedger,
    EvaluationRequest,
    ReciprocityPolicy,
    ReleaseAttribution,
    ServiceAttestation,
)
from trustfed.ledger import HmacSigner

from openmed_demo.fixtures import YEAR


def _serve(
    credit: CreditLedger,
    signers: Dict[str, HmacSigner],
    actor: str,
    counterparty: str,
    institution: str,
    ref: str,
) -> None:
    """Record an evaluation ``actor`` served for ``counterparty``, with its signature."""
    attestation = ServiceAttestation.create(
        signers[counterparty],
        actor=actor,
        counterparty=counterparty,
        kind="evaluation_served",
        ref=ref,
    )
    credit.record_evaluation_served(
        actor,
        counterparty=counterparty,
        attestation=attestation,
        institution=institution,
        ref=ref,
    )


def credit_and_reciprocity(
    credit: CreditLedger, signers: Dict[str, HmacSigner], bundle_id: str
) -> None:
    """Record contributions, mint a citation, and exercise the reciprocity rule."""
    credit.record("site_a", "data_contribution", quantity=3, institution="INST_A")
    credit.record("site_a", "training_round", quantity=5, institution="INST_A")
    credit.record("site_b", "training_round", quantity=5, institution="INST_B")
    _serve(credit, signers, "site_b", "site_a", "INST_B", bundle_id)
    _serve(credit, signers, "site_c", "site_a", "INST_C", bundle_id)
    credit.record("site_c", "review_signed", quantity=1, institution="INST_C")

    release = ReleaseAttribution.for_bundle(
        bundle_id,
        title="PD rapid-decline classifier (synthetic demo)",
        year=YEAR,
        version="1.0.0",
        contributors=[
            Attribution("site_a", "data and training", "INST_A", share=0.5),
            Attribution("site_b", "evaluation", "INST_B", share=0.3),
            Attribution("site_c", "review", "INST_C", share=0.2),
        ],
    )
    credit.record_release(release)
    print(f"citation                : {release.to_citation()}")

    policy = ReciprocityPolicy(min_evaluations_served=2, require_attested_service=True)
    request = EvaluationRequest("site_a", "site_b", bundle_id)
    refused = policy.decide_and_record(request, credit)
    print(f"evaluation request      : allowed={refused.allowed} ({refused.reason_code})")
    print(f"  reason                : {refused.detail}")

    # A site cannot clear the bar by asserting service; the counterparty signs.
    unattested = credit.record_evaluation_served(
        "site_a", counterparty="site_b", institution="INST_A"
    )
    still_refused = policy.decide(request, credit)
    print(
        f"unattested self-claim   : recorded (attested={unattested.attested}), "
        f"still allowed={still_refused.allowed}"
    )

    _serve(credit, signers, "site_a", "site_b", "INST_A", bundle_id)
    _serve(credit, signers, "site_a", "site_c", "INST_A", bundle_id)
    granted = policy.decide_and_record(request, credit)
    print(f"after reciprocating     : allowed={granted.allowed} ({granted.reason_code})")
    print()
    print(credit.standing_report().to_markdown().rstrip())


__all__ = ["credit_and_reciprocity"]
