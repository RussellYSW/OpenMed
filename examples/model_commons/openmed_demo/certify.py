"""Section 2: multi-institution certification, refusals included.

The case is opened with the *bundle*, not a bundle id plus a declared owner, so
the owner institution is read from ``published_by`` -- content the bundle id
commits to -- rather than from the submitting site's word about itself.
"""

from __future__ import annotations

from typing import Any

from trustfed.certification import (
    CertificationAuthority,
    ConflictOfInterestError,
    ThresholdNotMetError,
)

from openmed_demo.fixtures import make_manual


def certify(authority: CertificationAuthority, bundle: Any) -> None:
    """Run the multi-institution certification, refusals included."""
    case = authority.submit(
        bundle,
        submitted_by="site_a",
        manual=make_manual("INST_A").to_dict(),
    )
    bundle_id = case.bundle_id
    print(f"owner institution       : {case.facts.owner_institution} (derived)")
    print(f"  self-asserted facts   : {case.facts.self_asserted}")

    try:
        authority.review(bundle_id, "rev_a1", "approve", statement="ship it")
    except ConflictOfInterestError as exc:
        print(f"self-certification      : refused ({exc.reason_code})")

    authority.review(bundle_id, "rev_b1", "approve", statement="protocol checks out")
    try:
        authority.certify(bundle_id)
    except ThresholdNotMetError as exc:
        print(f"one institution only    : refused ({exc})")

    authority.review(bundle_id, "rev_c1", "approve", statement="subgroups reviewed")
    certified = authority.certify(bundle_id)
    print(
        f"certified base          : {certified.state.value} by "
        f"{', '.join(certified.approving_institutions())}"
    )
    print(f"decision log entries    : {len(authority.decision_log(bundle_id))}")
    print(f"decision log verifies   : {authority.verify_log().ok}")
    print(
        "note                    : review() signs with a key the authority holds; "
        "submit_signature() is the path for remote reviewers"
    )


__all__ = ["certify"]
