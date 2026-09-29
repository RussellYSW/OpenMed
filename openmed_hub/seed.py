"""Synthetic demo data for a fresh hub: four institutions, reviewers, one
certified model, one derived model under review, one evaluation exchange.

Nothing here is real: names are the founding sites of the demonstration,
weights are random numbers, metrics come from a seeded RNG.
"""

from __future__ import annotations

import io
import json
from typing import Any, Dict

import numpy as np

from openmed_hub.db import ROLE_ADMIN, ROLE_MAINTAINER, ROLE_REVIEWER_CLINICAL, ROLE_REVIEWER_TECHNICAL
from openmed_hub.services import HubError, HubServices
from openmed_hub.web import EXAMPLE_CARD, EXAMPLE_EVALUATION, EXAMPLE_MANUAL

DEMO_PASSWORD = "openmed-demo-2026"


def _weights(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    np.savez(buf, coef=rng.normal(size=24), intercept=rng.normal(size=1))
    return buf.getvalue()


def _card(name: str, version: str, owner: str, auc: float) -> str:
    card = json.loads(json.dumps(EXAMPLE_CARD))
    card["model_details"].update(name=name, version=version, owner=owner)
    card["quantitative_analyses"] = {"auc": auc, "accuracy": round(auc - 0.05, 3)}
    return json.dumps(card)


def _evaluation(owner: str, auc: float, gap: float = 0.03) -> str:
    ev = dict(EXAMPLE_EVALUATION)
    ev["metrics"] = {"auc": auc, "accuracy": round(auc - 0.05, 3)}
    ev["subgroup_metrics"] = {"age<65": {"auc": round(auc + gap / 2, 3)}, "age>=65": {"auc": round(auc - gap / 2, 3)}}
    ev["evaluated_by"] = owner
    return json.dumps(ev)


def _node(svc: HubServices, institution):
    """Stand up a node in-process: keypair, handshake; returns the signer."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    from trustfed.ledger.crypto import Ed25519Signer

    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    nonce = svc.begin_node_registration(institution, public)
    svc.complete_node_registration(institution, nonce, private.sign(nonce.encode("utf-8")).hex())
    return Ed25519Signer(private)


def _quote(svc: HubServices, user, signer, measurement: str) -> str:
    """What `openmed submit` does at a site: challenge, sign, attach."""
    from trustfed.attestation import NodeKeyAttestor

    challenge = svc.issue_challenge(user)
    quote = NodeKeyAttestor.sign_quote(
        signer, client_id=challenge["client_id"], measurement=measurement,
        config_hash="0" * 64, nonce=challenge["nonce"],
    )
    return json.dumps(quote.to_dict())


def seed_demo(svc: HubServices) -> Dict[str, Any]:
    """Populate ``svc``'s hub; idempotent enough to refuse a second run."""
    if svc.list_institutions():
        raise HubError(409, "not_empty", "seed-demo needs an empty hub")

    sites = [
        ("University of Miami", "umiami", "medical_school", True),
        ("University of Maryland Medical System", "umaryland", "health_system", True),
        ("VA Maryland Health Care System", "va-maryland", "health_system", True),
        ("MedStar Health", "medstar", "health_system", True),
        ("Example External Neurology Lab", "external-neuro-lab", "research_lab", False),
    ]
    users: Dict[str, Any] = {}
    institutions: Dict[str, Any] = {}
    signers: Dict[str, Any] = {}
    for name, slug, kind, founding in sites:
        institution, user = svc.register_institution(
            name=name, slug=slug, kind=kind, country="US",
            admin_email=f"lead@{slug}.example", admin_name=f"{name} lead", admin_password=DEMO_PASSWORD,
            is_founding=founding,
        )
        institutions[slug] = institution
        users[slug] = user
        signers[slug] = _node(svc, institution)

    svc.grant_roles(users["umiami"], [ROLE_ADMIN, ROLE_MAINTAINER, ROLE_REVIEWER_TECHNICAL])

    # The maintainer approves the measurement of the installed pipeline, which
    # is exactly what `openmed measure` prints at a site running this release.
    from trustfed.attestation import software_measurement

    measured = software_measurement()  # identical to what `openmed measure` prints at a site
    svc.approve_measurement(user=users["umiami"], measurement=measured.measurement,
                            label="trustfed as installed for the demo", manifest=measured.manifest)
    nodekey = svc.settings.attestation_mode == "nodekey"

    def attest_kwargs(slug: str) -> Dict[str, Any]:
        if nodekey:
            return {"quote": _quote(svc, users[slug], signers[slug], measured.measurement)}
        return {"code_identity": "openmed-training-pipeline@v1", "config": "training-config-v1"}
    for slug in ("umaryland", "va-maryland", "medstar"):
        svc.grant_roles(users[slug], [ROLE_REVIEWER_TECHNICAL, ROLE_REVIEWER_CLINICAL])
    reviewer2 = svc.register_user(
        email="clinician@umaryland.example", name="UMaryland clinical reviewer", password=DEMO_PASSWORD,
        institution=institutions["umaryland"],
    )
    svc.grant_roles(reviewer2, [ROLE_REVIEWER_CLINICAL])

    # 1. Miami publishes a root model; Maryland and the VA certify it.
    root = svc.submit(
        user=users["umiami"], name="cognitive-decline-risk", version="1.0.0",
        weights=_weights(1), weights_format="npz",
        model_card=_card("cognitive-decline-risk", "1.0.0", "umiami", 0.81),
        evaluation=_evaluation("umiami", 0.81), manual=json.dumps(EXAMPLE_MANUAL),
        tags=["neurology", "cognitive-decline", "demo"], **attest_kwargs("umiami"),
    )
    svc.review(submission=root, user=users["umiami"], decision="approve", statement="looks good")  # refused: self-certification
    svc.review(submission=root, user=users["umaryland"], decision="approve", statement="protocol and card check out; subgroup gap within tolerance")
    svc.review(submission=root, user=users["va-maryland"], decision="approve", statement="re-ran the evaluation on our cohort: AUC 0.79")

    # 2. MedStar fine-tunes the certified base; the case is under review.
    derived = svc.submit(
        user=users["medstar"], name="cognitive-decline-risk", version="1.1.0-medstar",
        weights=_weights(2), weights_format="npz",
        model_card=_card("cognitive-decline-risk", "1.1.0-medstar", "medstar", 0.83),
        evaluation=_evaluation("medstar", 0.83), manual=json.dumps(EXAMPLE_MANUAL),
        parent_bundle_id=root.bundle_id, tags=["derived", "demo"], **attest_kwargs("medstar"),
    )
    svc.review(submission=derived, user=users["umaryland"], decision="approve", statement="derived model keeps the parent's calibration")

    # 3. The external lab submits a model with a large subgroup gap: the gate blocks it.
    blocked = svc.submit(
        user=users["external-neuro-lab"], name="eeg-decline-screen", version="0.9.0",
        weights=_weights(3), weights_format="npz",
        model_card=_card("eeg-decline-screen", "0.9.0", "external-neuro-lab", 0.77),
        evaluation=_evaluation("external-neuro-lab", 0.77, gap=0.30), **attest_kwargs("external-neuro-lab"),
    )

    # 4. An evaluation exchange: Miami asks Maryland (grace request), Maryland serves, Miami acknowledges.
    task = svc.request_evaluation(user=users["umiami"], bundle_id=root.bundle_id, evaluator_slug="umaryland", detail="older adults subgroup")
    if task.state == "requested":
        svc.serve_evaluation(task=task, user=users["umaryland"], report=_evaluation("umaryland", 0.79))
        svc.acknowledge_evaluation(task=task, user=users["umiami"])

    return {
        "attestation_mode": svc.settings.attestation_mode,
        "approved_measurement": measured.measurement,
        "password_for_all_demo_users": DEMO_PASSWORD,
        "users": {slug: u.email for slug, u in users.items()},
        "admin": users["umiami"].email,
        "root_model": {"bundle_id": root.bundle_id, "state": root.state, "identifier": root.identifier},
        "derived_model": {"bundle_id": derived.bundle_id, "state": derived.state},
        "gate_blocked_model": {"bundle_id": blocked.bundle_id, "state": blocked.state, "gate": blocked.gate_status},
        "evaluation_task": svc.evaluation_summary(task),
    }


__all__ = ["seed_demo", "DEMO_PASSWORD"]
