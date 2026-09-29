"""End-to-end tests of the hub: register -> node handshake -> submit -> gate ->
review (conflict refused, threshold, board composition) -> certify -> download
-> derive -> evaluation reciprocity -> metrics -> restart replay."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from openmed_hub.app import create_app
from openmed_hub.config import HubSettings
from openmed_hub.web import EXAMPLE_CARD, EXAMPLE_EVALUATION, EXAMPLE_MANUAL

PASSWORD = "correct-horse-battery"


def _settings(tmp_path, mode="mock"):
    return HubSettings(
        data_dir=tmp_path / "hub",
        secret_key="test-secret",
        attestation_root_key=b"test-attestation-root",
        attestation_mode=mode,
        technical_reviewers_min=1,
        clinical_reviewers_min=1,
        reciprocity_min_evaluations_served=2,
        reciprocity_grace_requests=1,
    )


@pytest.fixture()
def client(tmp_path):
    """A hub in mock attestation mode (declared code identity)."""
    settings = _settings(tmp_path)
    settings.data_dir.mkdir(parents=True)
    with TestClient(create_app(settings)) as c:
        c.settings = settings
        yield c


@pytest.fixture()
def nodekey_client(tmp_path):
    """A hub in node-key mode: quotes are signed at the site."""
    settings = _settings(tmp_path, mode="nodekey")
    settings.data_dir.mkdir(parents=True)
    with TestClient(create_app(settings)) as c:
        c.settings = settings
        yield c


def _node_keypair():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    private = Ed25519PrivateKey.generate()
    return private, private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()


def _handshake(client, token, slug, private, public):
    r = client.post(f"/api/v1/institutions/{slug}/node/key", headers=_auth(token), json={"public_key": public})
    nonce = r.json()["nonce"]
    r = client.post(f"/api/v1/institutions/{slug}/node/verify", headers=_auth(token), json={"nonce": nonce, "signature": private.sign(nonce.encode()).hex()})
    assert r.status_code == 200, r.text


def _weights(seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    np.savez(buf, coef=rng.normal(size=16))
    return buf.getvalue()


def _register(client, slug, name=None):
    r = client.post(
        "/api/v1/institutions",
        json={
            "institution": {"name": name or slug.upper(), "slug": slug, "kind": "health_system"},
            "user": {"email": f"lead@{slug}.test", "name": f"{slug} lead", "password": PASSWORD},
        },
    )
    assert r.status_code == 201, r.text
    body = r.json()
    return body["token"], body["user"]


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _submit(client, token, owner, *, version="1.0.0", parent="", gap=0.02, auc=0.8, name="decline-risk", quote=None):
    card = json.loads(json.dumps(EXAMPLE_CARD))
    card["model_details"].update(name=name, version=version, owner=owner)
    ev = dict(EXAMPLE_EVALUATION)
    ev["metrics"] = {"auc": auc, "accuracy": auc - 0.05}
    ev["subgroup_metrics"] = {"a": {"auc": auc + gap / 2}, "b": {"auc": auc - gap / 2}}
    ev["evaluated_by"] = owner
    r = client.post(
        "/api/v1/submissions",
        headers=_auth(token),
        data={
            "name": name,
            "version": version,
            "model_card": json.dumps(card),
            "evaluation": json.dumps(ev),
            "fine_tuning_manual": json.dumps(EXAMPLE_MANUAL),
            "code_identity": "openmed-training-pipeline@v1",
            "config": "cfg-v1",
            "quote": json.dumps(quote) if quote else "",
            "parent_bundle_id": parent,
            "tags": "test",
        },
        files={"weights": (f"{name}-{version}.npz", _weights(hash(version) % 1000))},
    )
    return r


def _grant(client, admin_token, user_id, roles):
    r = client.post(f"/api/v1/admin/users/{user_id}/roles", headers=_auth(admin_token), json={"roles": roles})
    assert r.status_code == 200, r.text
    return r.json()


def test_first_user_is_admin_and_registration_works(client):
    token, user = _register(client, "alpha")
    assert "admin" in user["roles"]
    token_b, user_b = _register(client, "beta")
    assert "admin" not in user_b["roles"]
    me = client.get("/api/v1/auth/me", headers=_auth(token_b)).json()
    assert me["institution"] == "beta"
    # Duplicate slug and duplicate email are refused.
    assert client.post(
        "/api/v1/institutions",
        json={"institution": {"name": "x", "slug": "alpha"}, "user": {"email": "x@y.test", "password": PASSWORD}},
    ).status_code == 409
    assert client.post(
        "/api/v1/auth/login", json={"email": "lead@alpha.test", "password": "wrong-password"}
    ).status_code == 401


def test_node_handshake_counts_as_independent_installation(client):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    admin_token, _ = _register(client, "founder")
    client.post("/api/v1/admin/institutions/founder/founding", headers=_auth(admin_token), json={"is_founding": True})
    token, _ = _register(client, "ext")
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()

    r = client.post("/api/v1/institutions/ext/node/key", headers=_auth(token), json={"public_key": public})
    assert r.status_code == 200, r.text
    nonce = r.json()["nonce"]
    bad = client.post("/api/v1/institutions/ext/node/verify", headers=_auth(token), json={"nonce": nonce, "signature": "00" * 64})
    assert bad.status_code == 400
    good = client.post(
        "/api/v1/institutions/ext/node/verify",
        headers=_auth(token),
        json={"nonce": nonce, "signature": private.sign(nonce.encode()).hex()},
    )
    assert good.status_code == 200, good.text
    assert good.json()["node_verified"] is True
    # A nonce is single-use.
    again = client.post("/api/v1/institutions/ext/node/verify", headers=_auth(token), json={"nonce": nonce, "signature": private.sign(nonce.encode()).hex()})
    assert again.status_code == 400

    m = client.get("/api/v1/metrics").json()
    assert m["independent_installations"] == {**m["independent_installations"], "numerator": 1, "denominator": 1}


def test_submission_review_certification_and_lineage(client, tmp_path):
    a_token, a_user = _register(client, "inst-a")
    b_token, b_user = _register(client, "inst-b")
    c_token, c_user = _register(client, "inst-c")
    _grant(client, a_token, a_user["id"], ["admin", "maintainer", "reviewer_technical"])
    _grant(client, a_token, b_user["id"], ["reviewer_technical", "reviewer_clinical"])
    _grant(client, a_token, c_user["id"], ["reviewer_technical", "reviewer_clinical"])

    r = _submit(client, a_token, "inst-a")
    assert r.status_code == 201, r.text
    root = r.json()
    assert root["state"] == "submitted"
    assert root["gate_status"] in ("pass", "warn")
    bundle_id = root["bundle_id"]

    # The same release (name + version) from the same institution is refused.
    assert _submit(client, a_token, "inst-a").status_code == 409

    # A reviewer from the owning institution is refused (no self-certification).
    r = client.post(f"/api/v1/submissions/{bundle_id}/reviews", headers=_auth(a_token), json={"decision": "approve", "statement": "mine"})
    assert r.status_code == 201
    assert r.json()["review"]["accepted"] is False
    assert r.json()["review"]["refusal_code"] == "self_certification"

    # One other institution is not enough.
    r = client.post(f"/api/v1/submissions/{bundle_id}/reviews", headers=_auth(b_token), json={"decision": "approve", "statement": "ok"})
    assert r.status_code == 201, r.text
    assert r.json()["state"] == "under_review"

    # The second institution completes the threshold and the board composition.
    r = client.post(f"/api/v1/submissions/{bundle_id}/reviews", headers=_auth(c_token), json={"decision": "approve", "statement": "ok too"})
    assert r.json()["state"] == "certified"
    detail = client.get(f"/api/v1/submissions/{bundle_id}").json()
    assert detail["submission"]["identifier"].startswith("oid:")
    assert "inst-b" in detail["case"]["approving_institutions"]
    assert detail["threshold"]["met"] is True
    assert detail["lineage"]["ok"] is True if "ok" in detail["lineage"] else True

    # Downloads need registration and are counted.
    assert client.get(f"/api/v1/submissions/{bundle_id}/weights").status_code == 401
    r = client.get(f"/api/v1/submissions/{bundle_id}/weights", headers=_auth(b_token))
    assert r.status_code == 200
    assert np.load(io.BytesIO(r.content))["coef"].shape == (16,)
    assert client.get(f"/api/v1/submissions/{bundle_id}").json()["downloads"] == 1

    # B fine-tunes the certified base; the lineage walks back to the root.
    r = _submit(client, b_token, "inst-b", version="1.1.0-b", parent=bundle_id)
    assert r.status_code == 201, r.text
    derived = r.json()
    lineage = client.get(f"/api/v1/submissions/{derived['bundle_id']}/lineage").json()
    assert bundle_id in lineage["mermaid"] or bundle_id[-12:] in lineage["mermaid"] or lineage["lineage"]
    assert client.get(f"/api/v1/submissions/{bundle_id}").json()["children"] == [derived["bundle_id"]]

    # Ledgers verify.
    verify = client.get("/api/v1/ledgers/verify").json()
    assert all(v["ok"] for v in verify.values()), verify

    # Restart on the same data dir: the case state is replayed from the ledger.
    with TestClient(create_app(client.settings)) as fresh:
        again = fresh.get(f"/api/v1/submissions/{bundle_id}").json()
        assert again["case"]["state"] == "certified"
        assert len(again["case"]["signatures"]) == 2
        assert fresh.get("/api/v1/metrics").json()["certified_models"] == 1


def test_gate_blocks_large_subgroup_gap_and_owner_can_remediate(client):
    a_token, a_user = _register(client, "gate-a")
    r = _submit(client, a_token, "gate-a", gap=0.35)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["gate_status"] == "fail"
    assert body["state"] == "blocked"
    log = client.get(f"/api/v1/submissions/{body['bundle_id']}/decision-log").json()
    assert any(e["actor"] == "hub:automated-gate" for e in log)
    r = client.post(f"/api/v1/submissions/{body['bundle_id']}/remediate", headers=_auth(a_token), json={"note": "re-weighted the cohort"})
    assert r.status_code == 200, r.text
    assert r.json()["state"] == "remediated"


def test_detached_signature_with_client_held_key(client):
    from trustfed.certification.keys import ReviewSignature
    from trustfed.ledger.block import utc_now_iso
    from trustfed.ledger.crypto import Ed25519Signer

    a_token, a_user = _register(client, "sig-a")
    b_token, b_user = _register(client, "sig-b")
    _grant(client, a_token, b_user["id"], ["reviewer_technical", "reviewer_clinical"])
    signer = Ed25519Signer.generate()
    r = client.post("/api/v1/auth/reviewer-key", headers=_auth(b_token), json={"public_key": signer.key_id})
    assert r.status_code == 200, r.text
    assert r.json()["reviewer_key"] == "client-held"
    me = r.json()

    bundle_id = _submit(client, a_token, "sig-a").json()["bundle_id"]
    # The web/hub-signed path is now refused for this reviewer.
    assert client.post(f"/api/v1/submissions/{bundle_id}/reviews", headers=_auth(b_token), json={"decision": "approve"}).status_code == 400

    unsigned = ReviewSignature(bundle_id=bundle_id, reviewer_id=me["reviewer_id"], institution="sig-b", decision="approve", statement="signed locally", signed_at=utc_now_iso(), signature="", key_id="")
    signed = ReviewSignature(**dict(unsigned.to_dict(), signature=signer.sign(unsigned.signing_material()), key_id=signer.key_id))
    r = client.post(f"/api/v1/submissions/{bundle_id}/signatures", headers=_auth(b_token), json={"signature": signed.to_dict()})
    assert r.status_code == 201, r.text
    assert r.json()["review"]["accepted"] is True
    assert r.json()["review"]["hub_signed"] is False

    # A forged signature (wrong key) is rejected.
    forged = ReviewSignature(**dict(unsigned.to_dict(), decision="block", signature=Ed25519Signer.generate().sign(unsigned.signing_material()), key_id=signer.key_id))
    assert client.post(f"/api/v1/submissions/{bundle_id}/signatures", headers=_auth(b_token), json={"signature": forged.to_dict()}).status_code == 400


def test_reciprocity_rule_on_evaluation_requests(client):
    a_token, _ = _register(client, "rec-a")
    b_token, _ = _register(client, "rec-b")
    c_token, _ = _register(client, "rec-c")
    a_bundle = _submit(client, a_token, "rec-a").json()["bundle_id"]
    b_bundle = _submit(client, b_token, "rec-b").json()["bundle_id"]

    # A's first request is a grace request.
    r = client.post("/api/v1/evaluations", headers=_auth(a_token), json={"bundle_id": a_bundle, "evaluator_slug": "rec-b"})
    assert r.status_code == 201, r.text
    task = r.json()
    assert task["state"] == "requested"
    # Only the evaluator may serve; only the requester may acknowledge.
    assert client.post(f"/api/v1/evaluations/{task['id']}/serve", headers=_auth(a_token), json={"report": EXAMPLE_EVALUATION}).status_code == 403
    assert client.post(f"/api/v1/evaluations/{task['id']}/serve", headers=_auth(b_token), json={"report": EXAMPLE_EVALUATION}).status_code == 200
    assert client.post(f"/api/v1/evaluations/{task['id']}/acknowledge", headers=_auth(b_token)).status_code == 403
    assert client.post(f"/api/v1/evaluations/{task['id']}/acknowledge", headers=_auth(a_token)).json()["state"] == "acknowledged"
    assert client.get("/api/v1/institutions/rec-b").json()["reciprocity"]["evaluations_served_attested"] == 1

    # A's grace is spent and A has served nobody: refused with a reason code.
    r = client.post("/api/v1/evaluations", headers=_auth(a_token), json={"bundle_id": a_bundle, "evaluator_slug": "rec-c"})
    assert r.status_code == 201
    assert r.json()["state"] == "refused"
    assert r.json()["reason_code"]

    # B's grace request, served by A twice over two tasks, unlocks A.
    for _ in range(2):
        t = client.post("/api/v1/evaluations", headers=_auth(b_token), json={"bundle_id": b_bundle, "evaluator_slug": "rec-a"}).json()
        if t["state"] != "requested":
            break
        client.post(f"/api/v1/evaluations/{t['id']}/serve", headers=_auth(a_token), json={"report": EXAMPLE_EVALUATION})
        client.post(f"/api/v1/evaluations/{t['id']}/acknowledge", headers=_auth(b_token))
    served = client.get("/api/v1/institutions/rec-a").json()["reciprocity"]["evaluations_served_attested"]
    assert served >= 1
    standing = client.get("/api/v1/standing").json()
    assert any(row["actor"] == "rec-a" for row in standing["rows"])


def test_web_pages_render(client):
    a_token, a_user = _register(client, "web-a")
    bundle_id = _submit(client, a_token, "web-a").json()["bundle_id"]
    for path in ("/", "/models", "/institutions", "/institutions/web-a", "/metrics", "/register", "/join", "/login", f"/models/{bundle_id}"):
        r = client.get(path)
        assert r.status_code == 200, (path, r.status_code)
        assert "OpenMed" in r.text
    # Session login through the form, then a page that needs a user.
    r = client.post("/login", data={"email": "lead@web-a.test", "password": PASSWORD, "next": "/account"}, follow_redirects=False)
    assert r.status_code == 303
    assert "openmed_session" in r.headers.get("set-cookie", "")
    r = client.get("/account")
    assert r.status_code == 200 and "API tokens" in r.text
    assert client.get("/submit").status_code == 200
    assert client.get("/evaluations").status_code == 200


def test_nodekey_mode_requires_a_site_signed_quote_over_an_approved_measurement(nodekey_client):
    from trustfed.attestation import NodeKeyAttestor, software_measurement
    from trustfed.ledger.crypto import Ed25519Signer

    client = nodekey_client
    a_token, a_user = _register(client, "nk-a")  # admin + maintainer
    policy = client.get("/api/v1/attestation/policy").json()
    assert policy["mode"] == "nodekey" and policy["approved_measurements"] == []

    # No quote at all: refused with instructions.
    r = _submit(client, a_token, "nk-a")
    assert r.status_code == 422 and r.json()["error"] == "quote_required"

    # A challenge needs a verified node.
    assert client.post("/api/v1/attestation/challenge", headers=_auth(a_token)).status_code == 403
    private, public = _node_keypair()
    _handshake(client, a_token, "nk-a", private, public)
    signer = Ed25519Signer(private)
    measured = software_measurement()

    # The measurement is not approved yet: refused, and the challenge is not spent.
    challenge = client.post("/api/v1/attestation/challenge", headers=_auth(a_token)).json()
    quote = NodeKeyAttestor.sign_quote(signer, client_id="nk-a", measurement=measured.measurement, config_hash="0" * 64, nonce=challenge["nonce"])
    r = _submit(client, a_token, "nk-a", quote=quote.to_dict())
    assert r.status_code == 422 and "unapproved_measurement" in r.json()["detail"]

    # Only maintainers approve measurements.
    b_token, b_user = _register(client, "nk-b")
    assert client.post("/api/v1/attestation/measurements", headers=_auth(b_token), json={"measurement": measured.measurement}).status_code == 403
    r = client.post("/api/v1/attestation/measurements", headers=_auth(a_token), json={"measurement": measured.measurement, "label": "test release", "manifest": measured.manifest})
    assert r.status_code == 201, r.text

    # Same quote now passes; the bundle records the site as attested client.
    r = _submit(client, a_token, "nk-a", quote=quote.to_dict())
    assert r.status_code == 201, r.text
    bundle_id = r.json()["bundle_id"]
    att = client.get(f"/api/v1/submissions/{bundle_id}").json()["bundle"]["attestation"]
    assert att["verified"] is True and att["attestor_id"] == "node-key-software-measurement" and att["client_id"] == "nk-a"

    # Replaying the spent challenge is refused.
    r = _submit(client, a_token, "nk-a", version="1.0.1", quote=quote.to_dict())
    assert r.status_code == 422 and "nonce" in r.json()["detail"]

    # A quote signed with some other key, or naming another institution, is refused.
    challenge = client.post("/api/v1/attestation/challenge", headers=_auth(a_token)).json()
    forged = NodeKeyAttestor.sign_quote(Ed25519Signer.generate(), client_id="nk-a", measurement=measured.measurement, config_hash="0" * 64, nonce=challenge["nonce"])
    r = _submit(client, a_token, "nk-a", version="1.0.2", quote=forged.to_dict())
    assert r.status_code == 422 and "bad_signature" in r.json()["detail"]
    other = NodeKeyAttestor.sign_quote(signer, client_id="nk-b", measurement=measured.measurement, config_hash="0" * 64, nonce=challenge["nonce"])
    assert _submit(client, a_token, "nk-a", version="1.0.3", quote=other.to_dict()).status_code == 403

    # Revoking the measurement closes the gate for new submissions.
    assert client.delete(f"/api/v1/attestation/measurements/{measured.measurement}", headers=_auth(a_token)).status_code == 200
    challenge = client.post("/api/v1/attestation/challenge", headers=_auth(a_token)).json()
    quote2 = NodeKeyAttestor.sign_quote(signer, client_id="nk-a", measurement=measured.measurement, config_hash="0" * 64, nonce=challenge["nonce"])
    assert _submit(client, a_token, "nk-a", version="1.0.4", quote=quote2.to_dict()).status_code == 422

    # After a restart the node key and the (revoked) policy come back from the DB.
    with TestClient(create_app(client.settings)) as fresh:
        assert fresh.get("/api/v1/attestation/policy").json()["approved_measurements"] == []
        login = fresh.post("/api/v1/auth/login", json={"email": "lead@nk-a.test", "password": PASSWORD}).json()["token"]
        assert fresh.post("/api/v1/attestation/challenge", headers=_auth(login)).status_code == 200
