"""Account security: 2FA, lockout, CSRF, sessions, membership approval, headers."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from openmed_hub.app import create_app
from openmed_hub.config import HubSettings
from openmed_hub.db import ensure_columns, make_engine
from openmed_hub.totp import totp

PASSWORD = "correct-horse-battery"


def _app(tmp_path, **overrides):
    settings = HubSettings(
        data_dir=tmp_path / "hub", secret_key="s", attestation_root_key=b"r", attestation_mode="mock",
        technical_reviewers_min=1, clinical_reviewers_min=1, **overrides,
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings, create_app(settings)


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _register(client, slug):
    r = client.post("/api/v1/institutions", json={"institution": {"name": slug, "slug": slug}, "user": {"email": f"lead@{slug}.test", "password": PASSWORD}})
    assert r.status_code == 201, r.text
    return r.json()["token"], r.json()["user"]


def test_weak_passwords_are_refused(tmp_path):
    _, app = _app(tmp_path)
    with TestClient(app) as c:
        for bad in ("short", "password1234", "lead@x.test-ish", "aaaaaaaaaaaa"):
            r = c.post("/api/v1/institutions", json={"institution": {"name": "x", "slug": "x"}, "user": {"email": "lead@x.test", "password": bad}})
            assert r.status_code == 422, bad
            assert r.json()["error"] == "weak_password"


def test_two_factor_enrolment_login_and_recovery(tmp_path):
    settings, app = _app(tmp_path, require_2fa_roles=())
    with TestClient(app) as c:
        token, user = _register(c, "tf")
        enrol = c.post("/api/v1/auth/2fa/enroll", headers=_auth(token)).json()
        assert enrol["otpauth_uri"].startswith("otpauth://totp/")
        assert c.post("/api/v1/auth/2fa/confirm", headers=_auth(token), json={"code": "000000"}).status_code == 400
        r = c.post("/api/v1/auth/2fa/confirm", headers=_auth(token), json={"code": totp(enrol["secret"])})
        assert r.status_code == 200, r.text
        codes = r.json()["recovery_codes"]
        assert len(codes) == 10 and r.json()["user"]["totp_enabled"] is True

        # Password alone is no longer enough.
        r = c.post("/api/v1/auth/login", json={"email": "lead@tf.test", "password": PASSWORD})
        assert r.status_code == 401 and r.json()["error"] == "totp_required"
        r = c.post("/api/v1/auth/login", json={"email": "lead@tf.test", "password": PASSWORD, "totp": "123456"})
        assert r.status_code == 401 and r.json()["error"] == "bad_totp"
        r = c.post("/api/v1/auth/login", json={"email": "lead@tf.test", "password": PASSWORD, "totp": totp(enrol["secret"])})
        assert r.status_code == 200
        # A recovery code works once.
        assert c.post("/api/v1/auth/login", json={"email": "lead@tf.test", "password": PASSWORD, "totp": codes[0]}).status_code == 200
        assert c.post("/api/v1/auth/login", json={"email": "lead@tf.test", "password": PASSWORD, "totp": codes[0]}).status_code == 401

        # Web two-step: password -> /login/2fa -> code.
        page = c.get("/login")
        csrf = c.cookies.get("openmed_csrf")
        r = c.post("/login", data={"email": "lead@tf.test", "password": PASSWORD, "next": "/account", "csrf": csrf}, follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"].startswith("/login/2fa")
        assert "openmed_session" not in c.cookies
        r = c.post("/login/2fa", data={"code": totp(enrol["secret"]), "next": "/account", "csrf": csrf}, follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"].startswith("/account")
        assert c.get("/account").status_code == 200

        # Disabling needs password and code.
        assert c.post("/api/v1/auth/2fa/disable", headers=_auth(token), json={"password": "nope", "code": totp(enrol["secret"])}).status_code == 401
        assert c.post("/api/v1/auth/2fa/disable", headers=_auth(token), json={"password": PASSWORD, "code": totp(enrol["secret"])}).json()["totp_enabled"] is False


def test_privileged_roles_require_two_factor(tmp_path):
    settings, app = _app(tmp_path)  # default policy: reviewers/maintainers/admins need 2FA
    with TestClient(app) as c:
        a_token, a_user = _register(c, "pa")
        b_token, b_user = _register(c, "pb")
        # The admin cannot even grant roles without 2FA.
        r = c.post(f"/api/v1/admin/users/{b_user['id']}/roles", headers=_auth(a_token), json={"roles": ["reviewer_technical", "reviewer_clinical"]})
        assert r.status_code == 403 and r.json()["error"] == "totp_required"
        secret = c.post("/api/v1/auth/2fa/enroll", headers=_auth(a_token)).json()["secret"]
        c.post("/api/v1/auth/2fa/confirm", headers=_auth(a_token), json={"code": totp(secret)})
        assert c.post(f"/api/v1/admin/users/{b_user['id']}/roles", headers=_auth(a_token), json={"roles": ["reviewer_technical", "reviewer_clinical"]}).status_code == 200
        # B is a reviewer now but has no second factor: signing is refused.
        from tests.test_hub_flow import _submit

        bundle_id = _submit(c, a_token, "pa").json()["bundle_id"]
        r = c.post(f"/api/v1/submissions/{bundle_id}/reviews", headers=_auth(b_token), json={"decision": "approve"})
        assert r.status_code == 403 and r.json()["error"] == "totp_required"


def test_lockout_after_repeated_failures(tmp_path):
    settings, app = _app(tmp_path, lockout_threshold=3)
    with TestClient(app) as c:
        _register(c, "lk")
        for _ in range(3):
            assert c.post("/api/v1/auth/login", json={"email": "lead@lk.test", "password": "wrong-password-1"}).status_code == 401
        r = c.post("/api/v1/auth/login", json={"email": "lead@lk.test", "password": PASSWORD})
        assert r.status_code == 423 and r.json()["error"] == "account_locked"
        audit = c.get("/api/v1/admin/audit", headers=_auth(_register(c, "lk2")[0]))  # not admin
        assert audit.status_code == 403


def test_password_change_signs_out_other_sessions(tmp_path):
    settings, app = _app(tmp_path)
    with TestClient(app) as c:
        token, _ = _register(c, "pw")
        c.get("/login")
        csrf = c.cookies.get("openmed_csrf")
        c.post("/login", data={"email": "lead@pw.test", "password": PASSWORD, "next": "/", "csrf": csrf})
        old_session = c.cookies.get("openmed_session")
        assert c.get("/account").status_code == 200
        r = c.post("/api/v1/auth/password", headers=_auth(token), json={"current_password": PASSWORD, "new_password": "another-good-passphrase"})
        assert r.status_code == 200, r.text
        c.cookies.set("openmed_session", old_session)
        assert c.get("/account", follow_redirects=False).status_code == 303  # bounced to login
        assert c.post("/api/v1/auth/login", json={"email": "lead@pw.test", "password": PASSWORD}).status_code == 401
        assert c.post("/api/v1/auth/login", json={"email": "lead@pw.test", "password": "another-good-passphrase"}).status_code == 200


def test_csrf_and_security_headers(tmp_path):
    settings, app = _app(tmp_path)
    with TestClient(app) as c:
        _register(c, "cs")
        r = c.post("/login", data={"email": "lead@cs.test", "password": PASSWORD, "next": "/"}, follow_redirects=False)
        assert r.status_code == 303 and "token" in r.headers["location"]  # refused: no token
        r = c.get("/")
        assert r.headers["X-Frame-Options"] == "DENY"
        assert "default-src 'self'" in r.headers["Content-Security-Policy"]
        assert r.headers["X-Content-Type-Options"] == "nosniff"
        csrf = c.cookies.get("openmed_csrf")
        r = c.post("/login", data={"email": "lead@cs.test", "password": PASSWORD, "next": "/", "csrf": csrf}, follow_redirects=False)
        assert r.status_code == 303 and "signed" in r.headers["location"]


def test_membership_requires_invite_domain_or_approval(tmp_path):
    settings, app = _app(tmp_path, require_2fa_roles=())
    with TestClient(app) as c:
        lead_token, lead = _register(c, "mem")
        assert lead["institution_admin"] is True
        # A stranger joining without proof is pending and cannot act.
        r = c.post("/api/v1/auth/join", json={"institution_slug": "mem", "email": "stranger@elsewhere.test", "password": PASSWORD})
        assert r.status_code == 201 and r.json()["user"]["membership"] == "pending"
        stranger_token = c.post("/api/v1/auth/login", json={"email": "stranger@elsewhere.test", "password": PASSWORD}).json()["token"]
        from tests.test_hub_flow import _submit

        r = _submit(c, stranger_token, "mem")
        assert r.status_code == 403 and r.json()["error"] == "membership_pending"
        # Wrong invite code is refused outright; the right one approves immediately.
        settings_r = c.post("/api/v1/institutions/mem/settings", headers=_auth(lead_token), json={"allowed_email_domains": "mem.test"})
        code = settings_r.json()["invite_code"]
        assert c.post("/api/v1/auth/join", json={"institution_slug": "mem", "email": "x@y.test", "password": PASSWORD, "invite_code": "NOPE"}).status_code == 403
        r = c.post("/api/v1/auth/join", json={"institution_slug": "mem", "email": "invited@y.test", "password": PASSWORD, "invite_code": code})
        assert r.json()["user"]["membership"] == "active"
        # An approved email domain also works.
        r = c.post("/api/v1/auth/join", json={"institution_slug": "mem", "email": "colleague@mem.test", "password": PASSWORD})
        assert r.json()["user"]["membership"] == "active"
        # The institution admin approves the stranger; then they can act.
        members = c.get("/api/v1/institutions/mem/members", headers=_auth(lead_token)).json()
        stranger = next(m for m in members if m["email"] == "stranger@elsewhere.test")
        assert c.post(f"/api/v1/institutions/mem/members/{stranger['id']}", headers=_auth(stranger_token), json={"approve": True}).status_code == 403
        assert c.post(f"/api/v1/institutions/mem/members/{stranger['id']}", headers=_auth(lead_token), json={"approve": True}).json()["membership"] == "active"
        assert _submit(c, stranger_token, "mem").status_code == 201
        audit = c.get("/api/v1/admin/audit", headers=_auth(lead_token)).json()
        assert any(e["action"] == "member_approved" for e in audit)


def test_ensure_columns_adds_missing_columns(tmp_path):
    from sqlalchemy import text

    engine = make_engine(tmp_path / "old.db")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY, email VARCHAR(200), name VARCHAR(200), password_hash VARCHAR(300), institution_id INTEGER, roles VARCHAR(200), is_active BOOLEAN, created_at DATETIME)"))
    added = ensure_columns(engine)
    assert "users.totp_secret" in added and "users.security_stamp" in added
    assert ensure_columns(engine) == []
