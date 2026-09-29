"""Workflow services: the commons' rules applied to the trust plane.

Everything a route handler does goes through :class:`HubServices`, so the web
UI, the JSON API and the CLI enforce the same rules: who may submit, what the
automated gate blocks, which reviewers may sign, when a model becomes a
certified base, who may download, and when an evaluation request is refused
under the reciprocity rule.
"""

from __future__ import annotations

import io
import json
import re
import secrets
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from trustfed.attestation import Quote
from trustfed.certification import FineTuningManual
from trustfed.certification.errors import (
    CertificationError,
    ConflictOfInterestError,
    InvalidTransitionError,
    SignatureRejectedError,
    ThresholdNotMetError,
    UnknownCaseError,
    UnknownReviewerError,
)
from trustfed.certification.keys import ReviewSignature
from trustfed.certification.states import CertificationState
from trustfed.incentives import (
    Attribution,
    EvaluationRequest,
    KIND_EVALUATION_SERVED,
    ReleaseAttribution,
    ServiceAttestation,
)
from trustfed.incentives.errors import AttestationRejectedError
from trustfed.quality import QualityAnalyzer, QualityEvidence
from trustfed.registry import (
    BundleNotFoundError,
    DuplicateBundleError,
    EvaluationReport,
    ModelCard,
    RegistryError,
    ValidationError,
    WeightsRef,
)

from openmed_hub.config import HubSettings
from openmed_hub.db import (
    ALL_ROLES,
    INSTITUTION_KINDS,
    REVIEWER_ROLES,
    ROLE_ADMIN,
    ROLE_CONTRIBUTOR,
    ROLE_MAINTAINER,
    ROLE_REVIEWER_CLINICAL,
    ROLE_REVIEWER_TECHNICAL,
    MEMBERSHIP_ACTIVE,
    MEMBERSHIP_PENDING,
    MEMBERSHIP_REJECTED,
    ApiToken,
    ApprovedMeasurement,
    AuditEvent,
    Download,
    EvaluationTask,
    Institution,
    NodeChallenge,
    Review,
    Submission,
    User,
    utcnow,
)
from openmed_hub.security import (
    hash_password,
    new_stamp,
    new_token,
    password_problem,
    token_hash,
    verify_password,
)
from openmed_hub.totp import (
    consume_recovery,
    hash_recovery,
    new_recovery_codes,
    new_secret,
    otpauth_uri,
    qr_svg,
    verify_totp,
)
from openmed_hub.trustplane import TrustPlane, verifier_from_public_hex

GATE_ACTOR = "hub:automated-gate"
#: Model names and versions end up in file names, URLs and citations.
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

# Per-IP login attempts within a sliding ten-minute window (process-local).
_LOGIN_WINDOW_SECONDS = 600
_login_attempts: "Dict[str, deque]" = defaultdict(deque)
_login_lock = threading.Lock()
HUB_ACTOR = "hub"
WEIGHT_FORMATS = ("npz", "npy", "pt", "pth", "onnx", "pkl", "bin", "safetensors", "other")


class HubError(Exception):
    """An error with an HTTP status, a machine-readable code and a detail."""

    def __init__(self, status_code: int, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.code = code
        self.detail = detail


def _slugify(value: str) -> str:
    out = []
    for ch in value.strip().lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in " -_./":
            out.append("-")
    slug = "".join(out).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug[:64]


def _parse_json(value: Any, field: str) -> Any:
    if value is None or value == "":
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError) as exc:
        raise HubError(422, "invalid_json", f"{field} is not valid JSON: {exc}")


def load_weights_array(blob: bytes, fmt: str) -> Optional[np.ndarray]:
    """Return a flat parameter vector for numpy formats, else ``None``."""
    try:
        if fmt == "npy":
            return np.asarray(np.load(io.BytesIO(blob), allow_pickle=False), dtype=float).ravel()
        if fmt == "npz":
            with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
                parts = [np.asarray(archive[k], dtype=float).ravel() for k in sorted(archive.files)]
            return np.concatenate(parts) if parts else None
    except Exception:
        return None
    return None


class HubServices:
    """All hub operations, bound to one DB session and the shared trust plane."""

    def __init__(self, trustplane: TrustPlane, settings: HubSettings, db: Session, *, client_ip: str = "") -> None:
        self.tp = trustplane
        self.settings = settings
        self.db = db
        self.client_ip = client_ip

    # ================================================================== audit

    def audit(self, action: str, *, actor: Optional[User] = None, subject: str = "", detail: str = "") -> None:
        """Record a security-relevant event; never raises."""
        self.db.add(
            AuditEvent(
                action=action,
                actor_email=actor.email if actor else "",
                actor_id=actor.id if actor else None,
                subject=subject[:200],
                ip=self.client_ip[:64],
                detail=detail[:2000],
            )
        )
        self.db.commit()

    def audit_log(self, *, limit: int = 200) -> List[AuditEvent]:
        return list(self.db.scalars(select(AuditEvent).order_by(AuditEvent.created_at.desc(), AuditEvent.id.desc()).limit(limit)))

    # ========================================================= policy checks

    def require_member(self, user: User) -> None:
        """Only approved members of an institution act on its behalf."""
        if not user.membership_active:
            raise HubError(
                403,
                "membership_pending",
                "your membership of this institution has not been approved yet; ask an institution "
                "admin to approve it (or join with the institution's invite code)",
            )

    def require_second_factor(self, user: User) -> None:
        """Privileged roles must have two-factor authentication enabled."""
        needed = [r for r in user.role_list if r in self.settings.require_2fa_roles]
        if needed and not user.totp_enabled:
            raise HubError(
                403,
                "totp_required",
                f"the {', '.join(needed)} role(s) require two-factor authentication; enable it under /account first",
            )

    # ================================================================ accounts

    def register_institution(
        self,
        *,
        name: str,
        slug: Optional[str],
        kind: str,
        country: str,
        admin_email: str,
        admin_name: str,
        admin_password: str,
        is_founding: bool = False,
    ) -> "tuple[Institution, User]":
        """Create an institution and its first user (its local admin)."""
        slug = _slugify(slug or name)
        if not slug:
            raise HubError(422, "invalid_slug", "institution needs a name")
        if kind not in INSTITUTION_KINDS:
            raise HubError(422, "invalid_kind", f"kind must be one of {INSTITUTION_KINDS}")
        if self.db.scalar(select(Institution).where(Institution.slug == slug)):
            raise HubError(409, "institution_exists", f"institution {slug!r} already registered")
        institution = Institution(
            slug=slug,
            name=name.strip(),
            kind=kind,
            country=country.strip() or "US",
            is_founding=is_founding,
            service_key_seed=secrets.token_hex(32),
            invite_code=_new_invite_code(),
        )
        self.db.add(institution)
        self.db.flush()
        self.tp.register_institution(institution)
        # Bootstrap: the very first account on a fresh hub is its admin and
        # maintainer, so a hub can be set up without editing the database.
        first_account = self.db.scalar(select(func.count(User.id))) == 0
        user = self.register_user(
            email=admin_email,
            name=admin_name,
            password=admin_password,
            institution=institution,
            roles=(ROLE_CONTRIBUTOR, ROLE_ADMIN, ROLE_MAINTAINER) if first_account else (ROLE_CONTRIBUTOR,),
        )
        user.is_institution_admin = True
        self.db.commit()
        self.audit("institution_registered", actor=user, subject=institution.slug)
        return institution, user

    def register_user(
        self,
        *,
        email: str,
        name: str,
        password: str,
        institution: Institution,
        roles: Sequence[str] = (ROLE_CONTRIBUTOR,),
        invite_code: Optional[str] = None,
    ) -> User:
        """Create an account. Membership is active only when the joiner proves
        a link to the institution (invite code or allowed email domain) or is
        its first member; otherwise it waits for an institution admin."""
        email = email.strip().lower()
        if "@" not in email or email.count("@") != 1:
            raise HubError(422, "invalid_email", "a valid email is required")
        problem = password_problem(password, email=email, name=name)
        if problem:
            raise HubError(422, "weak_password", problem)
        if self.db.scalar(select(User).where(User.email == email)):
            raise HubError(409, "user_exists", f"{email} is already registered")
        first_member = not institution.users
        domain = email.rsplit("@", 1)[1]
        code_ok = bool(invite_code) and bool(institution.invite_code) and secrets.compare_digest(
            invite_code.strip().upper(), institution.invite_code.upper()
        )
        if invite_code and not code_ok:
            raise HubError(403, "bad_invite_code", "that invite code does not match this institution")
        status = MEMBERSHIP_ACTIVE if (first_member or code_ok or domain in institution.domain_list) else MEMBERSHIP_PENDING
        user = User(
            email=email,
            name=name.strip() or email,
            password_hash=hash_password(password),
            institution_id=institution.id,
            roles=",".join(dict.fromkeys(roles)),
            security_stamp=new_stamp(),
            membership_status=status,
            is_institution_admin=first_member,
            password_changed_at=utcnow(),
        )
        self.db.add(user)
        self.db.flush()
        user.institution = institution
        self.db.commit()
        self.audit("user_registered", actor=user, subject=institution.slug, detail=f"membership={status}")
        return user

    # ------------------------------------------------------------ membership

    def approve_member(self, *, actor: User, member: User, approve: bool) -> User:
        """An institution admin (or hub admin) decides a pending membership."""
        self._require_institution_admin(actor, member.institution)
        if member.id == actor.id:
            raise HubError(400, "self_approval", "you cannot decide your own membership")
        member.membership_status = MEMBERSHIP_ACTIVE if approve else MEMBERSHIP_REJECTED
        member.is_active = bool(approve)
        self.db.commit()
        self.audit("member_approved" if approve else "member_rejected", actor=actor, subject=member.email)
        return member

    def set_institution_admin(self, *, actor: User, member: User, flag: bool) -> User:
        self._require_institution_admin(actor, member.institution)
        member.is_institution_admin = bool(flag)
        self.db.commit()
        self.audit("institution_admin_changed", actor=actor, subject=member.email, detail=str(flag))
        return member

    def update_institution_settings(
        self, *, actor: User, institution: Institution, allowed_email_domains: Optional[str] = None,
        rotate_invite_code: bool = False,
    ) -> Institution:
        self._require_institution_admin(actor, institution)
        if allowed_email_domains is not None:
            domains = [d.strip().lower().lstrip("@") for d in allowed_email_domains.split(",") if d.strip()]
            for d in domains:
                if "." not in d or "/" in d or " " in d:
                    raise HubError(422, "invalid_domain", f"{d!r} is not a domain name")
            institution.allowed_email_domains = ",".join(domains)
        if rotate_invite_code:
            institution.invite_code = _new_invite_code()
        self.db.commit()
        self.audit("institution_settings_changed", actor=actor, subject=institution.slug)
        return institution

    def _require_institution_admin(self, actor: User, institution: Institution) -> None:
        if actor.is_admin:
            return
        if actor.institution_id != institution.id or not actor.institution_admin or not actor.membership_active:
            raise HubError(403, "institution_admin_only", "only an admin of that institution may do this")

    def members(self, institution: Institution) -> List[User]:
        return sorted(institution.users, key=lambda u: (u.membership_status or MEMBERSHIP_ACTIVE, u.id))

    def _rate_limit_login(self) -> None:
        if not self.client_ip:
            return
        now = datetime.now().timestamp()
        with _login_lock:
            attempts = _login_attempts[self.client_ip]
            while attempts and attempts[0] < now - _LOGIN_WINDOW_SECONDS:
                attempts.popleft()
            if len(attempts) >= self.settings.login_rate_limit:
                raise HubError(429, "too_many_attempts", "too many sign-in attempts from this address; try again later")
            attempts.append(now)

    def authenticate(self, email: str, password: str) -> User:
        """First factor. Locks the account for a growing delay after repeated failures."""
        self._rate_limit_login()
        email = email.strip().lower()
        user = self.db.scalar(select(User).where(User.email == email))
        if user is not None and user.locked_until and user.locked_until > utcnow():
            remaining = int((user.locked_until - utcnow()).total_seconds()) + 1
            self.audit("login_locked", subject=email)
            raise HubError(423, "account_locked", f"account temporarily locked after repeated failures; try again in {remaining}s")
        if user is None or not user.is_active or not verify_password(password, user.password_hash):
            if user is not None:
                self._register_failure(user)
            self.audit("login_failed", subject=email)
            raise HubError(401, "bad_credentials", "email or password is incorrect")
        return user

    def _register_failure(self, user: User) -> None:
        user.failed_logins = (user.failed_logins or 0) + 1
        over = user.failed_logins - self.settings.lockout_threshold
        if over >= 0:
            delay = min(30 * (2 ** over), self.settings.lockout_max_seconds)
            user.locked_until = utcnow() + timedelta(seconds=delay)
        self.db.commit()

    def _clear_failures(self, user: User) -> None:
        if user.failed_logins or user.locked_until:
            user.failed_logins = 0
            user.locked_until = None
            self.db.commit()

    def second_factor_ok(self, user: User, code: str) -> bool:
        """Verify a TOTP code or consume a recovery code."""
        if not user.totp_enabled:
            return True
        code = (code or "").strip()
        if verify_totp(user.totp_secret or "", code):
            return True
        remaining = consume_recovery(code, json.loads(user.recovery_codes or "[]"))
        if remaining is not None:
            user.recovery_codes = json.dumps(remaining)
            self.db.commit()
            self.audit("recovery_code_used", actor=user, detail=f"{len(remaining)} left")
            return True
        return False

    def login(self, email: str, password: str, totp: Optional[str] = None) -> User:
        """Both factors in one call (the API path)."""
        user = self.authenticate(email, password)
        if user.totp_enabled:
            if not totp:
                raise HubError(401, "totp_required", "two-factor authentication is enabled; supply the current code as `totp`")
            if not self.second_factor_ok(user, totp):
                self._register_failure(user)
                self.audit("login_failed", actor=user, detail="bad second factor")
                raise HubError(401, "bad_totp", "the two-factor code is incorrect")
        return self.complete_login(user)

    def complete_login(self, user: User) -> User:
        self._clear_failures(user)
        self.audit("login_ok", actor=user)
        return user

    # ------------------------------------------------------------- 2FA

    def start_totp_enrollment(self, user: User) -> Dict[str, Any]:
        """Issue a secret; nothing is enforced until :meth:`confirm_totp`."""
        if user.totp_enabled:
            raise HubError(409, "totp_already_enabled", "two-factor authentication is already enabled; disable it first to re-enrol")
        user.totp_secret = new_secret()
        self.db.commit()
        uri = otpauth_uri(user.totp_secret, user.email, self.settings.hub_name)
        return {"secret": user.totp_secret, "otpauth_uri": uri, "qr_svg": qr_svg(uri)}

    def confirm_totp(self, user: User, code: str) -> List[str]:
        """Prove the authenticator works; returns the recovery codes, once."""
        if user.totp_enabled:
            raise HubError(409, "totp_already_enabled", "already enabled")
        if not user.totp_secret or not verify_totp(user.totp_secret, code):
            raise HubError(400, "bad_totp", "that code did not match; check the authenticator's clock and try the next code")
        codes = new_recovery_codes()
        user.recovery_codes = json.dumps([hash_recovery(c) for c in codes])
        user.totp_enabled_at = utcnow()
        user.security_stamp = new_stamp()  # other sessions must re-authenticate
        self.db.commit()
        self.audit("totp_enabled", actor=user)
        return codes

    def disable_totp(self, user: User, *, password: str, code: str) -> User:
        if not verify_password(password, user.password_hash):
            raise HubError(401, "bad_credentials", "password is incorrect")
        if not self.second_factor_ok(user, code):
            raise HubError(400, "bad_totp", "the two-factor or recovery code is incorrect")
        user.totp_secret = None
        user.totp_enabled_at = None
        user.recovery_codes = None
        user.security_stamp = new_stamp()
        self.db.commit()
        self.audit("totp_disabled", actor=user)
        return user

    def regenerate_recovery_codes(self, user: User, *, code: str) -> List[str]:
        if not user.totp_enabled:
            raise HubError(400, "totp_not_enabled", "enable two-factor authentication first")
        if not verify_totp(user.totp_secret or "", code):
            raise HubError(400, "bad_totp", "the two-factor code is incorrect")
        codes = new_recovery_codes()
        user.recovery_codes = json.dumps([hash_recovery(c) for c in codes])
        self.db.commit()
        self.audit("recovery_codes_regenerated", actor=user)
        return codes

    def change_password(self, user: User, *, current_password: str, new_password: str) -> User:
        if not verify_password(current_password, user.password_hash):
            raise HubError(401, "bad_credentials", "current password is incorrect")
        problem = password_problem(new_password, email=user.email, name=user.name)
        if problem:
            raise HubError(422, "weak_password", problem)
        user.password_hash = hash_password(new_password)
        user.password_changed_at = utcnow()
        user.security_stamp = new_stamp()  # every other session is signed out
        self.db.commit()
        self.audit("password_changed", actor=user)
        return user

    # ----------------------------------------------------------- tokens

    def create_token(self, user: User, label: str = "") -> str:
        # Pending members may hold a token (to check their status); every
        # action they attempt is refused by require_member().
        token = new_token()
        self.db.add(ApiToken(user_id=user.id, token_hash=token_hash(token), label=label[:100]))
        self.db.commit()
        self.audit("token_created", actor=user, detail=label[:100])
        return token

    def revoke_token(self, user: User, token_id: int) -> None:
        row = self.db.get(ApiToken, token_id)
        if row is None or row.user_id != user.id:
            raise HubError(404, "token_not_found", "no such token")
        self.db.delete(row)
        self.db.commit()
        self.audit("token_revoked", actor=user, detail=str(token_id))

    def institution_by_slug(self, slug: str) -> Institution:
        institution = self.db.scalar(select(Institution).where(Institution.slug == slug))
        if institution is None:
            raise HubError(404, "institution_not_found", f"no institution {slug!r}")
        return institution

    def list_institutions(self) -> List[Institution]:
        return list(self.db.scalars(select(Institution).order_by(Institution.created_at)))

    # -------------------------------------------------------------- roles/keys

    def grant_roles(self, user: User, roles: Iterable[str], *, actor: Optional[User] = None) -> User:
        """Set a user's roles; registers a reviewer identity when needed."""
        if actor is not None:
            self.require_second_factor(actor)
        roles = list(dict.fromkeys(r for r in roles if r))
        unknown = [r for r in roles if r not in ALL_ROLES]
        if unknown:
            raise HubError(422, "unknown_role", f"unknown roles: {unknown}")
        if ROLE_CONTRIBUTOR not in roles:
            roles.insert(0, ROLE_CONTRIBUTOR)
        user.roles = ",".join(roles)
        if any(r in roles for r in REVIEWER_ROLES):
            if not user.reviewer_id:
                user.reviewer_id = f"rev-{user.institution.slug}-{user.id}"
            if not user.reviewer_public_key and not user.reviewer_key_seed:
                user.reviewer_key_seed = secrets.token_hex(32)
            self.tp.register_reviewer(user)
        elif user.reviewer_id:
            self.tp.revoke_reviewer(user.reviewer_id)
        self.db.commit()
        self.audit("roles_changed", actor=actor, subject=user.email, detail=",".join(roles))
        return user

    def set_reviewer_public_key(self, user: User, public_key_hex: str) -> User:
        """Switch a reviewer to a client-held key (detached signatures only)."""
        if not user.is_reviewer:
            raise HubError(403, "not_a_reviewer", "only reviewers hold a review key")
        public_key_hex = public_key_hex.strip().lower()
        try:
            verifier_from_public_hex(public_key_hex)
        except Exception as exc:
            raise HubError(422, "invalid_public_key", f"not a 32-byte Ed25519 public key: {exc}")
        user.reviewer_public_key = public_key_hex
        user.reviewer_key_seed = None
        self.tp.register_reviewer(user)
        self.db.commit()
        self.audit("reviewer_key_changed", actor=user, detail="client-held")
        return user

    def set_founding(self, institution: Institution, is_founding: bool, *, actor: Optional[User] = None) -> Institution:
        if actor is not None:
            self.require_second_factor(actor)
        institution.is_founding = bool(is_founding)
        self.db.commit()
        self.audit("founding_changed", actor=actor, subject=institution.slug, detail=str(bool(is_founding)))
        return institution

    # ------------------------------------------------------------- node keys

    def begin_node_registration(self, institution: Institution, public_key_hex: str, *, actor: Optional[User] = None) -> str:
        """Store the node's public key and issue a challenge nonce."""
        if actor is not None:
            self.require_member(actor)
        public_key_hex = public_key_hex.strip().lower()
        try:
            verifier_from_public_hex(public_key_hex)
        except Exception as exc:
            raise HubError(422, "invalid_public_key", f"not a 32-byte Ed25519 public key: {exc}")
        institution.node_public_key = public_key_hex
        institution.node_verified_at = None
        self.tp.register_node_key(institution)  # drops the old key until verified
        nonce = secrets.token_hex(32)
        self.db.add(NodeChallenge(institution_id=institution.id, nonce=nonce))
        self.db.commit()
        return nonce

    def complete_node_registration(
        self, institution: Institution, nonce: str, signature_hex: str
    ) -> Institution:
        """Verify the node's signature over the nonce; mark the node verified."""
        challenge = self.db.scalar(
            select(NodeChallenge).where(
                NodeChallenge.institution_id == institution.id,
                NodeChallenge.nonce == nonce,
                NodeChallenge.used.is_(False),
            )
        )
        if challenge is None:
            raise HubError(400, "unknown_challenge", "no outstanding challenge with that nonce")
        if challenge.created_at < utcnow() - timedelta(minutes=15):
            raise HubError(400, "challenge_expired", "challenge expired; request a new one")
        if not institution.node_public_key:
            raise HubError(400, "no_node_key", "register the node public key first")
        verifier = verifier_from_public_hex(institution.node_public_key)
        if not verifier.verify(nonce.encode("utf-8"), signature_hex.strip()):
            raise HubError(400, "bad_signature", "signature over the nonce did not verify")
        challenge.used = True
        institution.node_verified_at = utcnow()
        self.db.commit()
        self.tp.register_node_key(institution)
        self.audit("node_verified", subject=institution.slug, detail=institution.node_public_key or "")
        return institution

    # ============================================================ submissions

    def submit(
        self,
        *,
        user: User,
        name: str,
        version: str,
        weights: bytes,
        weights_format: str,
        model_card: Any,
        evaluation: Any,
        manual: Any = None,
        parent_bundle_id: Optional[str] = None,
        code_identity: str = "",
        config: str = "",
        quote: Any = None,
        tags: Sequence[str] = (),
        evidence: Any = None,
    ) -> Submission:
        """Publish a bundle, run the automated gate, and open its case.

        In ``nodekey`` attestation mode the submission must carry ``quote``: a
        :class:`~trustfed.attestation.Quote` signed at the site with its
        verified node key over a software measurement of the code that ran
        (see ``openmed submit``). In ``mock`` mode the hub signs the declared
        ``code_identity`` itself, which is a demo convenience only.
        """
        self.require_member(user)
        institution = user.institution
        name = name.strip()
        version = version.strip()
        if not NAME_RE.match(name) or not NAME_RE.match(version):
            raise HubError(422, "invalid_name", "name and version may only contain letters, digits, '.', '_' and '-' (max 64)")
        if not weights:
            raise HubError(422, "missing_weights", "a weights file is required")
        if len(weights) > self.settings.max_upload_bytes:
            raise HubError(413, "too_large", "weights exceed the upload limit")
        weights_format = (weights_format or "other").lower().lstrip(".")
        if weights_format not in WEIGHT_FORMATS:
            weights_format = "other"

        card_raw = _parse_json(model_card, "model_card") or {}
        try:
            card = ModelCard.from_dict(card_raw)
        except ValidationError as exc:
            raise HubError(422, "invalid_model_card", str(exc))
        card_check = card.validate()
        if not card_check.ok:
            raise HubError(
                422,
                "incomplete_model_card",
                "model card incomplete: missing sections "
                f"{list(card_check.missing_sections)}, missing model_details "
                f"{list(card_check.missing_details)}",
            )

        eval_raw = _parse_json(evaluation, "evaluation") or {}
        try:
            report = EvaluationReport(
                dataset_id=str(eval_raw.get("dataset_id", "")),
                n_samples=int(eval_raw.get("n_samples", 0) or 0),
                metrics={k: float(v) for k, v in dict(eval_raw.get("metrics", {})).items()},
                subgroup_metrics={
                    g: {k: float(v) for k, v in dict(m).items()}
                    for g, m in dict(eval_raw.get("subgroup_metrics", {})).items()
                },
                protocol=str(eval_raw.get("protocol", "")),
                evaluated_by=str(eval_raw.get("evaluated_by", "")),
                seed=eval_raw.get("seed"),
            )
        except (ValidationError, TypeError, ValueError) as exc:
            raise HubError(422, "invalid_evaluation", f"evaluation report: {exc}")

        manual_raw = _parse_json(manual, "fine_tuning_manual")
        manual_ok: Optional[bool] = None
        if manual_raw:
            try:
                manual_ok = FineTuningManual.from_dict(manual_raw).validate().ok
            except Exception as exc:
                raise HubError(422, "invalid_manual", f"fine-tuning manual: {exc}")

        parent = (parent_bundle_id or "").strip() or None
        # A version is a release: one per name and institution. (Bundle ids
        # also cover the fresh attestation nonce, so the registry alone would
        # not catch a re-upload of the same release.)
        existing = self.db.scalar(
            select(Submission).where(
                Submission.institution_id == institution.id,
                Submission.name == name,
                Submission.version == version,
            )
        )
        if existing is not None:
            raise HubError(409, "version_exists", f"{name} {version} was already submitted by {institution.slug}")
        with self.tp.lock:
            if parent and parent not in self.tp.registry:
                raise HubError(404, "parent_not_found", f"parent bundle not registered: {parent}")
            attestation = self._attest_submission(institution, code_identity, config, quote)
            sha = self.tp.blobs.put(weights)
            ref = WeightsRef(
                uri=self.tp.blobs.uri(sha), sha256=sha, size_bytes=len(weights), format=weights_format
            )
            try:
                bundle = self.tp.registry.publish(
                    name=name,
                    version=version,
                    weights=ref,
                    model_card=card,
                    evaluation=report,
                    attestation=attestation,
                    published_by=institution.slug,
                    parents=(parent,) if parent else (),
                    fine_tuning_manual=manual_raw,
                    tags=tuple(t.strip() for t in tags if t.strip()),
                )
            except DuplicateBundleError as exc:
                raise HubError(409, "duplicate_bundle", str(exc))
            except BundleNotFoundError as exc:
                raise HubError(404, "parent_not_found", str(exc))
            except RegistryError as exc:
                raise HubError(422, "publish_refused", str(exc))

            # Automated gate.
            evidence_raw = _parse_json(evidence, "evidence") or {}
            quality = QualityEvidence(
                member_scores=_optional_array(evidence_raw.get("member_scores")),
                nonmember_scores=_optional_array(evidence_raw.get("nonmember_scores")),
                subgroup_metrics=dict(report.subgroup_metrics) or None,
                weights=load_weights_array(weights, weights_format),
                notes={"submitted_by": user.email, "institution": institution.slug},
            )
            gate = QualityAnalyzer().analyze(bundle, quality)

            # Certification case, with the owner derived from the bundle.
            case = self.tp.authority.submit(
                bundle,
                submitted_by=user.email,
                manual=manual_raw,
                registry=self.tp.registry,
            )
            if gate.blocking:
                reasons = "; ".join(f"{r.check_id}: {r.summary}" for r in gate.blocking)
                self.tp.authority.dispute(
                    bundle.bundle_id,
                    raised_by=GATE_ACTOR,
                    reason=f"automated gate: {reasons}"[:2000],
                )

        submission = Submission(
            bundle_id=bundle.bundle_id,
            name=name,
            version=version,
            submitted_by_id=user.id,
            institution_id=institution.id,
            parent_bundle_id=parent,
            weights_sha256=sha,
            weights_size=len(weights),
            weights_format=weights_format,
            gate_status=gate_verdict(gate),
            gate_report=json.dumps(gate.to_dict(), sort_keys=True),
            state=case.state.value,
        )
        self.db.add(submission)
        self.db.commit()
        return submission

    def _attest_submission(self, institution: Institution, code_identity: str, config: str, quote: Any):
        """Run the attestation gate for one submission, per the configured mode."""
        if self.settings.attestation_mode == "nodekey":
            raw = _parse_json(quote, "quote")
            if not raw:
                raise HubError(
                    422,
                    "quote_required",
                    "this hub requires a quote signed with your institution's node key over a "
                    "software measurement of the code that ran; submit with `openmed submit` "
                    "(it measures, requests a challenge and signs) instead of the web form",
                )
            try:
                parsed = Quote(
                    client_id=str(raw["client_id"]),
                    measurement=str(raw["measurement"]),
                    config_hash=str(raw.get("config_hash", "")),
                    signature=str(raw.get("signature", "")),
                    nonce=str(raw.get("nonce", "")),
                    issued_at=float(raw.get("issued_at", 0.0) or 0.0),
                    attestor_id=str(raw.get("attestor_id", "")),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise HubError(422, "invalid_quote", f"malformed quote: {exc}")
            if parsed.client_id != institution.slug:
                raise HubError(403, "quote_client_mismatch", "the quote names a different institution")
            if not institution.node_verified:
                raise HubError(403, "node_not_verified", "complete the node verification handshake first (`openmed node register`)")
            attestation = self.tp.verify_node_quote(parsed)
            if not attestation.verified:
                raise HubError(
                    422,
                    "attestation_rejected",
                    f"quote did not verify ({attestation.verifier_reason}); approved measurements: "
                    f"{len(self.tp.approved_measurements())} on the allow-list (see /api/v1/attestation/policy)",
                )
            return attestation
        if not (code_identity or "").strip():
            raise HubError(422, "missing_code_identity", "code_identity is required in mock attestation mode")
        attestation = self.tp.attest(institution.slug, code_identity.strip(), config or "")
        if not attestation.verified:
            raise HubError(
                422,
                "attestation_rejected",
                f"pipeline attestation did not verify ({attestation.verifier_reason}); "
                f"approved code identities: {list(self.tp.approved_code_identities())}",
            )
        return attestation

    # ---------------------------------------------------- approved measurements

    def attestation_policy(self) -> Dict[str, Any]:
        mode = self.settings.attestation_mode
        return {
            "mode": mode,
            "attestor": (
                "node-key software measurement: the site measures its code and signs the quote with its verified node key"
                if mode == "nodekey"
                else "software mock: the hub signs the declared code identity (demo only)"
            ),
            "approved_measurements": [
                {"measurement": m.measurement, "label": m.label, "added_at": m.added_at.isoformat()}
                for m in self.list_measurements()
            ] + [
                {"measurement": m, "label": "configured at startup", "added_at": None}
                for m in self.settings.approved_measurements
            ],
            "approved_code_identities": list(self.tp.approved_code_identities()) if mode == "mock" else [],
            "require_nonce": True,
            "max_age_seconds": self.settings.attestation_max_age_seconds,
        }

    def list_measurements(self, *, include_revoked: bool = False) -> List[ApprovedMeasurement]:
        query = select(ApprovedMeasurement).order_by(ApprovedMeasurement.added_at.desc())
        if not include_revoked:
            query = query.where(ApprovedMeasurement.revoked_at.is_(None))
        return list(self.db.scalars(query))

    def approve_measurement(self, *, user: User, measurement: str, label: str = "", manifest: Any = None) -> ApprovedMeasurement:
        """Maintainers publish the digest of an approved pipeline release."""
        if not user.is_maintainer:
            raise HubError(403, "maintainer_only", "only maintainers approve pipeline measurements")
        self.require_second_factor(user)
        measurement = measurement.strip().lower()
        if len(measurement) != 64 or any(c not in "0123456789abcdef" for c in measurement):
            raise HubError(422, "invalid_measurement", "a measurement is a 64-hex-character SHA-256 digest")
        row = self.db.scalar(select(ApprovedMeasurement).where(ApprovedMeasurement.measurement == measurement))
        if row is None:
            row = ApprovedMeasurement(measurement=measurement, added_by_id=user.id)
            self.db.add(row)
        row.label = label.strip()[:200]
        row.manifest = json.dumps(_parse_json(manifest, "manifest") or {}, sort_keys=True)
        row.revoked_at = None
        row.added_at = utcnow()
        self.db.commit()
        self.tp.approve_measurement(measurement)
        self.audit("measurement_approved", actor=user, subject=measurement, detail=label[:200])
        return row

    def revoke_measurement(self, *, user: User, measurement: str) -> ApprovedMeasurement:
        if not user.is_maintainer:
            raise HubError(403, "maintainer_only", "only maintainers revoke pipeline measurements")
        self.require_second_factor(user)
        row = self.db.scalar(select(ApprovedMeasurement).where(ApprovedMeasurement.measurement == measurement.strip().lower()))
        if row is None:
            raise HubError(404, "measurement_not_found", "no such approved measurement")
        row.revoked_at = utcnow()
        self.db.commit()
        self.tp.revoke_measurement(row.measurement)
        self.audit("measurement_revoked", actor=user, subject=row.measurement)
        return row

    def issue_challenge(self, user: User) -> Dict[str, Any]:
        """A nonce the site's node must sign into its next quote."""
        self.require_member(user)
        if not user.institution.node_verified:
            raise HubError(403, "node_not_verified", "complete the node verification handshake first")
        with self.tp.lock:
            nonce = self.tp.issue_challenge(user.institution.slug)
        return {
            "client_id": user.institution.slug,
            "nonce": nonce,
            "expires_in_seconds": self.settings.attestation_max_age_seconds,
        }

    # ---------------------------------------------------------------- lookup

    def get_submission(self, bundle_id: str) -> Submission:
        submission = self.db.scalar(select(Submission).where(Submission.bundle_id == bundle_id))
        if submission is None:
            # Allow a short id prefix as a convenience.
            candidates = list(
                self.db.scalars(select(Submission).where(Submission.bundle_id.like(f"%{bundle_id}%")))
            )
            if len(candidates) == 1:
                submission = candidates[0]
        if submission is None:
            raise HubError(404, "submission_not_found", f"no submission for {bundle_id}")
        return submission

    def list_submissions(self, *, state: Optional[str] = None, institution_id: Optional[int] = None) -> List[Submission]:
        query = select(Submission).order_by(Submission.created_at.desc())
        if state:
            query = query.where(Submission.state == state)
        if institution_id is not None:
            query = query.where(Submission.institution_id == institution_id)
        return list(self.db.scalars(query))

    def review_queue(self, user: User) -> List[Submission]:
        """Submissions awaiting review from institutions other than the user's."""
        query = (
            select(Submission)
            .where(Submission.state.in_(("submitted", "under_review", "remediated")))
            .where(Submission.institution_id != user.institution_id)
            .order_by(Submission.created_at)
        )
        return list(self.db.scalars(query))

    def submission_detail(self, submission: Submission) -> Dict[str, Any]:
        """Everything the model page shows, as one JSON-able dict."""
        with self.tp.lock:
            bundle = self.tp.registry.get(submission.bundle_id)
            try:
                case = self.tp.authority.case(submission.bundle_id).to_dict()
                outcome = self.tp.authority.evaluate(submission.bundle_id).to_dict()
            except UnknownCaseError:
                case, outcome = None, None
            verdict = self.tp.registry.verify_lineage(submission.bundle_id)
            graph = self.tp.registry.lineage_graph(submission.bundle_id)
            children = list(self.tp.registry.children(submission.bundle_id))
            log = [b.payload for b in self.tp.authority.decision_log(submission.bundle_id)]
        return {
            "submission": self.submission_summary(submission),
            "bundle": bundle.to_dict(),
            "model_card_markdown": bundle.model_card.to_markdown(),
            "case": case,
            "threshold": outcome,
            "board": self.board_composition(submission),
            "gate": json.loads(submission.gate_report or "{}"),
            "lineage": verdict.to_dict(),
            "lineage_summary": verdict.summary(),
            "lineage_mermaid": graph.to_mermaid(),
            "children": children,
            "decision_log": log,
            "reviews": [self.review_summary(r) for r in submission.reviews],
            "downloads": self.db.scalar(
                select(func.count(Download.id)).where(Download.bundle_id == submission.bundle_id)
            ),
        }

    def submission_summary(self, submission: Submission) -> Dict[str, Any]:
        return {
            "bundle_id": submission.bundle_id,
            "short_id": submission.bundle_id[-12:],
            "name": submission.name,
            "version": submission.version,
            "institution": submission.institution.slug,
            "submitted_by": submission.submitted_by.email,
            "parent_bundle_id": submission.parent_bundle_id,
            "state": submission.state,
            "gate_status": submission.gate_status,
            "weights": {
                "sha256": submission.weights_sha256,
                "size_bytes": submission.weights_size,
                "format": submission.weights_format,
            },
            "identifier": submission.identifier,
            "citation": submission.citation,
            "created_at": submission.created_at.isoformat(),
            "certified_at": submission.certified_at.isoformat() if submission.certified_at else None,
        }

    @staticmethod
    def review_summary(review: Review) -> Dict[str, Any]:
        return {
            "id": review.id,
            "reviewer_id": review.reviewer_id,
            "institution": review.institution,
            "decision": review.decision,
            "statement": review.statement,
            "signed_at": review.signed_at,
            "key_id": review.key_id,
            "hub_signed": review.hub_signed,
            "accepted": review.accepted,
            "refusal_code": review.refusal_code,
            "refusal_detail": review.refusal_detail,
            "created_at": review.created_at.isoformat(),
        }

    # ---------------------------------------------------------------- reviews

    def review(self, *, submission: Submission, user: User, decision: str, statement: str) -> Review:
        """Sign a review on the web path (hub-held key) and record it."""
        if not user.is_reviewer or not user.reviewer_id:
            raise HubError(403, "not_a_reviewer", "only registered reviewers may sign")
        self.require_member(user)
        self.require_second_factor(user)
        if user.reviewer_public_key:
            raise HubError(
                400,
                "client_held_key",
                "your reviewer key is client-held; sign with `openmed review` and submit the signature",
            )
        if decision not in ("approve", "reject", "block"):
            raise HubError(422, "invalid_decision", "decision must be approve, reject or block")
        with self.tp.lock:
            try:
                signature = self.tp.authority.review(
                    submission.bundle_id, user.reviewer_id, decision, statement=statement.strip()
                )
            except ConflictOfInterestError as exc:
                row = self._record_review(submission, user, None, decision, statement, True, exc)
                self._sync_state(submission)
                return row
            except (InvalidTransitionError, UnknownReviewerError, SignatureRejectedError, CertificationError) as exc:
                raise HubError(409, "review_rejected", str(exc))
            row = self._record_review(submission, user, signature, decision, statement, True, None)
            self._after_review(submission, user, signature)
        return row

    def submit_signature(self, *, submission: Submission, signature_data: Dict[str, Any]) -> Review:
        """Accept a detached signature made with the reviewer's own key."""
        try:
            signature = ReviewSignature.from_dict(signature_data)
        except (KeyError, TypeError, ValueError) as exc:
            raise HubError(422, "invalid_signature", f"malformed signature: {exc}")
        if signature.bundle_id != submission.bundle_id:
            raise HubError(422, "bundle_mismatch", "signature covers a different bundle")
        user = self.db.scalar(select(User).where(User.reviewer_id == signature.reviewer_id))
        if user is None:
            raise HubError(404, "unknown_reviewer", f"no reviewer {signature.reviewer_id!r}")
        self.require_member(user)
        with self.tp.lock:
            try:
                accepted = self.tp.authority.submit_signature(signature)
            except ConflictOfInterestError as exc:
                row = self._record_review(submission, user, signature, signature.decision, signature.statement, False, exc)
                self._sync_state(submission)
                return row
            except SignatureRejectedError as exc:
                raise HubError(400, "signature_rejected", str(exc))
            except (InvalidTransitionError, UnknownReviewerError, CertificationError) as exc:
                raise HubError(409, "review_rejected", str(exc))
            row = self._record_review(submission, user, accepted, accepted.decision, accepted.statement, False, None)
            self._after_review(submission, user, accepted)
        return row

    def _record_review(
        self,
        submission: Submission,
        user: Optional[User],
        signature: Optional[ReviewSignature],
        decision: str,
        statement: str,
        hub_signed: bool,
        refusal: Optional[ConflictOfInterestError],
    ) -> Review:
        row = Review(
            submission_id=submission.id,
            user_id=user.id if user else None,
            reviewer_id=signature.reviewer_id if signature else (user.reviewer_id if user else ""),
            institution=signature.institution if signature else (user.institution.slug if user else ""),
            decision=decision,
            statement=statement,
            signed_at=signature.signed_at if signature else "",
            key_id=signature.key_id if signature else "",
            signature=signature.signature if signature else "",
            hub_signed=hub_signed,
            accepted=refusal is None,
            refusal_code=getattr(refusal, "reason_code", None) if refusal else None,
            refusal_detail=str(refusal) if refusal else None,
        )
        self.db.add(row)
        self.db.flush()
        return row

    def _after_review(self, submission: Submission, user: User, signature: ReviewSignature) -> None:
        self.tp.credit.record(
            user.institution.slug,
            "review_signed",
            institution=user.institution.slug,
            ref=submission.bundle_id,
            detail=f"{signature.reviewer_id}:{signature.decision}",
        )
        self._sync_state(submission)
        self._maybe_certify(submission)

    def board_composition(self, submission: Submission) -> Dict[str, Any]:
        """Count the approving reviewers by role, per the charter's board rule."""
        try:
            case = self.tp.authority.case(submission.bundle_id)
        except UnknownCaseError:
            return {"technical": 0, "clinical": 0, "institutions": [], "satisfied": False}
        latest: Dict[str, ReviewSignature] = {}
        for sig in case.signatures:
            if self.tp.keyring.verify(sig):
                latest[sig.reviewer_id] = sig
        approving = [s for s in latest.values() if s.decision == "approve"]
        technical = clinical = 0
        for sig in approving:
            reviewer = self.db.scalar(select(User).where(User.reviewer_id == sig.reviewer_id))
            if reviewer is None:
                continue
            if reviewer.has_role(ROLE_REVIEWER_TECHNICAL):
                technical += 1
            if reviewer.has_role(ROLE_REVIEWER_CLINICAL):
                clinical += 1
        return {
            "technical": technical,
            "clinical": clinical,
            "technical_required": self.settings.technical_reviewers_min,
            "clinical_required": self.settings.clinical_reviewers_min,
            "institutions": sorted({s.institution for s in approving}),
            "satisfied": (
                technical >= self.settings.technical_reviewers_min
                and clinical >= self.settings.clinical_reviewers_min
            ),
        }

    def _maybe_certify(self, submission: Submission) -> bool:
        """Certify when the threshold and the board composition are both met."""
        state = self.tp.authority.state(submission.bundle_id)
        if state is not CertificationState.UNDER_REVIEW:
            return False
        if not self.board_composition(submission)["satisfied"]:
            return False
        try:
            self.tp.authority.certify(submission.bundle_id, actor=HUB_ACTOR)
        except ThresholdNotMetError:
            return False
        self._on_certified(submission)
        return True

    def _on_certified(self, submission: Submission) -> None:
        bundle = self.tp.registry.get(submission.bundle_id)
        case = self.tp.authority.case(submission.bundle_id)
        contributors = [
            Attribution(bundle.published_by, "training and submission", bundle.published_by, share=0.6)
        ]
        approving = [s for s in case.signatures if s.decision == "approve" and self.tp.keyring.verify(s)]
        seen = set()
        for sig in approving:
            if sig.reviewer_id in seen:
                continue
            seen.add(sig.reviewer_id)
            contributors.append(Attribution(sig.reviewer_id, "certification review", sig.institution, share=0.4 / max(1, len(approving))))
        release = ReleaseAttribution.for_bundle(
            bundle.bundle_id,
            title=f"{bundle.name} (certified base model)",
            year=utcnow().year,
            version=bundle.version,
            contributors=contributors,
        )
        self.tp.credit.record_release(release)
        self.tp.credit.record(
            bundle.published_by,
            "release_published",
            institution=bundle.published_by,
            ref=bundle.bundle_id,
        )
        submission.identifier = release.identifier.uri
        submission.citation = release.to_citation()
        submission.certified_at = utcnow()
        self._sync_state(submission)

    def _sync_state(self, submission: Submission) -> None:
        try:
            submission.state = self.tp.authority.state(submission.bundle_id).value
        except UnknownCaseError:
            pass
        submission.updated_at = utcnow()
        self.db.commit()

    # --------------------------------------------------------------- disputes

    def dispute(self, *, submission: Submission, user: User, reason: str) -> Submission:
        self.require_member(user)
        if not reason.strip():
            raise HubError(422, "missing_reason", "a reason is required")
        with self.tp.lock:
            try:
                self.tp.authority.dispute(submission.bundle_id, raised_by=user.email, reason=reason.strip())
            except (InvalidTransitionError, CertificationError) as exc:
                raise HubError(409, "dispute_rejected", str(exc))
            self._sync_state(submission)
        return submission

    def remediate(self, *, submission: Submission, user: User, note: str) -> Submission:
        self._require_owner(submission, user)
        with self.tp.lock:
            try:
                self.tp.authority.remediate(submission.bundle_id, actor=user.email, note=note.strip())
            except (InvalidTransitionError, CertificationError) as exc:
                raise HubError(409, "remediate_rejected", str(exc))
            self._sync_state(submission)
        return submission

    def appeal(self, *, submission: Submission, user: User, grounds: str) -> Submission:
        self._require_owner(submission, user)
        with self.tp.lock:
            try:
                self.tp.authority.appeal(submission.bundle_id, actor=user.email, grounds=grounds.strip())
            except (InvalidTransitionError, CertificationError) as exc:
                raise HubError(409, "appeal_rejected", str(exc))
            self._sync_state(submission)
        return submission

    def resolve_appeal(self, *, submission: Submission, user: User, upheld: bool, note: str) -> Submission:
        if not user.is_admin:
            raise HubError(403, "board_only", "only the advisory board (admin) resolves appeals")
        self.require_second_factor(user)
        with self.tp.lock:
            try:
                self.tp.authority.resolve_appeal(
                    submission.bundle_id, actor=user.email, upheld=bool(upheld), note=note.strip()
                )
            except (InvalidTransitionError, CertificationError) as exc:
                raise HubError(409, "resolve_rejected", str(exc))
            self._sync_state(submission)
        return submission

    def revoke(self, *, submission: Submission, user: User, reason: str) -> Submission:
        if not user.is_maintainer:
            raise HubError(403, "maintainer_only", "only maintainers revoke a certification")
        self.require_second_factor(user)
        with self.tp.lock:
            try:
                self.tp.authority.revoke(submission.bundle_id, actor=user.email, reason=reason.strip())
            except (InvalidTransitionError, CertificationError) as exc:
                raise HubError(409, "revoke_rejected", str(exc))
            self._sync_state(submission)
        return submission

    def _require_owner(self, submission: Submission, user: User) -> None:
        self.require_member(user)
        if user.institution_id != submission.institution_id and not user.is_admin:
            raise HubError(403, "not_owner", "only the submitting institution may do that")

    # -------------------------------------------------------------- downloads

    def download(self, *, submission: Submission, user: User):
        """Return the weights path after recording the (registered) download."""
        self.require_member(user)
        path = self.tp.blobs.path(submission.weights_sha256)
        if not path.exists():
            raise HubError(410, "weights_missing", "weights blob is missing from the store")
        self.db.add(
            Download(bundle_id=submission.bundle_id, user_id=user.id, institution_id=user.institution_id)
        )
        self.db.commit()
        return path

    # ============================================================ evaluations

    def request_evaluation(
        self, *, user: User, bundle_id: str, evaluator_slug: str, detail: str = ""
    ) -> EvaluationTask:
        """Ask another site to evaluate a bundle, under the reciprocity rule."""
        self.require_member(user)
        submission = self.get_submission(bundle_id)
        requester = user.institution
        if submission.institution_id != requester.id:
            raise HubError(403, "not_owner", "only the owning institution may request evaluations")
        evaluator = self.institution_by_slug(evaluator_slug)
        if evaluator.id == requester.id:
            raise HubError(422, "self_evaluation", "a site cannot evaluate its own model")
        prior = self.db.scalar(
            select(func.count(EvaluationTask.id)).where(
                EvaluationTask.requester_institution_id == requester.id,
                EvaluationTask.state != "refused",
            )
        )
        exempt = {requester.slug} if prior < self.settings.reciprocity_grace_requests else set()
        request = EvaluationRequest(requester.slug, evaluator.slug, submission.bundle_id, detail.strip())
        with self.tp.lock:
            decision = self.tp.reciprocity_policy(exempt).decide_and_record(request, self.tp.credit)
        task = EvaluationTask(
            bundle_id=submission.bundle_id,
            requester_institution_id=requester.id,
            evaluator_institution_id=evaluator.id,
            requested_by_id=user.id,
            state="requested" if decision.allowed else "refused",
            reason_code=decision.reason_code,
            detail=decision.detail if not decision.allowed else detail.strip(),
        )
        self.db.add(task)
        self.db.commit()
        return task

    def serve_evaluation(self, *, task: EvaluationTask, user: User, report: Any) -> EvaluationTask:
        """The evaluator records the metrics it measured on its own cohort."""
        self.require_member(user)
        if user.institution_id != task.evaluator_institution_id:
            raise HubError(403, "not_evaluator", "only the evaluating institution may serve this task")
        if task.state != "requested":
            raise HubError(409, "bad_state", f"task is {task.state}, not requested")
        raw = _parse_json(report, "report") or {}
        try:
            EvaluationReport(
                dataset_id=str(raw.get("dataset_id", "")),
                n_samples=int(raw.get("n_samples", 0) or 0),
                metrics={k: float(v) for k, v in dict(raw.get("metrics", {})).items()},
                subgroup_metrics={
                    g: {k: float(v) for k, v in dict(m).items()}
                    for g, m in dict(raw.get("subgroup_metrics", {})).items()
                },
                protocol=str(raw.get("protocol", "")),
                evaluated_by=user.institution.slug,
            )
        except (ValidationError, TypeError, ValueError) as exc:
            raise HubError(422, "invalid_evaluation", f"evaluation report: {exc}")
        raw["evaluated_by"] = user.institution.slug
        task.report = json.dumps(raw, sort_keys=True)
        task.state = "served"
        task.served_at = utcnow()
        self.db.commit()
        return task

    def acknowledge_evaluation(self, *, task: EvaluationTask, user: User) -> EvaluationTask:
        """The requester signs that the service happened; credit becomes attested."""
        self.require_member(user)
        if user.institution_id != task.requester_institution_id:
            raise HubError(403, "not_requester", "only the requesting institution may acknowledge")
        if task.state != "served":
            raise HubError(409, "bad_state", f"task is {task.state}, not served")
        requester, evaluator = task.requester, task.evaluator
        with self.tp.lock:
            signer = self.tp.institution_signer(requester.slug)
            attestation = ServiceAttestation.create(
                signer,
                actor=evaluator.slug,
                counterparty=requester.slug,
                kind=KIND_EVALUATION_SERVED,
                ref=task.bundle_id,
            )
            try:
                self.tp.credit.record_evaluation_served(
                    evaluator.slug,
                    counterparty=requester.slug,
                    attestation=attestation,
                    institution=evaluator.slug,
                    ref=task.bundle_id,
                    detail=f"task:{task.id}",
                )
            except AttestationRejectedError as exc:
                raise HubError(409, "attestation_rejected", str(exc))
        task.state = "acknowledged"
        task.acknowledged_at = utcnow()
        self.db.commit()
        return task

    def list_evaluations(self, *, institution_id: Optional[int] = None) -> List[EvaluationTask]:
        query = select(EvaluationTask).order_by(EvaluationTask.created_at.desc())
        if institution_id is not None:
            query = query.where(
                (EvaluationTask.requester_institution_id == institution_id)
                | (EvaluationTask.evaluator_institution_id == institution_id)
            )
        return list(self.db.scalars(query))

    def get_evaluation(self, task_id: int) -> EvaluationTask:
        task = self.db.get(EvaluationTask, task_id)
        if task is None:
            raise HubError(404, "task_not_found", f"no evaluation task {task_id}")
        return task

    @staticmethod
    def evaluation_summary(task: EvaluationTask) -> Dict[str, Any]:
        return {
            "id": task.id,
            "bundle_id": task.bundle_id,
            "requester": task.requester.slug,
            "evaluator": task.evaluator.slug,
            "state": task.state,
            "reason_code": task.reason_code,
            "detail": task.detail,
            "report": json.loads(task.report) if task.report else None,
            "created_at": task.created_at.isoformat(),
            "served_at": task.served_at.isoformat() if task.served_at else None,
            "acknowledged_at": task.acknowledged_at.isoformat() if task.acknowledged_at else None,
        }

    def reciprocity_status(self, institution: Institution) -> Dict[str, Any]:
        served = self.tp.credit.served_evaluations(institution.slug, attested_only=True)
        prior = self.db.scalar(
            select(func.count(EvaluationTask.id)).where(
                EvaluationTask.requester_institution_id == institution.id,
                EvaluationTask.state != "refused",
            )
        )
        return {
            "evaluations_served_attested": served,
            "required": self.settings.reciprocity_min_evaluations_served,
            "grace_requests_remaining": max(0, self.settings.reciprocity_grace_requests - prior),
        }

    # ================================================================ metrics

    def metrics(self) -> Dict[str, Any]:
        """The charter's quarterly metrics, each with its denominator.

        *External* means an institution flagged as not founding.
        """
        now = utcnow()
        institutions = self.list_institutions()
        external = [i for i in institutions if not i.is_founding]
        external_ids = {i.id for i in external}
        installations = [i for i in external if i.node_verified]
        submissions = list(self.db.scalars(select(Submission)))
        reviews = list(self.db.scalars(select(Review).where(Review.accepted.is_(True))))
        tasks = list(self.db.scalars(select(EvaluationTask).where(EvaluationTask.state == "acknowledged")))

        installing_ids = {i.id for i in installations}
        converted = {s.institution_id for s in submissions if s.institution_id in installing_ids}

        # Ninety-day retention over contributors (users) whose first accepted
        # contribution is at least ninety days old.
        first: Dict[int, datetime] = {}
        activity: Dict[int, List[datetime]] = {}

        def note(user_id: int, when: datetime, accepted: bool) -> None:
            activity.setdefault(user_id, []).append(when)
            if accepted and (user_id not in first or when < first[user_id]):
                first[user_id] = when

        for s in submissions:
            note(s.submitted_by_id, s.created_at, s.certified_at is not None)
        for r in reviews:
            if r.user_id is not None:
                note(r.user_id, r.created_at, True)
        for t in tasks:
            note(t.requested_by_id, t.created_at, False)
        window_closed = {u: d for u, d in first.items() if d <= now - timedelta(days=90)}
        retained = [
            u for u, d in window_closed.items()
            if any(a >= d + timedelta(days=90) for a in activity.get(u, []))
        ]

        users = list(self.db.scalars(select(User)))
        independent_maintainers = [
            u for u in users if u.has_role("maintainer") and u.institution_id in external_ids
        ]
        certified = [s for s in submissions if s.certified_at is not None]
        quarters: Dict[str, Dict[str, int]] = {}
        for s in submissions:
            key = f"{s.created_at.year}-Q{(s.created_at.month - 1) // 3 + 1}"
            row = quarters.setdefault(key, {"submissions": 0, "certified": 0})
            row["submissions"] += 1
            if s.certified_at is not None:
                row["certified"] += 1

        def ratio(n: int, d: int) -> Optional[float]:
            return round(n / d, 3) if d else None

        return {
            "generated_at": now.isoformat(),
            "external_definition": "institutions not flagged as founding",
            "independent_installations": {
                "numerator": len(installations),
                "denominator": len(external),
                "ratio": ratio(len(installations), len(external)),
                "note": "external institutions whose node completed the verification handshake",
            },
            "first_contribution_conversion": {
                "numerator": len(converted),
                "denominator": len(installations),
                "ratio": ratio(len(converted), len(installations)),
                "note": "external installations that submitted a model, over external installations",
            },
            "ninety_day_retention": {
                "numerator": len(retained),
                "denominator": len(window_closed),
                "ratio": ratio(len(retained), len(window_closed)),
                "note": "contributors active 90 days after their first accepted contribution, over all whose window has closed",
            },
            "accepted_contributions": len(certified) + len(reviews) + len(tasks),
            "model_submissions": len(submissions),
            "certified_models": len(certified),
            "independent_maintainers": len(independent_maintainers),
            "registered_institutions": len(institutions),
            "registered_users": len(users),
            "downloads": self.db.scalar(select(func.count(Download.id))) or 0,
            "quarters": dict(sorted(quarters.items())),
        }

    def standing(self) -> Dict[str, Any]:
        report = self.tp.credit.standing_report()
        return {
            "generated_at": report.generated_at,
            "ledger_verified": report.ledger_verified,
            "rows": [row.to_dict() for row in report.rows],
            "markdown": report.to_markdown(),
        }


def _new_invite_code() -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(secrets.choice(alphabet) for _ in range(10))


def gate_verdict(report: Any) -> str:
    """Collapse a quality report to pass / warn / fail for the workflow.

    ``QualityReport.overall_status`` ranks ``skip`` above ``pass`` so that a
    check that could not run is never mistaken for one that passed; for the
    submission queue, a skipped check is a note, not a verdict, so it does not
    downgrade a passing gate. The full report keeps every ``skip``.
    """
    if report.blocking:
        return "fail"
    counts = report.counts
    if counts.get("warn"):
        return "warn"
    return "pass"


def _optional_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        arr = np.asarray(list(value), dtype=float)
    except (TypeError, ValueError):
        return None
    return arr if arr.size else None


__all__ = ["HubServices", "HubError", "GATE_ACTOR", "HUB_ACTOR", "WEIGHT_FORMATS", "load_weights_array", "gate_verdict"]
