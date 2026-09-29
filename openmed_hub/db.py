"""Relational state of the hub: accounts, nodes, workflow rows.

The *authoritative* records of the commons live on the hash-chained ledgers
(registry, certification, credit). What lives here is the operational state a
web service needs -- who can log in, which node keys verified, which
submission is in which queue -- plus cached views that are cheap to rebuild.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

ROLE_CONTRIBUTOR = "contributor"
ROLE_REVIEWER_TECHNICAL = "reviewer_technical"
ROLE_REVIEWER_CLINICAL = "reviewer_clinical"
ROLE_MAINTAINER = "maintainer"
ROLE_ADMIN = "admin"
ALL_ROLES = (
    ROLE_CONTRIBUTOR,
    ROLE_REVIEWER_TECHNICAL,
    ROLE_REVIEWER_CLINICAL,
    ROLE_MAINTAINER,
    ROLE_ADMIN,
)
REVIEWER_ROLES = (ROLE_REVIEWER_TECHNICAL, ROLE_REVIEWER_CLINICAL)

INSTITUTION_KINDS = ("health_system", "medical_school", "research_lab", "company", "other")


def utcnow() -> datetime:
    """Naive UTC timestamp (SQLite stores no zone)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class Institution(Base):
    __tablename__ = "institutions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    slug: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    kind: Mapped[str] = mapped_column(String(32), default="health_system")
    country: Mapped[str] = mapped_column(String(64), default="US")
    #: Founding institutions do not count as *external* in the metrics.
    is_founding: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    #: Client-held Ed25519 public key of the site's node (hex), if registered.
    node_public_key: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    node_verified_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    #: Hub-held seed for the institution's service key, used to sign evaluation
    #: acknowledgements made through the web UI.
    service_key_seed: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    users: Mapped[List["User"]] = relationship(back_populates="institution")

    @property
    def node_verified(self) -> bool:
        return self.node_verified_at is not None


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    password_hash: Mapped[str] = mapped_column(String(300))
    institution_id: Mapped[int] = mapped_column(ForeignKey("institutions.id"))
    roles: Mapped[str] = mapped_column(String(200), default=ROLE_CONTRIBUTOR)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    #: Reviewer identity registered in the certification keyring.
    reviewer_id: Mapped[Optional[str]] = mapped_column(String(80), unique=True, nullable=True)
    #: Hub-held key seed (hex). Present when the hub signs reviews on the
    #: reviewer's behalf (the web-UI convenience path).
    reviewer_key_seed: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    #: Client-held Ed25519 public key (hex). Present when the reviewer signs
    #: with their own key and submits detached signatures.
    reviewer_public_key: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    institution: Mapped[Institution] = relationship(back_populates="users")

    @property
    def role_list(self) -> List[str]:
        return [r for r in self.roles.split(",") if r]

    def has_role(self, role: str) -> bool:
        return role in self.role_list

    @property
    def is_reviewer(self) -> bool:
        return any(self.has_role(r) for r in REVIEWER_ROLES)

    @property
    def is_admin(self) -> bool:
        return self.has_role(ROLE_ADMIN)

    @property
    def is_maintainer(self) -> bool:
        return self.has_role(ROLE_MAINTAINER) or self.is_admin

    @property
    def reviewer_role_label(self) -> str:
        parts = []
        if self.has_role(ROLE_REVIEWER_TECHNICAL):
            parts.append("technical")
        if self.has_role(ROLE_REVIEWER_CLINICAL):
            parts.append("clinical")
        return "+".join(parts) or "reviewer"


class ApiToken(Base):
    __tablename__ = "api_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    label: Mapped[str] = mapped_column(String(100), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class NodeChallenge(Base):
    __tablename__ = "node_challenges"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    institution_id: Mapped[int] = mapped_column(ForeignKey("institutions.id"), index=True)
    nonce: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    used: Mapped[bool] = mapped_column(Boolean, default=False)


class Submission(Base):
    """One uploaded bundle and where it stands in the workflow."""

    __tablename__ = "submissions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    bundle_id: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    version: Mapped[str] = mapped_column(String(64))
    submitted_by_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    institution_id: Mapped[int] = mapped_column(ForeignKey("institutions.id"), index=True)
    parent_bundle_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    weights_sha256: Mapped[str] = mapped_column(String(64))
    weights_size: Mapped[int] = mapped_column(Integer, default=0)
    weights_format: Mapped[str] = mapped_column(String(16), default="npz")
    gate_status: Mapped[str] = mapped_column(String(16), default="skip")
    gate_report: Mapped[str] = mapped_column(Text, default="{}")
    #: Mirrors the certification case state; ``blocked`` also covers a failed gate.
    state: Mapped[str] = mapped_column(String(32), default="submitted", index=True)
    identifier: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    citation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    certified_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    submitted_by: Mapped[User] = relationship()
    institution: Mapped[Institution] = relationship()
    reviews: Mapped[List["Review"]] = relationship(back_populates="submission")


class Review(Base):
    """A reviewer's signed decision (or a refused attempt) on one submission."""

    __tablename__ = "reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    submission_id: Mapped[int] = mapped_column(ForeignKey("submissions.id"), index=True)
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True)
    reviewer_id: Mapped[str] = mapped_column(String(80))
    institution: Mapped[str] = mapped_column(String(64))
    decision: Mapped[str] = mapped_column(String(16))
    statement: Mapped[str] = mapped_column(Text, default="")
    signed_at: Mapped[str] = mapped_column(String(40), default="")
    key_id: Mapped[str] = mapped_column(String(80), default="")
    signature: Mapped[str] = mapped_column(Text, default="")
    #: True when the hub signed with a key it holds (web path); False when the
    #: reviewer supplied a detached signature made with their own key.
    hub_signed: Mapped[bool] = mapped_column(Boolean, default=True)
    accepted: Mapped[bool] = mapped_column(Boolean, default=False)
    refusal_code: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    refusal_detail: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    submission: Mapped[Submission] = relationship(back_populates="reviews")
    user: Mapped[Optional[User]] = relationship()


class EvaluationTask(Base):
    """A request for another site to evaluate a bundle on its own cohort."""

    __tablename__ = "evaluation_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    bundle_id: Mapped[str] = mapped_column(String(100), index=True)
    requester_institution_id: Mapped[int] = mapped_column(ForeignKey("institutions.id"))
    evaluator_institution_id: Mapped[int] = mapped_column(ForeignKey("institutions.id"))
    requested_by_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    #: requested | refused | served | acknowledged
    state: Mapped[str] = mapped_column(String(16), default="requested", index=True)
    reason_code: Mapped[str] = mapped_column(String(64), default="")
    detail: Mapped[str] = mapped_column(Text, default="")
    report: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    served_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    acknowledged_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    requester: Mapped[Institution] = relationship(foreign_keys=[requester_institution_id])
    evaluator: Mapped[Institution] = relationship(foreign_keys=[evaluator_institution_id])
    requested_by: Mapped[User] = relationship()


class ApprovedMeasurement(Base):
    """A software measurement (pipeline release) the attestation gate accepts."""

    __tablename__ = "approved_measurements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    measurement: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    label: Mapped[str] = mapped_column(String(200), default="")
    manifest: Mapped[str] = mapped_column(Text, default="{}")
    added_by_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True)
    added_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    @property
    def active(self) -> bool:
        return self.revoked_at is None


class Download(Base):
    __tablename__ = "downloads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    bundle_id: Mapped[str] = mapped_column(String(100), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    institution_id: Mapped[int] = mapped_column(ForeignKey("institutions.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


def make_engine(path):
    """SQLite engine for ``path`` (a :class:`pathlib.Path` or ``":memory:"``)."""
    url = "sqlite://" if str(path) == ":memory:" else f"sqlite:///{path}"
    return create_engine(url, connect_args={"check_same_thread": False}, future=True)


def make_session_factory(engine):
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


__all__ = [
    "ALL_ROLES",
    "INSTITUTION_KINDS",
    "REVIEWER_ROLES",
    "ROLE_ADMIN",
    "ROLE_CONTRIBUTOR",
    "ROLE_MAINTAINER",
    "ROLE_REVIEWER_CLINICAL",
    "ROLE_REVIEWER_TECHNICAL",
    "ApiToken",
    "ApprovedMeasurement",
    "Base",
    "Download",
    "EvaluationTask",
    "Institution",
    "NodeChallenge",
    "Review",
    "Submission",
    "User",
    "make_engine",
    "make_session_factory",
    "utcnow",
]
