"""Hub settings, read from the environment with sane local defaults."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

#: Code identities whose measurement the attestation gate approves by default.
DEFAULT_CODE_IDENTITIES: Tuple[str, ...] = ("openmed-training-pipeline@v1",)


def _load_or_create_secret(path: Path) -> str:
    """Return the secret stored at ``path``, creating a fresh one if absent."""
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    value = secrets.token_hex(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:  # pragma: no cover - platform dependent
        pass
    return value


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else default


@dataclass
class HubSettings:
    """Everything the hub needs to run, in one place.

    Governance parameters mirror the charter (see ``docs/governance.html``):
    certification needs ``certification_k`` approvals from at least
    ``certification_min_institutions`` distinct institutions, and the
    model-review board must include ``technical_reviewers_min`` technical and
    ``clinical_reviewers_min`` clinical reviewers. A site may receive a
    multi-site evaluation only after serving as an evaluator
    ``reciprocity_min_evaluations_served`` times, with
    ``reciprocity_grace_requests`` free requests for a newly joined site.
    """

    data_dir: Path
    secret_key: str
    attestation_root_key: bytes
    approved_code_identities: Tuple[str, ...] = DEFAULT_CODE_IDENTITIES
    #: ``nodekey``: submissions carry a quote signed at the site with its
    #: verified node key over a software measurement of the code that ran
    #: (the production setting). ``mock``: the hub signs a declared code
    #: identity; demo and development only.
    attestation_mode: str = "nodekey"
    #: Measurement digests approved at startup, in addition to those the admin
    #: approves at runtime (kept in the database).
    approved_measurements: Tuple[str, ...] = ()
    certification_k: int = 2
    certification_min_institutions: int = 2
    technical_reviewers_min: int = 2
    clinical_reviewers_min: int = 2
    reciprocity_min_evaluations_served: int = 2
    reciprocity_grace_requests: int = 1
    attestation_max_age_seconds: int = 900
    max_upload_bytes: int = 512 * 1024 * 1024
    session_ttl_seconds: int = 14 * 24 * 3600
    hub_name: str = "OpenMed Hub"
    hub_url: str = "http://127.0.0.1:8000"
    #: Roles that must have two-factor authentication enabled before they can
    #: act in that role (sign reviews, grant roles, approve measurements, ...).
    require_2fa_roles: Tuple[str, ...] = ("admin", "maintainer", "reviewer_technical", "reviewer_clinical")
    #: Login throttling: lock after this many consecutive failures, for a
    #: growing delay capped at ``lockout_max_seconds``.
    lockout_threshold: int = 5
    lockout_max_seconds: int = 900
    #: Per-IP login attempts allowed per ten minutes.
    login_rate_limit: int = 60
    #: Mark cookies Secure (set automatically when ``hub_url`` is https).
    secure_cookies: bool = False
    #: Read the client address from X-Forwarded-For (only behind a proxy you control).
    trust_proxy_headers: bool = False

    @classmethod
    def from_env(cls, data_dir: Optional[os.PathLike] = None) -> "HubSettings":
        """Build settings from ``OPENMED_*`` environment variables."""
        root = Path(data_dir or os.environ.get("OPENMED_DATA_DIR", "openmed-data")).resolve()
        root.mkdir(parents=True, exist_ok=True)
        secret = os.environ.get("OPENMED_SECRET_KEY") or _load_or_create_secret(
            root / "secret.key"
        )
        att = os.environ.get("OPENMED_ATTESTATION_ROOT_KEY") or _load_or_create_secret(
            root / "attestation-root.key"
        )
        raw_ids = os.environ.get("OPENMED_APPROVED_CODE_IDENTITIES", "")
        identities = tuple(x.strip() for x in raw_ids.split(",") if x.strip())
        raw_measurements = os.environ.get("OPENMED_APPROVED_MEASUREMENTS", "")
        measurements = tuple(x.strip().lower() for x in raw_measurements.split(",") if x.strip())
        hub_url = os.environ.get("OPENMED_HUB_URL", "http://127.0.0.1:8000")
        mode = os.environ.get("OPENMED_ATTESTATION_MODE", "nodekey").strip().lower()
        if mode not in ("nodekey", "mock"):
            raise ValueError("OPENMED_ATTESTATION_MODE must be 'nodekey' or 'mock'")
        return cls(
            data_dir=root,
            secret_key=secret,
            attestation_root_key=att.encode("utf-8"),
            approved_code_identities=identities or DEFAULT_CODE_IDENTITIES,
            attestation_mode=mode,
            approved_measurements=measurements,
            certification_k=_env_int("OPENMED_CERT_K", 2),
            certification_min_institutions=_env_int("OPENMED_CERT_MIN_INSTITUTIONS", 2),
            technical_reviewers_min=_env_int("OPENMED_TECHNICAL_REVIEWERS_MIN", 2),
            clinical_reviewers_min=_env_int("OPENMED_CLINICAL_REVIEWERS_MIN", 2),
            reciprocity_min_evaluations_served=_env_int("OPENMED_RECIPROCITY_MIN_SERVED", 2),
            reciprocity_grace_requests=_env_int("OPENMED_RECIPROCITY_GRACE", 1),
            max_upload_bytes=_env_int("OPENMED_MAX_UPLOAD_BYTES", 512 * 1024 * 1024),
            hub_name=os.environ.get("OPENMED_HUB_NAME", "OpenMed Hub"),
            hub_url=hub_url,
            require_2fa_roles=tuple(
                r.strip() for r in os.environ.get(
                    "OPENMED_REQUIRE_2FA_ROLES", "admin,maintainer,reviewer_technical,reviewer_clinical"
                ).split(",") if r.strip()
            ),
            lockout_threshold=_env_int("OPENMED_LOCKOUT_THRESHOLD", 5),
            lockout_max_seconds=_env_int("OPENMED_LOCKOUT_MAX_SECONDS", 900),
            login_rate_limit=_env_int("OPENMED_LOGIN_RATE_LIMIT", 60),
            secure_cookies=os.environ.get("OPENMED_SECURE_COOKIES", "").lower() in ("1", "true", "yes")
            or hub_url.lower().startswith("https://"),
            trust_proxy_headers=os.environ.get("OPENMED_TRUST_PROXY", "").lower() in ("1", "true", "yes"),
        )

    @property
    def db_path(self) -> Path:
        return self.data_dir / "hub.db"

    @property
    def ledger_dir(self) -> Path:
        return self.data_dir / "ledgers"

    @property
    def blob_dir(self) -> Path:
        return self.data_dir / "blobs"


__all__ = ["HubSettings", "DEFAULT_CODE_IDENTITIES"]
