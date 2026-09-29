"""Password hashing, API tokens, signed cookies and CSRF tokens (stdlib only).

Sessions are stateless: ``user_id.expires.stamp.signature``. ``stamp`` is the
user's *security stamp*, rotated whenever the password or the second factor
changes, so every session issued before that moment stops verifying without
the hub keeping a session table.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import time
from typing import Optional, Tuple

PBKDF2_ITERATIONS = 200_000
SESSION_COOKIE = "openmed_session"
PREAUTH_COOKIE = "openmed_preauth"
CSRF_COOKIE = "openmed_csrf"
MIN_PASSWORD_LENGTH = 10


def hash_password(password: str) -> str:
    """Return a self-describing PBKDF2-HMAC-SHA256 hash."""
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return "pbkdf2_sha256${}${}${}".format(
        PBKDF2_ITERATIONS,
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(digest).decode("ascii"),
    )


def verify_password(password: str, stored: str) -> bool:
    """Check ``password`` against a hash produced by :func:`hash_password`."""
    try:
        scheme, iterations, salt_b64, digest_b64 = stored.split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(digest_b64)
    except (ValueError, TypeError):
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations))
    return hmac.compare_digest(actual, expected)


def password_problem(password: str, *, email: str = "", name: str = "") -> Optional[str]:
    """Return why a password is unacceptable, or ``None``."""
    if len(password) < MIN_PASSWORD_LENGTH:
        return f"password must be at least {MIN_PASSWORD_LENGTH} characters"
    lowered = password.lower()
    local = email.split("@")[0].lower() if email else ""
    if local and len(local) >= 4 and local in lowered:
        return "password must not contain your email address"
    if name and len(name) >= 4 and name.lower() in lowered:
        return "password must not contain your name"
    if lowered in _COMMON or lowered.rstrip("0123456789!") in _COMMON:
        return "that password is on the list of most common passwords"
    if len(set(password)) < 4:
        return "password needs more variety"
    return None


_COMMON = frozenset(
    """password passwords passw0rd p@ssword p@ssw0rd qwertyuiop qwerty1234 1234567890 12345678901
    letmein123 welcome123 admin12345 administrator iloveyou1 changeme123 openmed123 trustfed123
    abcdefghij monkey12345 dragon12345 football123 baseball123 sunshine123 princess123 password1
    password12 password123 password1234 qwerty123456 1q2w3e4r5t 1qaz2wsx3edc""".split()
)


def new_token() -> str:
    """Return a fresh API token (shown to the user once)."""
    return "omh_" + secrets.token_urlsafe(32)


def token_hash(token: str) -> str:
    """Hash a token for storage; the plaintext is never stored."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_stamp() -> str:
    return secrets.token_hex(8)


def _sign(secret: str, message: str) -> str:
    return hmac.new(secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()


def make_session(secret: str, user_id: int, ttl_seconds: int, stamp: str = "", *, purpose: str = "session") -> str:
    """Return a signed value ``user_id.expires.stamp.signature``."""
    expires = int(time.time()) + int(ttl_seconds)
    body = f"{user_id}.{expires}.{stamp}"
    return body + "." + _sign(secret, purpose + "|" + body)


def read_session(secret: str, value: Optional[str], *, purpose: str = "session") -> Optional[Tuple[int, str]]:
    """Return ``(user_id, stamp)`` for a valid, unexpired value."""
    if not value:
        return None
    parts = value.split(".")
    if len(parts) != 4:
        return None
    body = ".".join(parts[:3])
    if not hmac.compare_digest(_sign(secret, purpose + "|" + body), parts[3]):
        return None
    try:
        user_id = int(parts[0])
        expires = int(parts[1])
    except ValueError:
        return None
    if expires < time.time():
        return None
    return user_id, parts[2]


def new_csrf_token() -> str:
    return secrets.token_urlsafe(24)


def csrf_matches(cookie_value: Optional[str], form_value: Optional[str]) -> bool:
    if not cookie_value or not form_value:
        return False
    return hmac.compare_digest(cookie_value, form_value)


__all__ = [
    "CSRF_COOKIE",
    "MIN_PASSWORD_LENGTH",
    "PREAUTH_COOKIE",
    "SESSION_COOKIE",
    "csrf_matches",
    "hash_password",
    "make_session",
    "new_csrf_token",
    "new_stamp",
    "new_token",
    "password_problem",
    "read_session",
    "token_hash",
    "verify_password",
]
