"""Password hashing, API tokens and signed session cookies (stdlib only)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import time
from typing import Optional

PBKDF2_ITERATIONS = 200_000
SESSION_COOKIE = "openmed_session"


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


def new_token() -> str:
    """Return a fresh API token (shown to the user once)."""
    return "omh_" + secrets.token_urlsafe(32)


def token_hash(token: str) -> str:
    """Hash a token for storage; the plaintext is never stored."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _sign(secret: str, message: str) -> str:
    return hmac.new(secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()


def make_session(secret: str, user_id: int, ttl_seconds: int) -> str:
    """Return a signed session value ``user_id.expires.signature``."""
    expires = int(time.time()) + int(ttl_seconds)
    body = f"{user_id}.{expires}"
    return body + "." + _sign(secret, body)


def read_session(secret: str, value: Optional[str]) -> Optional[int]:
    """Return the user id carried by a valid, unexpired session value."""
    if not value:
        return None
    parts = value.split(".")
    if len(parts) != 3:
        return None
    body = parts[0] + "." + parts[1]
    if not hmac.compare_digest(_sign(secret, body), parts[2]):
        return None
    try:
        user_id = int(parts[0])
        expires = int(parts[1])
    except ValueError:
        return None
    if expires < time.time():
        return None
    return user_id


__all__ = [
    "SESSION_COOKIE",
    "hash_password",
    "verify_password",
    "new_token",
    "token_hash",
    "make_session",
    "read_session",
]
