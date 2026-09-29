"""Time-based one-time passwords (RFC 6238) and recovery codes, stdlib only."""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import struct
import time
from typing import Iterable, List, Optional
from urllib.parse import quote

STEP_SECONDS = 30
DIGITS = 6


def new_secret() -> str:
    """A 160-bit secret, base32 without padding (what authenticator apps take)."""
    return base64.b32encode(secrets.token_bytes(20)).decode("ascii").rstrip("=")


def _key(secret: str) -> bytes:
    padded = secret.strip().replace(" ", "").upper()
    padded += "=" * (-len(padded) % 8)
    return base64.b32decode(padded)


def totp(secret: str, for_time: Optional[float] = None, *, step: int = STEP_SECONDS, digits: int = DIGITS) -> str:
    counter = int((time.time() if for_time is None else for_time) // step)
    digest = hmac.new(_key(secret), struct.pack(">Q", counter), hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    code = (struct.unpack(">I", digest[offset : offset + 4])[0] & 0x7FFFFFFF) % (10 ** digits)
    return str(code).zfill(digits)


def verify_totp(secret: str, code: str, *, window: int = 1, for_time: Optional[float] = None) -> bool:
    """Accept the current code and ``window`` steps either side (clock drift)."""
    code = (code or "").strip().replace(" ", "")
    if not code.isdigit() or len(code) != DIGITS:
        return False
    now = time.time() if for_time is None else for_time
    ok = False
    for delta in range(-window, window + 1):
        expected = totp(secret, now + delta * STEP_SECONDS)
        ok |= hmac.compare_digest(expected, code)
    return ok


def otpauth_uri(secret: str, account: str, issuer: str) -> str:
    return (
        f"otpauth://totp/{quote(issuer)}:{quote(account)}"
        f"?secret={secret}&issuer={quote(issuer)}&algorithm=SHA1&digits={DIGITS}&period={STEP_SECONDS}"
    )


def qr_svg(uri: str) -> Optional[str]:
    """Inline SVG of the enrolment QR code, or ``None`` when ``segno`` is absent."""
    try:
        import io

        import segno
    except ImportError:
        return None
    buf = io.BytesIO()
    segno.make(uri, error="m").save(buf, kind="svg", scale=4, xmldecl=False, svgns=True, border=2)
    return buf.getvalue().decode("utf-8")


def new_recovery_codes(n: int = 10) -> List[str]:
    """Single-use recovery codes, shown once. Store only their hashes."""
    alphabet = "abcdefghjkmnpqrstuvwxyz23456789"
    out = []
    for _ in range(n):
        raw = "".join(secrets.choice(alphabet) for _ in range(12))
        out.append(f"{raw[:4]}-{raw[4:8]}-{raw[8:]}")
    return out


def normalize_recovery(code: str) -> str:
    return (code or "").strip().lower().replace("-", "").replace(" ", "")


def hash_recovery(code: str) -> str:
    return hashlib.sha256(("recovery|" + normalize_recovery(code)).encode("utf-8")).hexdigest()


def consume_recovery(code: str, hashes: Iterable[str]) -> Optional[List[str]]:
    """Return the remaining hashes if ``code`` matched one, else ``None``."""
    target = hash_recovery(code)
    remaining = list(hashes)
    for i, h in enumerate(remaining):
        if hmac.compare_digest(h, target):
            del remaining[i]
            return remaining
    return None


__all__ = [
    "consume_recovery",
    "hash_recovery",
    "new_recovery_codes",
    "new_secret",
    "normalize_recovery",
    "otpauth_uri",
    "qr_svg",
    "totp",
    "verify_totp",
]
