"""FastAPI dependencies shared by the JSON API and the HTML routes."""

from __future__ import annotations

from typing import Iterator, Optional

from fastapi import Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from openmed_hub.db import ApiToken, User, utcnow
from openmed_hub.security import SESSION_COOKIE, read_session, token_hash
from openmed_hub.services import HubServices


def client_ip(request: Request) -> str:
    """The peer address; honour X-Forwarded-For only behind a trusted proxy."""
    if request.app.state.settings.trust_proxy_headers:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else ""


def get_db(request: Request) -> Iterator[Session]:
    db = request.app.state.session_factory()
    try:
        yield db
    finally:
        db.close()


def get_services(request: Request, db: Session = Depends(get_db)) -> HubServices:
    return HubServices(request.app.state.trustplane, request.app.state.settings, db, client_ip=client_ip(request))


def optional_user(request: Request, db: Session = Depends(get_db)) -> Optional[User]:
    """Resolve the caller from a Bearer token or the session cookie.

    A session cookie is valid only while its security stamp matches the
    user's current one, so changing the password or the second factor signs
    every other session out.
    """
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        token = header[7:].strip()
        row = db.scalar(select(ApiToken).where(ApiToken.token_hash == token_hash(token)))
        if row is not None:
            user = db.get(User, row.user_id)
            if user is not None and user.is_active:
                row.last_used_at = utcnow()
                db.commit()
                return user
        return None
    parsed = read_session(request.app.state.settings.secret_key, request.cookies.get(SESSION_COOKIE))
    if parsed is None:
        return None
    user_id, stamp = parsed
    user = db.get(User, user_id)
    if user is None or not user.is_active or user.stamp != stamp:
        return None
    return user


def current_user(user: Optional[User] = Depends(optional_user)) -> User:
    if user is None:
        raise HTTPException(status_code=401, detail="authentication required")
    return user


def admin_user(user: User = Depends(current_user)) -> User:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="admin role required")
    return user


__all__ = ["client_ip", "get_db", "get_services", "optional_user", "current_user", "admin_user"]
