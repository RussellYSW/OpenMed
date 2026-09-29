"""FastAPI dependencies shared by the JSON API and the HTML routes."""

from __future__ import annotations

from typing import Iterator, Optional

from fastapi import Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from openmed_hub.db import ApiToken, User, utcnow
from openmed_hub.security import SESSION_COOKIE, read_session, token_hash
from openmed_hub.services import HubServices


def get_db(request: Request) -> Iterator[Session]:
    db = request.app.state.session_factory()
    try:
        yield db
    finally:
        db.close()


def get_services(request: Request, db: Session = Depends(get_db)) -> HubServices:
    return HubServices(request.app.state.trustplane, request.app.state.settings, db)


def optional_user(request: Request, db: Session = Depends(get_db)) -> Optional[User]:
    """Resolve the caller from a Bearer token or the session cookie."""
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
    user_id = read_session(request.app.state.settings.secret_key, request.cookies.get(SESSION_COOKIE))
    if user_id is None:
        return None
    user = db.get(User, user_id)
    return user if user is not None and user.is_active else None


def current_user(user: Optional[User] = Depends(optional_user)) -> User:
    if user is None:
        raise HTTPException(status_code=401, detail="authentication required")
    return user


def admin_user(user: User = Depends(current_user)) -> User:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="admin role required")
    return user


__all__ = ["get_db", "get_services", "optional_user", "current_user", "admin_user"]
