"""Application factory."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from openmed_hub.api import hub_error_response, router as api_router
from openmed_hub.config import HubSettings
from openmed_hub.db import Base, ensure_columns, make_engine, make_session_factory
from openmed_hub.services import HubError
from openmed_hub.trustplane import TrustPlane
from openmed_hub.web import router as web_router

STATIC_DIR = Path(__file__).parent / "static"


def create_app(settings: Optional[HubSettings] = None) -> FastAPI:
    """Build the hub: DB, trust plane, API and web routes."""
    settings = settings or HubSettings.from_env()
    engine = make_engine(settings.db_path)
    Base.metadata.create_all(engine)
    ensure_columns(engine)
    session_factory = make_session_factory(engine)

    trustplane = TrustPlane(settings)
    with session_factory() as db:
        trustplane.load_from_db(db)

    app = FastAPI(
        title=settings.hub_name,
        description="A governed commons for clinical AI models: attested submission, "
        "automated gate, multi-institution certification, tamper-evident lineage.",
        version="0.1.0",
    )
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.trustplane = trustplane

    @app.exception_handler(HubError)
    async def _hub_error(request: Request, exc: HubError):
        if request.url.path.startswith("/api/"):
            return hub_error_response(exc)
        # Web routes: bounce back with the message in the query string.
        back = request.headers.get("referer") or "/"
        sep = "&" if "?" in back else "?"
        return RedirectResponse(url=f"{back}{sep}error={exc.detail}", status_code=303)

    @app.middleware("http")
    async def _security_headers(request: Request, call_next):
        response = await call_next(request)
        headers = response.headers
        headers.setdefault("X-Content-Type-Options", "nosniff")
        headers.setdefault("X-Frame-Options", "DENY")
        headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        if not request.url.path.startswith(("/docs", "/redoc", "/openapi.json")):
            headers.setdefault(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
                "script-src 'self'; form-action 'self'; frame-ancestors 'none'; base-uri 'self'",
            )
        if settings.secure_cookies:
            headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        return response

    app.include_router(api_router)
    app.include_router(web_router)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/healthz", include_in_schema=False)
    def healthz():
        return JSONResponse({"ok": True, "hub": settings.hub_name})

    return app


__all__ = ["create_app"]
