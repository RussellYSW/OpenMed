"""HTML routes: the pages a clinician or engineer uses in a browser."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select

from openmed_hub.db import ALL_ROLES, INSTITUTION_KINDS, ApiToken, Institution, User
from openmed_hub.deps import get_db, get_services, optional_user
from openmed_hub.security import (
    CSRF_COOKIE,
    PREAUTH_COOKIE,
    SESSION_COOKIE,
    csrf_matches,
    make_session,
    new_csrf_token,
    read_session,
)
from openmed_hub.services import HubError, HubServices

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


async def verify_csrf(request: Request) -> None:
    """Double-submit CSRF check for every HTML form post.

    The token lives in a cookie and must be echoed in the form; a cross-site
    page cannot read the cookie to forge the field. (Session cookies are also
    SameSite=Lax, so this is the second layer, not the only one.)
    """
    if request.method != "POST":
        return
    form = await request.form()
    if not csrf_matches(request.cookies.get(CSRF_COOKIE), form.get("csrf")):
        raise HubError(403, "csrf", "the form token is missing or stale; reload the page and try again")


router = APIRouter(include_in_schema=False, dependencies=[Depends(verify_csrf)])

EXAMPLE_CARD: Dict[str, Any] = {
    "model_details": {
        "name": "cognitive-decline-risk",
        "version": "1.0.0",
        "owner": "your-institution-slug",
        "date": "2026-09-01",
        "model_type": "logistic regression on tabular features",
        "license": "OpenRAIL-M (reciprocal)",
    },
    "intended_use": {
        "primary_use": "research: flag 24-month cognitive decline risk",
        "out_of_scope": "diagnosis, treatment selection, deployment without local validation",
    },
    "factors": {"groups": ["site", "age_band", "sex"], "instrumentation": "wearable + clinical records"},
    "metrics": {"reported": ["auc", "accuracy"], "decision_threshold": 0.5},
    "evaluation_data": {"dataset": "held-out split, single site", "n": 500},
    "training_data": {"dataset": "institutional cohort (data never leaves the site)", "n": 3000},
    "quantitative_analyses": {"auc": 0.81, "accuracy": 0.76},
    "ethical_considerations": {"risks": "single-site training; subgroup performance unverified elsewhere"},
    "caveats_and_recommendations": {"caveats": "requires multi-site evaluation before clinical use"},
}
EXAMPLE_EVALUATION: Dict[str, Any] = {
    "dataset_id": "site-holdout-2026",
    "n_samples": 500,
    "metrics": {"auc": 0.81, "accuracy": 0.76},
    "subgroup_metrics": {"age<65": {"auc": 0.83}, "age>=65": {"auc": 0.79}},
    "protocol": "fixed split, seed reported in the card",
    "evaluated_by": "your-institution-slug",
    "seed": 7,
}
EXAMPLE_MANUAL: Dict[str, Any] = {
    "intended_use": {
        "task": "flag cognitive decline risk within 24 months",
        "population": "adults 50+ with baseline visit",
        "care_setting": "research use only",
        "not_intended_for": "diagnosis, triage, or any clinical decision",
    },
    "data": {
        "sources": ["institutional cohort"],
        "n_records": 3000,
        "inclusion_criteria": "baseline plus 24-month follow-up",
        "label_definition": "threshold on the composite cognitive score",
    },
    "preprocessing": {
        "steps": ["drop incomplete visits", "z-score per site"],
        "normalization": "z-score with training-set statistics",
        "missing_data": "listwise deletion",
    },
    "hyperparameters": {"optimizer": "gradient descent", "learning_rate": 0.05, "epochs": 25, "batch_size": "full"},
    "failure_modes": {"known_failure_modes": ["site shift"], "monitoring": "subgroup AUC review before reuse"},
    "clinical_caveats": {
        "human_oversight": "advisory only; a clinician decides",
        "contraindications": "do not use outside the stated population",
        "escalation": "notify the site lead if drift is observed",
    },
}


def render(request: Request, name: str, user: Optional[User], **context: Any) -> HTMLResponse:
    settings = request.app.state.settings
    csrf_token = request.cookies.get(CSRF_COOKIE) or new_csrf_token()
    context.update(
        request=request,
        user=user,
        settings=settings,
        csrf_token=csrf_token,
        error=request.query_params.get("error"),
        message=request.query_params.get("message"),
    )
    response = TEMPLATES.TemplateResponse(request, name, context)
    if request.cookies.get(CSRF_COOKIE) != csrf_token:
        response.set_cookie(CSRF_COOKIE, csrf_token, samesite="lax", secure=settings.secure_cookies, max_age=7 * 86400)
    return response


def _redirect(url: str, message: Optional[str] = None, error: Optional[str] = None) -> RedirectResponse:
    if message:
        url += ("&" if "?" in url else "?") + "message=" + quote(message)
    if error:
        url += ("&" if "?" in url else "?") + "error=" + quote(error)
    return RedirectResponse(url=url, status_code=303)


def _login_redirect(request: Request) -> RedirectResponse:
    return _redirect("/login?next=" + quote(str(request.url.path)), error="please sign in first")


# ------------------------------------------------------------------- public


@router.get("/", response_class=HTMLResponse)
def home(request: Request, user: Optional[User] = Depends(optional_user), svc: HubServices = Depends(get_services)):
    certified = svc.list_submissions(state="certified")[:6]
    metrics = svc.metrics()
    return render(request, "index.html", user, certified=certified, metrics=metrics, institutions=svc.list_institutions())


@router.get("/register", response_class=HTMLResponse)
def register_form(request: Request, user: Optional[User] = Depends(optional_user)):
    return render(request, "register.html", user, kinds=INSTITUTION_KINDS)


@router.post("/register")
def register_submit(
    request: Request,
    name: str = Form(...),
    slug: str = Form(""),
    kind: str = Form("health_system"),
    country: str = Form("US"),
    email: str = Form(...),
    user_name: str = Form(""),
    password: str = Form(...),
    svc: HubServices = Depends(get_services),
):
    _, user = svc.register_institution(
        name=name, slug=slug or None, kind=kind, country=country,
        admin_email=email, admin_name=user_name, admin_password=password,
    )
    svc.complete_login(user)
    response = _redirect("/account", message="institution registered; next, enable two-factor authentication and register your node key")
    _set_session(request, response, user)
    return response


@router.get("/join", response_class=HTMLResponse)
def join_form(request: Request, user: Optional[User] = Depends(optional_user), svc: HubServices = Depends(get_services)):
    return render(request, "join.html", user, institutions=svc.list_institutions())


@router.post("/join")
def join_submit(
    request: Request,
    institution_slug: str = Form(...),
    email: str = Form(...),
    user_name: str = Form(""),
    password: str = Form(...),
    invite_code: str = Form(""),
    svc: HubServices = Depends(get_services),
):
    institution = svc.institution_by_slug(institution_slug)
    user = svc.register_user(
        email=email, name=user_name, password=password, institution=institution, invite_code=invite_code or None
    )
    svc.complete_login(user)
    if user.membership_active:
        response = _redirect("/account", message="account created")
    else:
        response = _redirect("/account", message="account created; an admin of your institution has to approve your membership before you can act for it")
    _set_session(request, response, user)
    return response


@router.get("/login", response_class=HTMLResponse)
def login_form(request: Request, user: Optional[User] = Depends(optional_user)):
    return render(request, "login.html", user, next=request.query_params.get("next", "/"))


@router.post("/login")
def login_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    next: str = Form("/"),
    svc: HubServices = Depends(get_services),
):
    user = svc.authenticate(email, password)
    target = next if next.startswith("/") else "/"
    settings = request.app.state.settings
    if user.totp_enabled:
        # Second step: a short-lived pre-auth cookie names the user; nothing
        # else is granted until the code checks out.
        response = _redirect("/login/2fa?next=" + quote(target))
        response.set_cookie(
            PREAUTH_COOKIE,
            make_session(settings.secret_key, user.id, 300, user.stamp, purpose="preauth"),
            httponly=True, samesite="lax", secure=settings.secure_cookies, max_age=300,
        )
        return response
    svc.complete_login(user)
    response = _redirect(target, message=f"signed in as {user.email}")
    _set_session(request, response, user)
    return response


@router.get("/login/2fa", response_class=HTMLResponse)
def login_2fa_form(request: Request):
    if _preauth_user_id(request) is None:
        return _redirect("/login", error="sign in first")
    return render(request, "login_2fa.html", None, next=request.query_params.get("next", "/"))


@router.post("/login/2fa")
def login_2fa_submit(request: Request, code: str = Form(...), next: str = Form("/"), svc: HubServices = Depends(get_services)):
    user_id = _preauth_user_id(request)
    user = svc.db.get(User, user_id) if user_id is not None else None
    if user is None or not user.is_active:
        return _redirect("/login", error="sign in first")
    if not svc.second_factor_ok(user, code):
        svc._register_failure(user)
        svc.audit("login_failed", actor=user, detail="bad second factor")
        return _redirect("/login/2fa?next=" + quote(next), error="that code did not match")
    svc.complete_login(user)
    response = _redirect(next if next.startswith("/") else "/", message=f"signed in as {user.email}")
    response.delete_cookie(PREAUTH_COOKIE)
    _set_session(request, response, user)
    return response


def _preauth_user_id(request: Request) -> Optional[int]:
    parsed = read_session(request.app.state.settings.secret_key, request.cookies.get(PREAUTH_COOKIE), purpose="preauth")
    return parsed[0] if parsed else None


@router.post("/logout")
def logout(request: Request):
    response = _redirect("/", message="signed out")
    response.delete_cookie(SESSION_COOKIE)
    response.delete_cookie(PREAUTH_COOKIE)
    return response


def _set_session(request: Request, response: RedirectResponse, user: User) -> None:
    settings = request.app.state.settings
    response.set_cookie(
        SESSION_COOKIE,
        make_session(settings.secret_key, user.id, settings.session_ttl_seconds, user.stamp),
        httponly=True,
        samesite="lax",
        secure=settings.secure_cookies,
        max_age=settings.session_ttl_seconds,
    )


# ------------------------------------------------------------------- models


@router.get("/models", response_class=HTMLResponse)
def models(request: Request, state: Optional[str] = None, user: Optional[User] = Depends(optional_user), svc: HubServices = Depends(get_services)):
    return render(request, "models.html", user, submissions=svc.list_submissions(state=state or None), state=state or "")


@router.get("/models/{bundle_id:path}", response_class=HTMLResponse)
def model_detail(request: Request, bundle_id: str, user: Optional[User] = Depends(optional_user), svc: HubServices = Depends(get_services)):
    submission = svc.get_submission(bundle_id)
    detail = svc.submission_detail(submission)
    institutions = [i for i in svc.list_institutions() if i.id != submission.institution_id]
    return render(
        request, "model_detail.html", user,
        s=submission, d=detail, institutions=institutions,
        is_owner=bool(user and user.institution_id == submission.institution_id),
        gate_json=json.dumps(detail["gate"], indent=2),
        bundle_json=json.dumps(detail["bundle"], indent=2),
    )


@router.get("/submit", response_class=HTMLResponse)
def submit_form(request: Request, parent: str = "", user: Optional[User] = Depends(optional_user), svc: HubServices = Depends(get_services)):
    if user is None:
        return _login_redirect(request)
    card = dict(EXAMPLE_CARD)
    card["model_details"] = dict(card["model_details"], owner=user.institution.slug)
    evaluation = dict(EXAMPLE_EVALUATION, evaluated_by=user.institution.slug)
    return render(
        request, "submit.html", user,
        parent=parent,
        mode=svc.settings.attestation_mode,
        identities=list(svc.tp.approved_code_identities()),
        example_card=json.dumps(card, indent=2),
        example_evaluation=json.dumps(evaluation, indent=2),
        example_manual=json.dumps(EXAMPLE_MANUAL, indent=2),
    )


@router.post("/submit")
async def submit_post(
    request: Request,
    name: str = Form(...),
    version: str = Form(...),
    code_identity: str = Form(""),
    config: str = Form(""),
    model_card: str = Form(...),
    evaluation: str = Form(...),
    fine_tuning_manual: str = Form(""),
    parent_bundle_id: str = Form(""),
    tags: str = Form(""),
    weights: UploadFile = File(...),
    user: Optional[User] = Depends(optional_user),
    svc: HubServices = Depends(get_services),
):
    if user is None:
        return _login_redirect(request)
    blob = await weights.read()
    fmt = weights.filename.rsplit(".", 1)[-1] if weights.filename and "." in weights.filename else "other"
    submission = svc.submit(
        user=user, name=name, version=version, weights=blob, weights_format=fmt,
        model_card=model_card, evaluation=evaluation, manual=fine_tuning_manual or None,
        parent_bundle_id=parent_bundle_id or None, code_identity=code_identity, config=config,
        tags=[t for t in tags.split(",") if t.strip()],
    )
    return _redirect(f"/models/{submission.bundle_id}", message="bundle published and case opened")


@router.post("/models/{bundle_id:path}/review")
def review_post(request: Request, bundle_id: str, decision: str = Form(...), statement: str = Form(""), user: Optional[User] = Depends(optional_user), svc: HubServices = Depends(get_services)):
    if user is None:
        return _login_redirect(request)
    submission = svc.get_submission(bundle_id)
    row = svc.review(submission=submission, user=user, decision=decision, statement=statement)
    if row.accepted:
        return _redirect(f"/models/{submission.bundle_id}", message=f"review recorded ({decision}); case is {submission.state}")
    return _redirect(f"/models/{submission.bundle_id}", error=f"review refused: {row.refusal_detail}")


@router.post("/models/{bundle_id:path}/action")
def action_post(request: Request, bundle_id: str, action: str = Form(...), text: str = Form(""), upheld: str = Form(""), user: Optional[User] = Depends(optional_user), svc: HubServices = Depends(get_services)):
    if user is None:
        return _login_redirect(request)
    submission = svc.get_submission(bundle_id)
    if action == "dispute":
        svc.dispute(submission=submission, user=user, reason=text)
    elif action == "remediate":
        svc.remediate(submission=submission, user=user, note=text)
    elif action == "appeal":
        svc.appeal(submission=submission, user=user, grounds=text)
    elif action == "resolve_appeal":
        svc.resolve_appeal(submission=submission, user=user, upheld=upheld == "yes", note=text)
    elif action == "revoke":
        svc.revoke(submission=submission, user=user, reason=text)
    else:
        raise HubError(400, "unknown_action", f"unknown action {action!r}")
    return _redirect(f"/models/{submission.bundle_id}", message=f"{action} recorded; case is {submission.state}")


@router.get("/models/{bundle_id:path}/download")
def download(request: Request, bundle_id: str, user: Optional[User] = Depends(optional_user), svc: HubServices = Depends(get_services)):
    if user is None:
        return _login_redirect(request)
    submission = svc.get_submission(bundle_id)
    path = svc.download(submission=submission, user=user)
    from fastapi.responses import FileResponse

    return FileResponse(path, media_type="application/octet-stream", filename=f"{submission.name}-{submission.version}.{submission.weights_format}")


@router.get("/review", response_class=HTMLResponse)
def review_queue(request: Request, user: Optional[User] = Depends(optional_user), svc: HubServices = Depends(get_services)):
    if user is None:
        return _login_redirect(request)
    return render(request, "review_queue.html", user, queue=svc.review_queue(user))


# -------------------------------------------------------------- evaluations


@router.get("/evaluations", response_class=HTMLResponse)
def evaluations(request: Request, user: Optional[User] = Depends(optional_user), svc: HubServices = Depends(get_services)):
    if user is None:
        return _login_redirect(request)
    mine = svc.list_submissions(institution_id=user.institution_id)
    others = [i for i in svc.list_institutions() if i.id != user.institution_id]
    tasks = svc.list_evaluations(institution_id=user.institution_id)
    return render(
        request, "evaluations.html", user,
        tasks=tasks, mine=mine, others=others,
        reciprocity=svc.reciprocity_status(user.institution),
        example_report=json.dumps(dict(EXAMPLE_EVALUATION, evaluated_by=user.institution.slug), indent=2),
    )


@router.post("/evaluations/request")
def evaluations_request(request: Request, bundle_id: str = Form(...), evaluator_slug: str = Form(...), detail: str = Form(""), user: Optional[User] = Depends(optional_user), svc: HubServices = Depends(get_services)):
    if user is None:
        return _login_redirect(request)
    task = svc.request_evaluation(user=user, bundle_id=bundle_id, evaluator_slug=evaluator_slug, detail=detail)
    if task.state == "refused":
        return _redirect("/evaluations", error=f"refused under the reciprocity rule ({task.reason_code}): {task.detail}")
    return _redirect("/evaluations", message=f"evaluation requested from {task.evaluator.slug}")


@router.post("/evaluations/{task_id}/serve")
def evaluations_serve(request: Request, task_id: int, report: str = Form(...), user: Optional[User] = Depends(optional_user), svc: HubServices = Depends(get_services)):
    if user is None:
        return _login_redirect(request)
    svc.serve_evaluation(task=svc.get_evaluation(task_id), user=user, report=report)
    return _redirect("/evaluations", message="evaluation report recorded; awaiting the requester's acknowledgement")


@router.post("/evaluations/{task_id}/acknowledge")
def evaluations_ack(request: Request, task_id: int, user: Optional[User] = Depends(optional_user), svc: HubServices = Depends(get_services)):
    if user is None:
        return _login_redirect(request)
    svc.acknowledge_evaluation(task=svc.get_evaluation(task_id), user=user)
    return _redirect("/evaluations", message="acknowledged; the evaluator's reciprocity credit is now attested")


# ------------------------------------------------------------- institutions


@router.get("/institutions", response_class=HTMLResponse)
def institutions(request: Request, user: Optional[User] = Depends(optional_user), svc: HubServices = Depends(get_services)):
    return render(request, "institutions.html", user, institutions=svc.list_institutions())


@router.get("/institutions/{slug}", response_class=HTMLResponse)
def institution_detail(request: Request, slug: str, user: Optional[User] = Depends(optional_user), svc: HubServices = Depends(get_services)):
    institution = svc.institution_by_slug(slug)
    standing = svc.tp.credit.standing(institution.slug).to_dict() if institution.slug in svc.tp.credit.contributors() else None
    return render(
        request, "institution_detail.html", user,
        i=institution, submissions=svc.list_submissions(institution_id=institution.id),
        reciprocity=svc.reciprocity_status(institution), standing=standing,
    )


# ------------------------------------------------------------------ account


@router.get("/account", response_class=HTMLResponse)
def account(request: Request, user: Optional[User] = Depends(optional_user), svc: HubServices = Depends(get_services)):
    if user is None:
        return _login_redirect(request)
    tokens = list(svc.db.scalars(select(ApiToken).where(ApiToken.user_id == user.id).order_by(ApiToken.created_at.desc())))
    enrolling = None
    if user.totp_secret and not user.totp_enabled:
        from openmed_hub.totp import otpauth_uri, qr_svg

        uri = otpauth_uri(user.totp_secret, user.email, svc.settings.hub_name)
        enrolling = {"secret": user.totp_secret, "otpauth_uri": uri, "qr_svg": qr_svg(uri)}
    needs_2fa = any(r in svc.settings.require_2fa_roles for r in user.role_list) and not user.totp_enabled
    return render(
        request, "account.html", user, tokens=tokens, new_token=request.query_params.get("token"),
        enrolling=enrolling, needs_2fa=needs_2fa, recovery_codes=request.query_params.getlist("rc"),
        members=svc.members(user.institution) if (user.institution_admin or user.is_admin) else [],
    )


@router.post("/account/2fa/enroll")
def account_2fa_enroll(request: Request, user: Optional[User] = Depends(optional_user), svc: HubServices = Depends(get_services)):
    if user is None:
        return _login_redirect(request)
    svc.start_totp_enrollment(user)
    return _redirect("/account", message="scan the code with your authenticator, then confirm with the 6-digit code")


@router.post("/account/2fa/confirm")
def account_2fa_confirm(request: Request, code: str = Form(...), user: Optional[User] = Depends(optional_user), svc: HubServices = Depends(get_services)):
    if user is None:
        return _login_redirect(request)
    codes = svc.confirm_totp(user, code)
    url = "/account?" + "&".join("rc=" + quote(c) for c in codes)
    response = _redirect(url, message="two-factor authentication enabled; save the recovery codes below, they are shown once")
    _set_session(request, response, user)  # the stamp rotated; keep this session
    return response


@router.post("/account/2fa/disable")
def account_2fa_disable(request: Request, password: str = Form(...), code: str = Form(...), user: Optional[User] = Depends(optional_user), svc: HubServices = Depends(get_services)):
    if user is None:
        return _login_redirect(request)
    svc.disable_totp(user, password=password, code=code)
    response = _redirect("/account", message="two-factor authentication disabled")
    _set_session(request, response, user)
    return response


@router.post("/account/2fa/recovery-codes")
def account_2fa_recovery(request: Request, code: str = Form(...), user: Optional[User] = Depends(optional_user), svc: HubServices = Depends(get_services)):
    if user is None:
        return _login_redirect(request)
    codes = svc.regenerate_recovery_codes(user, code=code)
    return _redirect("/account?" + "&".join("rc=" + quote(c) for c in codes), message="new recovery codes; the old ones no longer work")


@router.post("/account/password")
def account_password(request: Request, current_password: str = Form(...), new_password: str = Form(...), user: Optional[User] = Depends(optional_user), svc: HubServices = Depends(get_services)):
    if user is None:
        return _login_redirect(request)
    svc.change_password(user, current_password=current_password, new_password=new_password)
    response = _redirect("/account", message="password changed; every other session has been signed out")
    _set_session(request, response, user)
    return response


@router.post("/account/institution")
def account_institution(request: Request, allowed_email_domains: str = Form(""), rotate: str = Form(""), user: Optional[User] = Depends(optional_user), svc: HubServices = Depends(get_services)):
    if user is None:
        return _login_redirect(request)
    svc.update_institution_settings(actor=user, institution=user.institution, allowed_email_domains=allowed_email_domains, rotate_invite_code=rotate == "yes")
    return _redirect("/account", message="institution settings saved")


@router.post("/account/members/{user_id}")
def account_member(request: Request, user_id: int, decision: str = Form(...), user: Optional[User] = Depends(optional_user), svc: HubServices = Depends(get_services)):
    if user is None:
        return _login_redirect(request)
    member = svc.db.get(User, user_id)
    if member is None:
        raise HubError(404, "member_not_found", "no such member")
    if decision == "admin":
        svc.set_institution_admin(actor=user, member=member, flag=True)
    elif decision == "unadmin":
        svc.set_institution_admin(actor=user, member=member, flag=False)
    else:
        svc.approve_member(actor=user, member=member, approve=decision == "approve")
    return _redirect("/account", message=f"{member.email}: {decision}")


@router.post("/account/tokens")
def account_token(request: Request, label: str = Form(""), user: Optional[User] = Depends(optional_user), svc: HubServices = Depends(get_services)):
    if user is None:
        return _login_redirect(request)
    token = svc.create_token(user, label=label)
    return _redirect(f"/account?token={quote(token)}", message="token created; copy it now, it is not stored")


@router.post("/account/tokens/{token_id}/delete")
def account_token_delete(request: Request, token_id: int, user: Optional[User] = Depends(optional_user), svc: HubServices = Depends(get_services)):
    if user is None:
        return _login_redirect(request)
    svc.revoke_token(user, token_id)
    return _redirect("/account", message="token revoked")


@router.post("/account/reviewer-key")
def account_reviewer_key(request: Request, public_key: str = Form(...), user: Optional[User] = Depends(optional_user), svc: HubServices = Depends(get_services)):
    if user is None:
        return _login_redirect(request)
    svc.set_reviewer_public_key(user, public_key)
    return _redirect("/account", message="reviewer key is now client-held; sign reviews with the CLI")


# ------------------------------------------------------------------ reports


@router.get("/metrics", response_class=HTMLResponse)
def metrics(request: Request, user: Optional[User] = Depends(optional_user), svc: HubServices = Depends(get_services)):
    return render(request, "metrics.html", user, m=svc.metrics(), standing=svc.standing(), ledgers=svc.tp.verify_ledgers())


@router.get("/admin", response_class=HTMLResponse)
def admin(request: Request, user: Optional[User] = Depends(optional_user), svc: HubServices = Depends(get_services)):
    if user is None:
        return _login_redirect(request)
    if not user.is_admin:
        return _redirect("/", error="admin role required")
    users = list(svc.db.scalars(select(User).order_by(User.id)))
    return render(
        request, "admin.html", user, users=users, roles=ALL_ROLES, institutions=svc.list_institutions(),
        identities=list(svc.tp.approved_code_identities()), policy=svc.attestation_policy(),
        measurements=svc.list_measurements(include_revoked=True), audit=svc.audit_log(limit=100),
    )


@router.post("/admin/measurements")
def admin_measurement_add(request: Request, measurement: str = Form(...), label: str = Form(""), user: Optional[User] = Depends(optional_user), svc: HubServices = Depends(get_services)):
    if user is None or not user.is_maintainer:
        return _redirect("/", error="maintainer role required")
    svc.approve_measurement(user=user, measurement=measurement, label=label)
    return _redirect("/admin", message="measurement approved")


@router.post("/admin/measurements/{measurement}/revoke")
def admin_measurement_revoke(request: Request, measurement: str, user: Optional[User] = Depends(optional_user), svc: HubServices = Depends(get_services)):
    if user is None or not user.is_maintainer:
        return _redirect("/", error="maintainer role required")
    svc.revoke_measurement(user=user, measurement=measurement)
    return _redirect("/admin", message="measurement revoked")


@router.post("/admin/users/{user_id}/roles")
async def admin_roles(request: Request, user_id: int, user: Optional[User] = Depends(optional_user), svc: HubServices = Depends(get_services)):
    if user is None or not user.is_admin:
        return _redirect("/", error="admin role required")
    form = await request.form()
    target = svc.db.get(User, user_id)
    if target is None:
        raise HubError(404, "user_not_found", "no such user")
    svc.grant_roles(target, form.getlist("roles"), actor=user)
    return _redirect("/admin", message=f"roles updated for {target.email}")


@router.post("/admin/institutions/{slug}/founding")
def admin_founding(request: Request, slug: str, is_founding: str = Form(""), user: Optional[User] = Depends(optional_user), svc: HubServices = Depends(get_services)):
    if user is None or not user.is_admin:
        return _redirect("/", error="admin role required")
    svc.set_founding(svc.institution_by_slug(slug), is_founding == "yes", actor=user)
    return _redirect("/admin", message=f"{slug} updated")


__all__ = ["router", "EXAMPLE_CARD", "EXAMPLE_EVALUATION", "EXAMPLE_MANUAL"]
