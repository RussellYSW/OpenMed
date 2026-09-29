"""JSON API under ``/api/v1`` -- what the CLI and other nodes talk to."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from openmed_hub.db import User
from openmed_hub.deps import admin_user, current_user, get_services
from openmed_hub.services import HubError, HubServices

router = APIRouter(prefix="/api/v1", tags=["api"])


# ------------------------------------------------------------------ schemas


class InstitutionIn(BaseModel):
    name: str
    slug: Optional[str] = None
    kind: str = "health_system"
    country: str = "US"


class AdminUserIn(BaseModel):
    email: str
    name: str = ""
    password: str


class RegisterInstitutionIn(BaseModel):
    institution: InstitutionIn
    user: AdminUserIn


class JoinIn(BaseModel):
    institution_slug: str
    email: str
    name: str = ""
    password: str


class LoginIn(BaseModel):
    email: str
    password: str


class TokenIn(BaseModel):
    label: str = ""


class NodeKeyIn(BaseModel):
    public_key: str = Field(..., description="32-byte Ed25519 public key, hex")


class NodeVerifyIn(BaseModel):
    nonce: str
    signature: str = Field(..., description="Ed25519 signature over the nonce bytes, hex")


class ReviewIn(BaseModel):
    decision: str
    statement: str = ""


class SignatureIn(BaseModel):
    signature: Dict[str, Any]


class ReasonIn(BaseModel):
    reason: str = ""
    note: str = ""
    grounds: str = ""
    upheld: bool = False


class EvaluationRequestIn(BaseModel):
    bundle_id: str
    evaluator_slug: str
    detail: str = ""


class EvaluationServeIn(BaseModel):
    report: Dict[str, Any]


class RolesIn(BaseModel):
    roles: List[str]


class FoundingIn(BaseModel):
    is_founding: bool


class ReviewerKeyIn(BaseModel):
    public_key: str


class MeasurementIn(BaseModel):
    measurement: str
    label: str = ""
    manifest: Optional[Dict[str, Any]] = None


# ------------------------------------------------------------------ helpers


def _user_dict(user: User) -> Dict[str, Any]:
    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "institution": user.institution.slug,
        "roles": user.role_list,
        "reviewer_id": user.reviewer_id,
        "reviewer_key": (
            "client-held" if user.reviewer_public_key else ("hub-held" if user.reviewer_key_seed else None)
        ),
    }


def _institution_dict(svc: HubServices, institution) -> Dict[str, Any]:
    return {
        "slug": institution.slug,
        "name": institution.name,
        "kind": institution.kind,
        "country": institution.country,
        "is_founding": institution.is_founding,
        "node_registered": bool(institution.node_public_key),
        "node_verified": institution.node_verified,
        "node_verified_at": institution.node_verified_at.isoformat() if institution.node_verified_at else None,
        "reciprocity": svc.reciprocity_status(institution),
        "members": len(institution.users),
    }


# ----------------------------------------------------------------- accounts


@router.post("/institutions", status_code=201)
def register_institution(body: RegisterInstitutionIn, svc: HubServices = Depends(get_services)):
    institution, user = svc.register_institution(
        name=body.institution.name,
        slug=body.institution.slug,
        kind=body.institution.kind,
        country=body.institution.country,
        admin_email=body.user.email,
        admin_name=body.user.name,
        admin_password=body.user.password,
    )
    token = svc.create_token(user, label="registration")
    return {"institution": _institution_dict(svc, institution), "user": _user_dict(user), "token": token}


@router.get("/institutions")
def list_institutions(svc: HubServices = Depends(get_services)):
    return [_institution_dict(svc, i) for i in svc.list_institutions()]


@router.get("/institutions/{slug}")
def get_institution(slug: str, svc: HubServices = Depends(get_services)):
    return _institution_dict(svc, svc.institution_by_slug(slug))


@router.post("/auth/join", status_code=201)
def join(body: JoinIn, svc: HubServices = Depends(get_services)):
    institution = svc.institution_by_slug(body.institution_slug)
    user = svc.register_user(email=body.email, name=body.name, password=body.password, institution=institution)
    return {"user": _user_dict(user), "token": svc.create_token(user, label="registration")}


@router.post("/auth/login")
def login(body: LoginIn, svc: HubServices = Depends(get_services)):
    user = svc.authenticate(body.email, body.password)
    return {"user": _user_dict(user), "token": svc.create_token(user, label="login")}


@router.get("/auth/me")
def me(user: User = Depends(current_user)):
    return _user_dict(user)


@router.post("/auth/tokens", status_code=201)
def create_token(body: TokenIn, user: User = Depends(current_user), svc: HubServices = Depends(get_services)):
    return {"token": svc.create_token(user, label=body.label)}


@router.post("/auth/reviewer-key")
def set_reviewer_key(body: ReviewerKeyIn, user: User = Depends(current_user), svc: HubServices = Depends(get_services)):
    return _user_dict(svc.set_reviewer_public_key(user, body.public_key))


# --------------------------------------------------------------------- nodes


@router.post("/institutions/{slug}/node/key")
def node_key(slug: str, body: NodeKeyIn, user: User = Depends(current_user), svc: HubServices = Depends(get_services)):
    institution = svc.institution_by_slug(slug)
    if user.institution_id != institution.id and not user.is_admin:
        raise HubError(403, "not_member", "only members register their institution's node")
    nonce = svc.begin_node_registration(institution, body.public_key)
    return {"nonce": nonce, "expires_in_seconds": 900}


@router.post("/institutions/{slug}/node/verify")
def node_verify(slug: str, body: NodeVerifyIn, user: User = Depends(current_user), svc: HubServices = Depends(get_services)):
    institution = svc.institution_by_slug(slug)
    if user.institution_id != institution.id and not user.is_admin:
        raise HubError(403, "not_member", "only members verify their institution's node")
    svc.complete_node_registration(institution, body.nonce, body.signature)
    return _institution_dict(svc, institution)


# --------------------------------------------------------------- attestation


@router.get("/attestation/policy")
def attestation_policy(svc: HubServices = Depends(get_services)):
    return svc.attestation_policy()


@router.post("/attestation/challenge")
def attestation_challenge(user: User = Depends(current_user), svc: HubServices = Depends(get_services)):
    """Issue the nonce a site signs into its next quote (node-key mode)."""
    return svc.issue_challenge(user)


@router.post("/attestation/measurements", status_code=201)
def approve_measurement(body: MeasurementIn, user: User = Depends(current_user), svc: HubServices = Depends(get_services)):
    row = svc.approve_measurement(user=user, measurement=body.measurement, label=body.label, manifest=body.manifest)
    return {"measurement": row.measurement, "label": row.label, "added_at": row.added_at.isoformat()}


@router.delete("/attestation/measurements/{measurement}")
def revoke_measurement(measurement: str, user: User = Depends(current_user), svc: HubServices = Depends(get_services)):
    row = svc.revoke_measurement(user=user, measurement=measurement)
    return {"measurement": row.measurement, "revoked_at": row.revoked_at.isoformat()}


# --------------------------------------------------------------- submissions


@router.post("/submissions", status_code=201)
async def create_submission(
    name: str = Form(...),
    version: str = Form(...),
    model_card: str = Form(...),
    evaluation: str = Form(...),
    code_identity: str = Form(""),
    config: str = Form(""),
    quote: str = Form(""),
    fine_tuning_manual: str = Form(""),
    parent_bundle_id: str = Form(""),
    tags: str = Form(""),
    evidence: str = Form(""),
    weights: UploadFile = File(...),
    user: User = Depends(current_user),
    svc: HubServices = Depends(get_services),
):
    blob = await weights.read()
    fmt = (weights.filename or "").rsplit(".", 1)[-1] if weights.filename and "." in weights.filename else "other"
    submission = svc.submit(
        user=user,
        name=name,
        version=version,
        weights=blob,
        weights_format=fmt,
        model_card=model_card,
        evaluation=evaluation,
        manual=fine_tuning_manual or None,
        parent_bundle_id=parent_bundle_id or None,
        code_identity=code_identity,
        config=config,
        quote=quote or None,
        tags=[t for t in tags.split(",") if t.strip()],
        evidence=evidence or None,
    )
    return svc.submission_summary(submission)


@router.get("/submissions")
def list_submissions(state: Optional[str] = None, institution: Optional[str] = None, svc: HubServices = Depends(get_services)):
    inst_id = svc.institution_by_slug(institution).id if institution else None
    return [svc.submission_summary(s) for s in svc.list_submissions(state=state, institution_id=inst_id)]


@router.get("/submissions/{bundle_id:path}/lineage")
def lineage(bundle_id: str, svc: HubServices = Depends(get_services)):
    detail = svc.submission_detail(svc.get_submission(bundle_id))
    return {"lineage": detail["lineage"], "summary": detail["lineage_summary"], "mermaid": detail["lineage_mermaid"], "children": detail["children"]}


@router.get("/submissions/{bundle_id:path}/decision-log")
def decision_log(bundle_id: str, svc: HubServices = Depends(get_services)):
    return svc.submission_detail(svc.get_submission(bundle_id))["decision_log"]


@router.get("/submissions/{bundle_id:path}/weights")
def download_weights(bundle_id: str, user: User = Depends(current_user), svc: HubServices = Depends(get_services)):
    submission = svc.get_submission(bundle_id)
    path = svc.download(submission=submission, user=user)
    filename = f"{submission.name}-{submission.version}.{submission.weights_format}"
    return FileResponse(path, media_type="application/octet-stream", filename=filename)


@router.post("/submissions/{bundle_id:path}/reviews", status_code=201)
def review(bundle_id: str, body: ReviewIn, user: User = Depends(current_user), svc: HubServices = Depends(get_services)):
    submission = svc.get_submission(bundle_id)
    row = svc.review(submission=submission, user=user, decision=body.decision, statement=body.statement)
    return {"review": svc.review_summary(row), "state": submission.state}


@router.post("/submissions/{bundle_id:path}/signatures", status_code=201)
def submit_signature(bundle_id: str, body: SignatureIn, user: User = Depends(current_user), svc: HubServices = Depends(get_services)):
    submission = svc.get_submission(bundle_id)
    row = svc.submit_signature(submission=submission, signature_data=body.signature)
    return {"review": svc.review_summary(row), "state": submission.state}


@router.post("/submissions/{bundle_id:path}/dispute")
def dispute(bundle_id: str, body: ReasonIn, user: User = Depends(current_user), svc: HubServices = Depends(get_services)):
    s = svc.dispute(submission=svc.get_submission(bundle_id), user=user, reason=body.reason)
    return svc.submission_summary(s)


@router.post("/submissions/{bundle_id:path}/remediate")
def remediate(bundle_id: str, body: ReasonIn, user: User = Depends(current_user), svc: HubServices = Depends(get_services)):
    s = svc.remediate(submission=svc.get_submission(bundle_id), user=user, note=body.note or body.reason)
    return svc.submission_summary(s)


@router.post("/submissions/{bundle_id:path}/appeal")
def appeal(bundle_id: str, body: ReasonIn, user: User = Depends(current_user), svc: HubServices = Depends(get_services)):
    s = svc.appeal(submission=svc.get_submission(bundle_id), user=user, grounds=body.grounds or body.reason)
    return svc.submission_summary(s)


@router.post("/submissions/{bundle_id:path}/appeal/resolve")
def resolve_appeal(bundle_id: str, body: ReasonIn, user: User = Depends(current_user), svc: HubServices = Depends(get_services)):
    s = svc.resolve_appeal(submission=svc.get_submission(bundle_id), user=user, upheld=body.upheld, note=body.note)
    return svc.submission_summary(s)


@router.post("/submissions/{bundle_id:path}/revoke")
def revoke(bundle_id: str, body: ReasonIn, user: User = Depends(current_user), svc: HubServices = Depends(get_services)):
    s = svc.revoke(submission=svc.get_submission(bundle_id), user=user, reason=body.reason)
    return svc.submission_summary(s)


@router.get("/submissions/{bundle_id:path}")
def get_submission(bundle_id: str, svc: HubServices = Depends(get_services)):
    return svc.submission_detail(svc.get_submission(bundle_id))


# --------------------------------------------------------------- evaluations


@router.post("/evaluations", status_code=201)
def request_evaluation(body: EvaluationRequestIn, user: User = Depends(current_user), svc: HubServices = Depends(get_services)):
    task = svc.request_evaluation(user=user, bundle_id=body.bundle_id, evaluator_slug=body.evaluator_slug, detail=body.detail)
    return svc.evaluation_summary(task)


@router.get("/evaluations")
def list_evaluations(mine: bool = False, user: User = Depends(current_user), svc: HubServices = Depends(get_services)):
    tasks = svc.list_evaluations(institution_id=user.institution_id if mine else None)
    return [svc.evaluation_summary(t) for t in tasks]


@router.post("/evaluations/{task_id}/serve")
def serve_evaluation(task_id: int, body: EvaluationServeIn, user: User = Depends(current_user), svc: HubServices = Depends(get_services)):
    return svc.evaluation_summary(svc.serve_evaluation(task=svc.get_evaluation(task_id), user=user, report=body.report))


@router.post("/evaluations/{task_id}/acknowledge")
def acknowledge_evaluation(task_id: int, user: User = Depends(current_user), svc: HubServices = Depends(get_services)):
    return svc.evaluation_summary(svc.acknowledge_evaluation(task=svc.get_evaluation(task_id), user=user))


# ------------------------------------------------------------------ reports


@router.get("/metrics")
def metrics(svc: HubServices = Depends(get_services)):
    return svc.metrics()


@router.get("/standing")
def standing(svc: HubServices = Depends(get_services)):
    return svc.standing()


@router.get("/ledgers/verify")
def verify_ledgers(svc: HubServices = Depends(get_services)):
    return svc.tp.verify_ledgers()


# -------------------------------------------------------------------- admin


@router.post("/admin/users/{user_id}/roles")
def set_roles(user_id: int, body: RolesIn, _: User = Depends(admin_user), svc: HubServices = Depends(get_services)):
    target = svc.db.get(User, user_id)
    if target is None:
        raise HubError(404, "user_not_found", f"no user {user_id}")
    return _user_dict(svc.grant_roles(target, body.roles))


@router.post("/admin/institutions/{slug}/founding")
def set_founding(slug: str, body: FoundingIn, _: User = Depends(admin_user), svc: HubServices = Depends(get_services)):
    return _institution_dict(svc, svc.set_founding(svc.institution_by_slug(slug), body.is_founding))


@router.get("/admin/users")
def list_users(_: User = Depends(admin_user), svc: HubServices = Depends(get_services)):
    from sqlalchemy import select

    return [_user_dict(u) for u in svc.db.scalars(select(User).order_by(User.id))]


def hub_error_response(exc: HubError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.code, "detail": exc.detail})


__all__ = ["router", "hub_error_response"]
