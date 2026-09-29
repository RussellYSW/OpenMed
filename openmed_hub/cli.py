"""``openmed`` -- run the hub, or talk to one from a site.

Server side::

    openmed init   --data-dir ./openmed-data --institution "University of Miami" --email admin@… --password …
    openmed serve  --data-dir ./openmed-data --port 8000
    openmed seed-demo --data-dir ./openmed-data      # synthetic institutions, users, a certified model

Site side (state in ``~/.openmed/``)::

    openmed login --hub http://hub:8000 --email … --password …   (or --token …)
    openmed node keygen && openmed node register
    openmed measure --script train.py                 # digest a maintainer approves
    openmed submit --name m --version 1.0.0 --weights w.npz --card card.json --evaluation eval.json --script train.py
    openmed models | openmed model <bundle-id> | openmed download <bundle-id> -o w.npz
    openmed reviewer keygen && openmed reviewer register
    openmed review --bundle <id> --decision approve --statement "…"
    openmed evaluate request|serve|ack|list …
    openmed metrics | openmed verify-ledgers
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

CONFIG_DIR = Path(os.environ.get("OPENMED_HOME", Path.home() / ".openmed"))
CONFIG_FILE = CONFIG_DIR / "config.json"
NODE_KEY_FILE = CONFIG_DIR / "node.key"
REVIEWER_KEY_FILE = CONFIG_DIR / "reviewer.key"


# ------------------------------------------------------------------ helpers


def _load_config() -> Dict[str, Any]:
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    return {}


def _save_config(cfg: Dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    try:
        os.chmod(CONFIG_FILE, 0o600)
    except OSError:  # pragma: no cover
        pass


def _client(args: argparse.Namespace):
    import httpx

    cfg = _load_config()
    hub = getattr(args, "hub", None) or cfg.get("hub") or "http://127.0.0.1:8000"
    token = getattr(args, "token", None) or cfg.get("token")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return httpx.Client(base_url=hub.rstrip("/") + "/api/v1", headers=headers, timeout=120.0)


def _print(data: Any) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def _check(response) -> Any:
    try:
        body = response.json()
    except ValueError:
        body = response.text
    if response.status_code >= 400:
        print(f"error {response.status_code}: {body}", file=sys.stderr)
        sys.exit(1)
    return body


def _write_key(path: Path, private_hex: str) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(private_hex, encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:  # pragma: no cover
        pass


def _ed25519_private(path: Path):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    if not path.exists():
        print(f"no key at {path}; run the matching keygen command first", file=sys.stderr)
        sys.exit(1)
    return Ed25519PrivateKey.from_private_bytes(bytes.fromhex(path.read_text().strip()))


def _public_hex(private) -> str:
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    return private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()


def _keygen(path: Path) -> str:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat

    if path.exists():
        print(f"{path} already exists; delete it to rotate", file=sys.stderr)
        sys.exit(1)
    private = Ed25519PrivateKey.generate()
    raw = private.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    _write_key(path, raw.hex())
    return _public_hex(private)


# ------------------------------------------------------------------- server


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    if args.data_dir:
        os.environ["OPENMED_DATA_DIR"] = str(args.data_dir)
    from openmed_hub.app import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")


def _services(data_dir: Optional[str]):
    from openmed_hub.app import create_app
    from openmed_hub.services import HubServices

    app = create_app(_settings(data_dir))
    db = app.state.session_factory()
    return app, HubServices(app.state.trustplane, app.state.settings, db)


def _settings(data_dir: Optional[str]):
    from openmed_hub.config import HubSettings

    return HubSettings.from_env(data_dir)


def cmd_init(args: argparse.Namespace) -> None:
    from openmed_hub.db import ROLE_ADMIN, ROLE_MAINTAINER

    _, svc = _services(args.data_dir)
    password = args.password or getpass.getpass("admin password: ")
    institution, user = svc.register_institution(
        name=args.institution,
        slug=args.slug,
        kind=args.kind,
        country=args.country,
        admin_email=args.email,
        admin_name=args.name or "Hub administrator",
        admin_password=password,
        is_founding=True,
    )
    svc.grant_roles(user, [ROLE_ADMIN, ROLE_MAINTAINER])
    print(f"initialised {svc.settings.data_dir}")
    print(f"founding institution: {institution.name} ({institution.slug})")
    print(f"admin user: {user.email}  roles: {user.roles}")


def cmd_seed_demo(args: argparse.Namespace) -> None:
    from openmed_hub.seed import seed_demo

    _, svc = _services(args.data_dir)
    summary = seed_demo(svc)
    _print(summary)


# --------------------------------------------------------------------- site


def cmd_login(args: argparse.Namespace) -> None:
    cfg = _load_config()
    cfg["hub"] = args.hub or cfg.get("hub") or "http://127.0.0.1:8000"
    if args.token:
        cfg["token"] = args.token
    else:
        password = args.password or getpass.getpass("password: ")
        with _client(argparse.Namespace(hub=cfg["hub"], token=None)) as client:
            body = _check(client.post("/auth/login", json={"email": args.email, "password": password}))
        cfg["token"] = body["token"]
    _save_config(cfg)
    with _client(argparse.Namespace(hub=cfg["hub"], token=cfg["token"])) as client:
        me = _check(client.get("/auth/me"))
    cfg["institution"] = me["institution"]
    cfg["reviewer_id"] = me.get("reviewer_id")
    _save_config(cfg)
    print(f"signed in to {cfg['hub']} as {me['email']} ({me['institution']}); roles {me['roles']}")


def cmd_register(args: argparse.Namespace) -> None:
    password = args.password or getpass.getpass("password: ")
    with _client(argparse.Namespace(hub=args.hub, token=None)) as client:
        body = _check(
            client.post(
                "/institutions",
                json={
                    "institution": {"name": args.institution, "slug": args.slug, "kind": args.kind, "country": args.country},
                    "user": {"email": args.email, "name": args.name or "", "password": password},
                },
            )
        )
    cfg = _load_config()
    cfg.update(hub=args.hub, token=body["token"], institution=body["institution"]["slug"])
    _save_config(cfg)
    print(f"registered {body['institution']['slug']}; signed in as {body['user']['email']}")


def cmd_whoami(args: argparse.Namespace) -> None:
    with _client(args) as client:
        _print(_check(client.get("/auth/me")))


def cmd_node(args: argparse.Namespace) -> None:
    if args.node_cmd == "keygen":
        public = _keygen(NODE_KEY_FILE)
        print(f"node key written to {NODE_KEY_FILE}\npublic key: {public}")
        return
    private = _ed25519_private(NODE_KEY_FILE)
    cfg = _load_config()
    slug = args.institution or cfg.get("institution")
    if not slug:
        print("login first (or pass --institution)", file=sys.stderr)
        sys.exit(1)
    with _client(args) as client:
        challenge = _check(client.post(f"/institutions/{slug}/node/key", json={"public_key": _public_hex(private)}))
        signature = private.sign(challenge["nonce"].encode("utf-8")).hex()
        result = _check(client.post(f"/institutions/{slug}/node/verify", json={"nonce": challenge["nonce"], "signature": signature}))
    print(f"node for {slug}: verified={result['node_verified']} at {result['node_verified_at']}")


def cmd_reviewer(args: argparse.Namespace) -> None:
    if args.reviewer_cmd == "keygen":
        public = _keygen(REVIEWER_KEY_FILE)
        print(f"reviewer key written to {REVIEWER_KEY_FILE}\npublic key: {public}")
        return
    private = _ed25519_private(REVIEWER_KEY_FILE)
    with _client(args) as client:
        me = _check(client.post("/auth/reviewer-key", json={"public_key": _public_hex(private)}))
    print(f"reviewer {me['reviewer_id']} now signs with a client-held key")


def _read_json_arg(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    path = Path(value)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return value


def _measurement(args: argparse.Namespace):
    from trustfed.attestation import software_measurement

    return software_measurement(
        script=Path(args.script) if getattr(args, "script", None) else None,
        extra_files=[Path(p) for p in (getattr(args, "extra_file", None) or [])],
    )


def _config_hash(args: argparse.Namespace) -> str:
    import hashlib

    text = ""
    if getattr(args, "config_file", None):
        text = Path(args.config_file).read_text(encoding="utf-8")
    elif getattr(args, "config", None):
        text = args.config
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cmd_measure(args: argparse.Namespace) -> None:
    """Print the software measurement a maintainer would approve."""
    m = _measurement(args)
    if args.json:
        _print(m.to_dict())
    else:
        print(m.measurement)
        for pkg in m.manifest["packages"]:
            print(f"  package {pkg['package']} {pkg.get('version','')}: {pkg['files']} files, {pkg['sha256'][:16]}…", file=sys.stderr)
        if m.manifest["script"]:
            print(f"  script {m.manifest['script']['name']}: {m.manifest['script']['sha256'][:16]}…", file=sys.stderr)


def _node_quote(client, args: argparse.Namespace, slug: str) -> Dict[str, Any]:
    """Measure, request a challenge, and sign the quote with the node key."""
    from trustfed.attestation import NodeKeyAttestor
    from trustfed.ledger.crypto import Ed25519Signer

    private = _ed25519_private(NODE_KEY_FILE)
    measurement = _measurement(args)
    challenge = _check(client.post("/attestation/challenge"))
    quote = NodeKeyAttestor.sign_quote(
        Ed25519Signer(private),
        client_id=challenge["client_id"] or slug,
        measurement=measurement.measurement,
        config_hash=_config_hash(args),
        nonce=challenge["nonce"],
    )
    return quote.to_dict()


def cmd_attest(args: argparse.Namespace) -> None:
    """Produce (and print) a signed quote without submitting anything."""
    cfg = _load_config()
    with _client(args) as client:
        _print(_node_quote(client, args, cfg.get("institution", "")))


def cmd_submit(args: argparse.Namespace) -> None:
    weights = Path(args.weights)
    cfg = _load_config()
    data = {
        "name": args.name,
        "version": args.version,
        "code_identity": args.code_identity,
        "config": args.config or "",
        "model_card": _read_json_arg(args.card) or "",
        "evaluation": _read_json_arg(args.evaluation) or "",
        "fine_tuning_manual": _read_json_arg(args.manual) or "",
        "parent_bundle_id": args.parent or "",
        "tags": args.tags or "",
        "evidence": _read_json_arg(args.evidence) or "",
    }
    with _client(args) as client, weights.open("rb") as fh:
        policy = _check(client.get("/attestation/policy"))
        if policy.get("mode") == "nodekey":
            data["quote"] = json.dumps(_node_quote(client, args, cfg.get("institution", "")))
            print(f"attested with node key: measurement {json.loads(data['quote'])['measurement'][:16]}…", file=sys.stderr)
        body = _check(client.post("/submissions", data=data, files={"weights": (weights.name, fh)}))
    _print(body)


def cmd_models(args: argparse.Namespace) -> None:
    with _client(args) as client:
        rows = _check(client.get("/submissions", params={k: v for k, v in {"state": args.state}.items() if v}))
    for row in rows:
        print(f"{row['short_id']}  {row['state']:<13} gate={row['gate_status']:<5} {row['institution']:<20} {row['name']} v{row['version']}")
    if not rows:
        print("no submissions")


def cmd_model(args: argparse.Namespace) -> None:
    with _client(args) as client:
        _print(_check(client.get(f"/submissions/{args.bundle}")))


def cmd_download(args: argparse.Namespace) -> None:
    with _client(args) as client:
        response = client.get(f"/submissions/{args.bundle}/weights")
        if response.status_code >= 400:
            _check(response)
        out = Path(args.out) if args.out else Path(response.headers.get("content-disposition", "weights.bin").split("filename=")[-1].strip('"'))
        out.write_bytes(response.content)
    print(f"wrote {out} ({len(response.content)} bytes)")


def cmd_review(args: argparse.Namespace) -> None:
    from trustfed.certification.keys import ReviewSignature
    from trustfed.ledger.block import utc_now_iso
    from trustfed.ledger.crypto import Ed25519Signer

    cfg = _load_config()
    with _client(args) as client:
        me = _check(client.get("/auth/me"))
        if not me.get("reviewer_id"):
            print("you are not a registered reviewer", file=sys.stderr)
            sys.exit(1)
        detail = _check(client.get(f"/submissions/{args.bundle}"))
        bundle_id = detail["submission"]["bundle_id"]
        if REVIEWER_KEY_FILE.exists():
            private = _ed25519_private(REVIEWER_KEY_FILE)
            signer = Ed25519Signer(private)
            unsigned = ReviewSignature(
                bundle_id=bundle_id,
                reviewer_id=me["reviewer_id"],
                institution=me["institution"],
                decision=args.decision,
                statement=args.statement,
                signed_at=utc_now_iso(),
                signature="",
                key_id="",
            )
            signed = ReviewSignature(
                **dict(unsigned.to_dict(), signature=signer.sign(unsigned.signing_material()), key_id=signer.key_id)
            )
            body = _check(client.post(f"/submissions/{bundle_id}/signatures", json={"signature": signed.to_dict()}))
            mode = "client-held key"
        else:
            body = _check(client.post(f"/submissions/{bundle_id}/reviews", json={"decision": args.decision, "statement": args.statement}))
            mode = "hub-held key"
        review = body["review"]
        status = "accepted" if review["accepted"] else f"refused ({review['refusal_code']}: {review['refusal_detail']})"
        print(f"review {status} with {mode}; case is {body['state']}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    with _client(args) as client:
        if args.eval_cmd == "request":
            _print(_check(client.post("/evaluations", json={"bundle_id": args.bundle, "evaluator_slug": args.evaluator, "detail": args.detail or ""})))
        elif args.eval_cmd == "serve":
            report = json.loads(_read_json_arg(args.report) or "{}")
            _print(_check(client.post(f"/evaluations/{args.task}/serve", json={"report": report})))
        elif args.eval_cmd == "ack":
            _print(_check(client.post(f"/evaluations/{args.task}/acknowledge")))
        else:
            _print(_check(client.get("/evaluations", params={"mine": "true"})))


def cmd_metrics(args: argparse.Namespace) -> None:
    with _client(args) as client:
        _print(_check(client.get("/metrics")))


def cmd_verify_ledgers(args: argparse.Namespace) -> None:
    with _client(args) as client:
        _print(_check(client.get("/ledgers/verify")))


# ------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="openmed", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def hub_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--hub", help="hub base URL (default: from ~/.openmed/config.json)")
        p.add_argument("--token", help="API token (default: from config)")

    p = sub.add_parser("serve", help="run the hub")
    p.add_argument("--data-dir")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("init", help="create the founding institution and admin user")
    p.add_argument("--data-dir")
    p.add_argument("--institution", required=True)
    p.add_argument("--slug")
    p.add_argument("--kind", default="medical_school")
    p.add_argument("--country", default="US")
    p.add_argument("--email", required=True)
    p.add_argument("--name")
    p.add_argument("--password")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("seed-demo", help="populate a hub with synthetic demo data")
    p.add_argument("--data-dir")
    p.set_defaults(func=cmd_seed_demo)

    p = sub.add_parser("login", help="sign in to a hub")
    hub_args(p)
    p.add_argument("--email")
    p.add_argument("--password")
    p.set_defaults(func=cmd_login)

    p = sub.add_parser("register", help="register a new institution and its first user")
    p.add_argument("--hub", required=True)
    p.add_argument("--institution", required=True)
    p.add_argument("--slug")
    p.add_argument("--kind", default="health_system")
    p.add_argument("--country", default="US")
    p.add_argument("--email", required=True)
    p.add_argument("--name")
    p.add_argument("--password")
    p.set_defaults(func=cmd_register)

    p = sub.add_parser("whoami")
    hub_args(p)
    p.set_defaults(func=cmd_whoami)

    p = sub.add_parser("node", help="node key and verification handshake")
    hub_args(p)
    p.add_argument("node_cmd", choices=["keygen", "register"])
    p.add_argument("--institution")
    p.set_defaults(func=cmd_node)

    p = sub.add_parser("reviewer", help="client-held reviewer key")
    hub_args(p)
    p.add_argument("reviewer_cmd", choices=["keygen", "register"])
    p.set_defaults(func=cmd_reviewer)

    def measure_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--script", help="training script to include in the measurement")
        p.add_argument("--extra-file", action="append", help="other files the pipeline loads (repeatable)")

    p = sub.add_parser("measure", help="print the software measurement of the installed pipeline")
    measure_args(p)
    p.add_argument("--json", action="store_true", help="print the manifest too")
    p.set_defaults(func=cmd_measure)

    p = sub.add_parser("attest", help="request a challenge and print a node-key-signed quote")
    hub_args(p)
    measure_args(p)
    p.add_argument("--config", default="")
    p.add_argument("--config-file")
    p.set_defaults(func=cmd_attest)

    p = sub.add_parser("submit", help="upload a model bundle (attests with the node key when the hub requires it)")
    hub_args(p)
    measure_args(p)
    p.add_argument("--config-file", help="training configuration file; its hash goes into the quote")
    p.add_argument("--name", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--card", required=True, help="model card JSON (file or inline)")
    p.add_argument("--evaluation", required=True, help="evaluation report JSON (file or inline)")
    p.add_argument("--manual", help="fine-tuning manual JSON (file or inline)")
    p.add_argument("--evidence", help="extra evidence JSON: member_scores / nonmember_scores")
    p.add_argument("--parent", help="parent bundle id for a derived model")
    p.add_argument("--code-identity", default="openmed-training-pipeline@v1", help="mock-mode hubs only")
    p.add_argument("--config", default="", help="training configuration label; hashed into the quote")
    p.add_argument("--tags", default="")
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("models", help="list submissions")
    hub_args(p)
    p.add_argument("--state")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("model", help="show one submission")
    hub_args(p)
    p.add_argument("bundle")
    p.set_defaults(func=cmd_model)

    p = sub.add_parser("download", help="download weights (records a registered download)")
    hub_args(p)
    p.add_argument("bundle")
    p.add_argument("-o", "--out")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("review", help="sign a review (client-held key if present, else hub-held)")
    hub_args(p)
    p.add_argument("--bundle", required=True)
    p.add_argument("--decision", required=True, choices=["approve", "reject", "block"])
    p.add_argument("--statement", default="")
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("evaluate", help="multi-site evaluation tasks")
    hub_args(p)
    p.add_argument("eval_cmd", choices=["request", "serve", "ack", "list"])
    p.add_argument("--bundle")
    p.add_argument("--evaluator")
    p.add_argument("--detail")
    p.add_argument("--task", type=int)
    p.add_argument("--report", help="evaluation report JSON (file or inline)")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("metrics")
    hub_args(p)
    p.set_defaults(func=cmd_metrics)

    p = sub.add_parser("verify-ledgers")
    hub_args(p)
    p.set_defaults(func=cmd_verify_ledgers)
    return parser


def main(argv: Optional[list] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
