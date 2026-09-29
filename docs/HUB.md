# OpenMed Hub

The hub is the running service around the `trustfed` library. `trustfed` gives you the
six components as Python objects; the hub is what makes them a *commons* that
institutions you do not know can join:

| Need (from the proposal) | Where it lives |
|---|---|
| Institution and user registration, roles (contributor → trusted reviewer → maintainer), API tokens | `openmed_hub/db.py`, `services.py` (accounts) |
| Node key registration and the verification handshake (the "independent installation" metric) | `POST /api/v1/institutions/{slug}/node/key` + `/verify`; `openmed node` |
| Model upload as a bundle: weights, model card, evaluation report, fine-tuning manual, parent link | `POST /api/v1/submissions`; `/submit` page; `openmed submit` |
| Attested submission: site-signed quote over a software measurement of the code that ran, against maintainer-approved measurements | `TrustPlane.node_attestor` (`NodeKeyAttestor`), `openmed measure` / `openmed submit`, `POST /api/v1/attestation/challenge`, `/attestation/measurements` |
| Automated gate (seven checks); a failing check blocks the case with the reason logged | `HubServices.submit` → `trustfed.quality` → `authority.dispute(...)` |
| Model-review board: reviewers from other institutions sign; no self-certification; k-of-n across institutions plus technical/clinical composition | `HubServices.review`, `submit_signature`, `board_composition`, `_maybe_certify` |
| Disputes decided by rule: dispute, remediate, appeal, resolve, revoke — all on the decision log | `HubServices.dispute/remediate/appeal/resolve_appeal/revoke` |
| Certified base with a citable identifier naming every contributor and reviewer | `_on_certified` → `ReleaseAttribution` on the credit ledger |
| Download after lightweight registration (the reciprocity lever) | `GET /api/v1/submissions/{id}/weights`; `/models/{id}/download` |
| Lineage graph and verification back to the root | `GET /api/v1/submissions/{id}/lineage` |
| Multi-site evaluation under the service-reciprocity rule, with counterparty-signed acknowledgements | `HubServices.request_evaluation/serve_evaluation/acknowledge_evaluation` |
| Quarterly metrics with denominators (installations, first-contribution conversion, 90-day retention, …) | `GET /api/v1/metrics`; `/metrics` |
| Tamper-evident record of everything above | three `FileLedger` chains under `<data-dir>/ledgers/` |

## Run it

```bash
pip install -e ".[hub,dev]"
openmed init --data-dir ./openmed-data --institution "University of Miami" --email admin@example.org
openmed serve --data-dir ./openmed-data --port 8000
```

Open http://127.0.0.1:8000. The first account on a fresh hub is its admin and
maintainer; grant reviewer roles on `/admin`. The JSON API is documented at `/docs`.

To see every workflow populated with synthetic data:

```bash
openmed seed-demo --data-dir ./demo-data && openmed serve --data-dir ./demo-data
```

The seeder prints the demo accounts (one password for all). It creates four founding
sites and one external lab, a certified root model, a derived model under review, a
model the gate blocked for a large subgroup gap, and one completed evaluation exchange.

## Use it from a site

```bash
openmed register --hub http://hub:8000 --institution "Somewhere Medical Center" --email you@somewhere.org
openmed node keygen && openmed node register        # verification handshake
openmed submit --name decline-risk --version 1.0.0 --weights model.npz \
               --card card.json --evaluation eval.json --manual manual.json
openmed models
openmed download <bundle-id> -o base.npz            # then fine-tune locally
openmed submit --name decline-risk --version 1.1.0-yours --parent <bundle-id> ...
```

Approving a pipeline release (maintainer): run `openmed measure --script train.py` on a
clean install of the release, then paste the digest on `/admin` (or
`POST /api/v1/attestation/measurements`). The digest excludes machine-specific facts, so
every site running that release measures identically; a site that patches the code
measures differently and is refused until the new digest is reviewed and approved.

Reviewers:

```bash
openmed reviewer keygen && openmed reviewer register  # optional: sign with your own key
openmed review --bundle <id> --decision approve --statement "what you checked"
```

Evaluations (reciprocity):

```bash
openmed evaluate request --bundle <id> --evaluator other-site
openmed evaluate serve --task 3 --report eval.json     # as the evaluator
openmed evaluate ack --task 3                          # as the requester; signs the acknowledgement
```

## Layout on disk

```
<data-dir>/
  hub.db                     accounts, node keys, workflow rows (SQLite)
  ledgers/registry.jsonl     publish events; the registry is rebuilt from this at start
  ledgers/certification.jsonl decision log; cases are replayed from this at start
  ledgers/credit.jsonl       contributions, releases, reciprocity decisions
  blobs/<aa>/<sha256>        weights, content-addressed
  secret.key, attestation-root.key
```

Every `.jsonl` chain has a `.head.json` anchor next to it. `GET /api/v1/ledgers/verify`
checks all three.

## Configuration

Environment variables, all optional:

| Variable | Default | Meaning |
|---|---|---|
| `OPENMED_DATA_DIR` | `./openmed-data` | where everything is stored |
| `OPENMED_ATTESTATION_MODE` | `nodekey` | `nodekey`: site-signed quotes over software measurements; `mock`: hub signs a declared identity (demo) |
| `OPENMED_APPROVED_MEASUREMENTS` | — | comma-separated measurement digests approved at startup (maintainers add more at runtime) |
| `OPENMED_APPROVED_CODE_IDENTITIES` | `openmed-training-pipeline@v1` | mock mode only: identities the gate approves |
| `OPENMED_CERT_K` / `OPENMED_CERT_MIN_INSTITUTIONS` | 2 / 2 | approvals and distinct institutions needed to certify |
| `OPENMED_TECHNICAL_REVIEWERS_MIN` / `OPENMED_CLINICAL_REVIEWERS_MIN` | 2 / 2 | board composition among approving reviewers |
| `OPENMED_RECIPROCITY_MIN_SERVED` | 2 | attested evaluations a site must have served before it can request one |
| `OPENMED_RECIPROCITY_GRACE` | 1 | free requests for a newly joined site |
| `OPENMED_MAX_UPLOAD_BYTES` | 512 MiB | weights size limit |
| `OPENMED_HUB_NAME`, `OPENMED_HUB_URL`, `OPENMED_SECRET_KEY` | — | cosmetics and the cookie/keyring secret |

## Account security

- **Passwords**: PBKDF2-HMAC-SHA256, per-user salt, at least 10 characters, not the email or
  name, not on the common-password list.
- **Two-factor authentication**: TOTP (RFC 6238) plus ten single-use recovery codes
  (`/account`, or `POST /api/v1/auth/2fa/enroll` then `/confirm`). Required for the roles in
  `OPENMED_REQUIRE_2FA_ROLES` (default: admin, maintainer, both reviewer roles): signing a
  review, granting roles, approving measurements, resolving appeals and revoking are refused
  without it. API login takes the code as `totp`; the web login has a second step.
- **Sessions**: HttpOnly SameSite=Lax cookies signed with the hub secret and bound to the
  user's security stamp; a password or second-factor change rotates the stamp and signs
  every other session out. Cookies are `Secure` when `OPENMED_HUB_URL` is https or
  `OPENMED_SECURE_COOKIES=1`.
- **Throttling**: `OPENMED_LOGIN_RATE_LIMIT` attempts per address per ten minutes; the account
  locks for `30 * 2^n` seconds after `OPENMED_LOCKOUT_THRESHOLD` failures, capped at
  `OPENMED_LOCKOUT_MAX_SECONDS`.
- **CSRF**: double-submit token on every HTML form (`openmed_csrf` cookie + `csrf` field).
- **Membership**: joining needs the institution's invite code, an email in
  `allowed_email_domains`, or an institution admin's approval; pending members cannot act.
  Institution admins manage this on `/account`; the first account of an institution is its
  admin.
- **API tokens**: random 256-bit, stored hashed, revocable.
- **Headers**: CSP, X-Frame-Options DENY, nosniff, Referrer-Policy, HSTS behind HTTPS.
  Set `OPENMED_TRUST_PROXY=1` only behind a proxy you control, so the audit log records the
  real client address.
- **Audit log**: `/admin` and `GET /api/v1/admin/audit`.

Not yet: email verification and self-service password reset (both need outgoing mail),
WebAuthn security keys, single sign-on.

## What is and is not a security boundary here

Read this before quoting the hub as an enforced control.

- **Attestation is software, not hardware -- and the difference is documented per
  mode.** In the default `nodekey` mode a submission carries a quote produced *at the
  site*: `openmed submit` measures the installed `trustfed` package, the training script
  and any extra files (`trustfed.attestation.software_measurement`), requests a
  single-use challenge, and signs measurement + config hash + nonce with the node key
  that completed the handshake. The hub holds only the public key
  (`trustfed.attestation.NodeKeyAttestor`), checks the signature, freshness, the
  challenge binding, and that the measurement is on the allow-list maintainers publish
  (`/admin`, `POST /api/v1/attestation/measurements`). What that proves: the holder of
  that site's key vouched for this code digest, now. What it cannot prove: anything about
  a site whose *operator* lies, since the operator holds the key -- that residual risk is
  what Byzantine-resilient merging and multi-site evaluation exist for. In `mock` mode
  (`OPENMED_ATTESTATION_MODE=mock`, demo only) the hub signs a declared code identity and
  measures nothing. A TEE backend replaces the node key with a hardware root of trust
  behind the same `Attestor` interface.
- **Two review paths.** The web form uses a key the hub holds for each reviewer
  (`review()` in the library; `hub_signed=true` on the record). The CLI can register a
  client-held key; then the hub only verifies detached signatures
  (`submit_signature()`), and cannot mint that reviewer's approval. The decision log
  records which path was used.
- **Evaluation acknowledgements** are signed with a hub-held per-institution service key.
  That makes the reciprocity count depend on a second party's *action* in the hub, not
  on a key the hub cannot use; a node-key-signed acknowledgement is the next step.
- **Dispute actions** (dispute, remediate, appeal, revoke) are recorded with the acting
  user's email and `actor_authenticated: false` on the ledger: the hub authenticated the
  session, the ledger entry itself is not signed by the actor.
- **No patient data.** The hub stores weights, cards, reports and signatures. Evaluation
  runs at the evaluator's site; only metrics come back.
- **Sessions and tokens**: PBKDF2 passwords, HMAC-signed cookies, SHA-256 token hashes.
  Put the hub behind TLS.
