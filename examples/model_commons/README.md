# Model-commons demo

End-to-end walkthrough of the OpenMed trust plane on **synthetic data only**:

```bash
python examples/model_commons/run_demo.py           # writes to a temp dir
python examples/model_commons/run_demo.py --out ./out --seed 11
```

Only `numpy` is required. `cryptography` is used for Ed25519 signatures when it
is importable; without it the ledger falls back to HMAC, which the code marks
plainly as *not* a security boundary.

## What the script does

| Step | Component | What you should see |
|---|---|---|
| 1. Attested publish | `trustfed.attestation`, `trustfed.registry` | A site publishes a bundle after answering a fresh attestation challenge. A site running unapproved code gets `verified=False` and the publish is refused with `AttestationRequiredError`. |
| 2. Certification | `trustfed.certification` | The case is opened with the bundle itself, so the owner institution is read from `published_by` rather than declared; the owning institution's own reviewer is then refused (`self_certification`); one institution's approval is not enough; two institutions certify the base model. |
| 3. Derivation | `trustfed.registry` | A second site fine-tunes the certified base; the derived bundle is auto-linked to its parent. |
| 4. Lineage | `trustfed.registry` | `verify_lineage` walks the derived model back to its root and prints a structured verdict plus a Mermaid graph. |
| 5. Credit | `trustfed.incentives` | Contributions are recorded, the release gets a citable identifier, standing tiers map to voting weight, and a reciprocity request is refused with a machine-readable reason until the requesting site has served as an evaluator twice. Evaluation service is recorded through the counterparty path, so each entry carries the evaluated site's signature; an unattested self-claim is recorded but does not satisfy the rule. |
| 6. Tamper evidence | `trustfed.ledger` | Dropping the tail of a chain is caught against its anchor (`truncated`); deleting the anchor as well does not turn the verdict green, it turns it into `missing_checkpoint` with `ok=False`. Editing one field then fails with `payload_mutated`. |

## Layout

`run_demo.py` is the driver. Each numbered section is one module under
`openmed_demo/`: `publish.py`, `certify.py`, `derive.py` (sections 3 and 4),
`credit.py`, `tamper.py`, with the synthetic inputs and wiring in
`fixtures.py`.

## What this demo is not

- **Not real data.** Every cohort, weight vector and metric is generated
  in-process from the `--seed` value. No patient data ships with this project.
- **Not a security boundary.** The attestor is a software mock: it signs the
  code identity a caller declares and measures nothing that is actually running.
  Real TEE backends implement the same `Attestor` interface.
- **Not clinical validation.** Certification here is a governance record —
  who signed, from which institution, under which policy. It is not regulatory
  clearance and says nothing about clinical safety.
- **Not a distributed ledger.** The hash chain is local and gives tamper
  *evidence* to a reader who knows the chain head, not tamper prevention.
- **Not an authenticated process.** `review()` signs on a reviewer's behalf with
  a key the authority itself holds, which is why the demo prints a note saying
  so. `CertificationAuthority.submit_signature()` is the path where reviewers
  sign elsewhere and the authority only verifies; `ReviewerKeyring.verify_only()`
  builds an authority that holds no private keys at all.

The output directory contains the three chains the demo writes —
`registry.jsonl`, `certification.jsonl`, `credit.jsonl` — each with a
`.head.json` checkpoint sidecar. The sidecar is what makes truncation visible;
it lives next to the ledger, so it defends against an editor of a single file,
not against anyone who can rewrite the whole directory. Publish `checkpoint()`
output externally for that.
