# OpenMed (Open NeuroAI)

**A governed commons for clinical AI models — federated training that tolerates malicious
sites, tamper-evident provenance, and certification no single institution can grant
itself.**

> **OpenMed (Open NeuroAI)** is the ecosystem; **`trustfed`** is the Python package that implements it.
> The two names refer to the same project.
> Our system is new and non-profit. Initially, we will focus on the Neuroscience AI ecosystem.
> The community name and similar details are still under discussion, since there are already many projects using the name "OpenMed."

[![CI](https://github.com/RussellYSW/OpenMed/actions/workflows/ci.yml/badge.svg)](https://github.com/RussellYSW/OpenMed/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

📖 **[Documentation](https://russellysw.github.io/OpenMed/)** ·
🚀 **[Quickstart](https://russellysw.github.io/OpenMed/quickstart.html)** ·
🏛 **[Governance](https://russellysw.github.io/OpenMed/governance.html)** ·
🤝 **[Contributing](CONTRIBUTING.md)**

---

## Why this exists

Patient privacy keeps training data inside hospital walls, so most clinical models are
trained at a single center. A model that has only ever seen one hospital's patients,
scanners, and practice patterns may not generalize beyond that institution — and may
encode its inequities.

Three questions then have no good answer:

- **Is the model any good?** External validation cohorts are hard to reach.
- **Is it safe and fair enough to use?** No independent review exists, and nobody tells the
  treating physician, in clinical terms, where the model stops being trustworthy.
- **Why start from zero?** Another team may already have trained it, but there is no
  trusted path to obtain and fine-tune it.

Existing infrastructure covers parts of this. Model zoos distribute weights but do not
review, evaluate, or secure them. Federated benchmarks evaluate but do not govern or track
provenance. Federated-learning frameworks ([Flower](https://flower.ai),
[NVIDIA FLARE](https://github.com/NVIDIA/NVFlare), [Fed-BioMed](https://fedbiomed.org),
[NeuroFLAME](https://neuroflame.org)) orchestrate training but assume **every participating
site is honest**.

OpenMed is the layer that stops assuming that.

## Threat model

**Mutually distrusting sites, any minority of which may be actively malicious.** No site
releases raw patient data, and no site is trusted to self-report what code it ran.

| Risk | Control | Module |
|---|---|---|
| Unapproved or tampered client contributes an update | Attestation gate; nonce freshness available but **off by default** | [`trustfed/attestation`](trustfed/attestation) |
| Poisoned updates from clients that pass attestation | Byzantine-robust aggregation | [`trustfed/aggregation`](trustfed/aggregation) |
| Version substitution or silent rollback | Append-only signed hash chain | [`trustfed/ledger`](trustfed/ledger) |
| A site certifying its own model | k-of-n across distinct institutions | [`trustfed/certification`](trustfed/certification) |
| A published model leaking its training data | Screen for the cheapest membership-inference attack — **not a privacy guarantee** | [`trustfed/quality`](trustfed/quality) |
| Contribute-nothing free riding | Reciprocity policy | [`trustfed/incentives`](trustfed/incentives) |

Every row above is a control that ships and is tested. Several hold only under conditions
worth reading — see [Security caveats](#security-caveats).

## Quickstart

Python 3.9+. The only required runtime dependency is numpy. No data, credentials, GPU, or
network access needed — every example generates synthetic cohorts in-process from a seed.

```bash
git clone https://github.com/RussellYSW/OpenMed.git
cd OpenMed
pip install -e ".[dev]"
python -m pytest -q
```

Then run any of these:

```bash
python examples/parkinson_decline/run_demo.py   # defenses under attack
python examples/model_commons/run_demo.py       # publish → certify → derive → verify
python -m trustfed.benchmark                    # the full defense grid
```

### Expected output (synthetic data, 8 sites, 3 malicious)

```
scenario                                  test AUC  accepted  rejected
--------------------------------------------------------------------------
Baseline: no attack, FedAvg                  0.795         8         0
Scaling attack, FedAvg [naive]               0.732         8         0
Scaling attack, Coordinate median            0.760         8         0
Scaling attack, Trimmed mean                 0.760         8         0
Scaling attack, Multi-Krum                   0.760         8         0
Scaling attack, Norm clipping                0.792         8         0
Sign-flip attack, FedAvg [naive]             0.218         8         0
Sign-flip attack, Multi-Krum                 0.775         8         0
ADAPTIVE attack, FedAvg [naive]              0.245         8         0
ADAPTIVE attack, Coordinate median           0.793         8         0
ADAPTIVE attack, Multi-Krum                  0.794         8         0
ADAPTIVE attack, Norm clipping               0.420         8         0
Code tampering + attestation ON              0.790         5         3
```

Reading it honestly: naive averaging falls to **0.218** at worst. Under the static scaling
attack the robust rules land between **0.760 and 0.792** against a clean baseline of
**0.795** — they *contain* the attack at a cost of roughly 0.035 AUC, rather than restoring
the baseline exactly. The `ADAPTIVE` rows are the ones that matter most: norm clipping
drops to **0.420** against an attacker who knows the defense, because bounding influence is
not the same as removing it. With attestation on, the three code-tampered sites are
rejected before aggregation (`accepted 5, rejected 3`).

These numbers are regenerated by the demo itself, which prints this block. They are valid
only for the configuration shown.

## Components

Two planes. The **learning plane** produces models from data that never leaves its
institution; the **trust plane** decides which models are admitted and makes that record
checkable by someone who trusts none of the participants.

| # | Component | Package | What it does |
|---|---|---|---|
| 1 | Secure federated co-training | [`federated`](trustfed/federated), [`aggregation`](trustfed/aggregation) | Coordinate median, trimmed mean, Krum / Multi-Krum, norm and centered clipping — each with its documented tolerance bound. Pluggable client selection including reputation weighting. |
| 2 | Tamper-evident lineage | [`ledger`](trustfed/ledger) | Append-only signed hash chain. `LedgerBackend.verify_chain()` detects modification and reordering; truncation only against an anchor (see caveats). |
| 3 | Model registry | [`registry`](trustfed/registry) | Content-addressed bundles: weights, model card, pipeline attestation, evaluation report, fine-tuning manual. Derived models link to their parent, forming a lineage graph that `ModelRegistry.verify_lineage()` walks to a root. |
| 4 | Multi-party certification | [`certification`](trustfed/certification) | k-of-n sign-off across distinct institutions, conflict-of-interest rules, append-only decision log, explicit dispute and appeal state machine. |
| 5 | Automated quality analysis | [`quality`](trustfed/quality) | Seven independent checks including subgroup performance gaps and a membership-inference leakage screen. Runs with no LLM required; the leakage check skips unless member/non-member scores are supplied. |
| 6 | Credit and reciprocity | [`incentives`](trustfed/incentives) | Citable identifiers, release attribution, contributor standing mapped to governance weight, and a reciprocity rule enforced in code. |

Supporting: [`attestation`](trustfed/attestation) (the admission gate),
[`attack`](trustfed/attack) (what the defenses are measured against, including an adaptive
adversary), [`benchmark`](trustfed/benchmark) (the shared harness),
[`data`](trustfed/data) (synthetic non-IID cohorts).

Full detail on the [architecture page](https://russellysw.github.io/OpenMed/architecture.html).
([docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) covers the learning plane only.)

## Example

```python
from trustfed.data import make_federated_parkinson
from trustfed.federated import Client, Server, APPROVED_CODE_IDENTITY
from trustfed.aggregation import multi_krum
from trustfed.attestation import MockSoftwareAttestor, measure_code

sites, X_test, y_test = make_federated_parkinson(n_sites=6, seed=0)

attestor = MockSoftwareAttestor(
    root_key=b"demo-root-key",
    approved_measurements=[measure_code(APPROVED_CODE_IDENTITY)],
)

clients = [Client(s.site_id, s.X, s.y, attestor=attestor) for s in sites]

server = Server(
    n_features=X_test.shape[1],
    aggregator=multi_krum,
    attestor=attestor,
    n_byzantine=1,
)

for r in server.fit(clients, X_test, y_test, rounds=10):
    print(f"round {r.round:>2}  auc={r.auc:.3f}  "
          f"accepted={r.n_accepted}  rejected={r.n_rejected}")
```

## Security caveats

Read these before treating any of it as a deployed control. They are stated here rather
than buried because a control you misunderstand is worse than one you know you lack.

- **The shipped attestor is a software mock.** It exercises the protocol and the
  accept/reject logic. It is *not* a hardware root of trust and *not* a security boundary.
  Implement the `Attestor` interface against SGX/TDX or SEV-SNP quoting APIs for a real
  deployment — nothing else changes.
- **Replay protection is off by default.** Nonce freshness exists and is tested, but it
  requires constructing the attestor with a `NonceStore` and a policy with
  `require_nonce=True`. Without that — including in the shipped demo — a captured quote
  replays indefinitely.
- **Truncation detection requires an anchor.** `verify_chain()` always detects modification
  and reordering. It detects a *missing tail* only when checked against an anchor:
  `FileLedger`'s `.head.json` sidecar, or an external `Checkpoint` passed as `expected=`.
  An unanchored verdict reports `anchored=False` and cannot rule out truncation.
  `InMemoryLedger` has no anchor.
- **The default ledger key is a constant in the source.** It is for demos only. Anyone
  reading this repository can forge a consistent chain signed with it — supply your own
  signer for anything real.
- **The ledger signs with Ed25519 when `cryptography` is importable, HMAC otherwise.** The
  HMAC fallback gives integrity under a shared secret; it is not a signature and gives no
  non-repudiation, so a verifier can forge blocks. Check
  `trustfed.ledger.HAVE_CRYPTOGRAPHY`. A clean `pip install -e ".[dev]"` does **not**
  install `cryptography`, so the default is HMAC.
- **The chain is tamper-evident, not immutable.** A holder of the signing key and the
  storage can rewrite it consistently. What it buys you is detectability by someone holding
  an independent anchor.
- **The reviewer keyring models the policy layer, not a PKI.** `submit_signature()`
  verifies a detached signature the authority did not produce; `review()` is a
  demo convenience that signs on the reviewer's behalf.
- **No-self-certification is fully enforced only when a case is opened with the bundle
  itself**, since the owner institution is then inside the content hash. A case opened with
  a declared owner is marked `self_asserted: true` on the log.
- **Reviewer-role composition is not enforced in code.** `ThresholdPolicy` enforces k-of-n
  across distinct institutions; `Reviewer.role` is a free string the policy never consults.

## Data policy

**No patient data ships with this project and none is required to run it.** Every example,
test, and benchmark generates synthetic cohorts in-process from a fixed seed. CI fails the
build if a data-like file is ever committed. See [docs/DATA.md](docs/DATA.md).

## Contributing

Contributions do not have to be framework code. Returning a model, or an evaluation of
someone else's model, is a first-class contribution.

Where an outside contributor has the most leverage right now:

- **A real TEE attestation backend** behind the existing `Attestor` interface.
- **Adaptive attacks.** The harness is only as honest as the strongest attack in it.
- **Fine-tuning manual review** by clinicians who will say where the schema fails to
  capture what a treating physician needs to know.
- **Integration adapters** so existing federated frameworks can use the robust aggregator
  and admission gate without switching.

Start with [CONTRIBUTING.md](CONTRIBUTING.md) and the
[community page](https://russellysw.github.io/OpenMed/community.html).
Report vulnerabilities privately per [SECURITY.md](SECURITY.md), not in a public issue.

## Citing

See [CITATION.cff](CITATION.cff).

## License

[Apache-2.0](LICENSE). By contributing you agree your contribution is licensed under Apache-2.0; see [CONTRIBUTING.md](CONTRIBUTING.md).
