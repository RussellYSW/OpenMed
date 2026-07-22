# Architecture & threat model

## Overview

TrustFed adds a *zero-trust security layer* around the admission and aggregation
step of collaborative, multi-site model training (the technical setting is
cross-silo federated learning; the contribution is integrity, not learning). One
coordinating **server** and several **clients** (one per hospital / silo) run
synchronous rounds:

```
        ┌────────────────────────── round r ──────────────────────────┐
        │                                                              │
  broadcast θ_global                                                   │
        │                                                              │
        ▼                                                              │
  ┌───────────┐   local train + attest    ┌───────────────────────┐   │
  │  Client i │ ─────────────────────────▶ │  Update(θ_i, quote_i) │   │
  └───────────┘                            └───────────────────────┘   │
        │                                            │                 │
        │                                            ▼                 │
        │                               ┌────────────────────────┐     │
        │                               │ 1. verify attestation  │  ◀─ admission gate
        │                               │    (drop failed quotes)│     │
        │                               │ 2. robust aggregate    │  ◀─ Byzantine tolerance
        │                               │    survivors → θ_global│     │
        │                               │ 3. evaluate on test set│     │
        │                               └────────────────────────┘     │
        └──────────────────────────────────────────────────────────────┘
```

Code map:

| Concern | Module |
|--------|--------|
| Local model (numpy logistic regression) | `trustfed/models/logistic.py` |
| Synthetic non-IID cohorts | `trustfed/data/synthetic.py` |
| Admission gate (attestation) | `trustfed/attestation/attestor.py` |
| Byzantine-robust aggregation | `trustfed/aggregation/robust.py` |
| Attack simulations | `trustfed/attack/poison.py` |
| Client / Server orchestration | `trustfed/federated/` |

## Threat model

We assume up to `f` of `n` clients are **Byzantine** (arbitrary behavior:
compromised host, corrupted local data, malicious operator, hardware fault). The
server and the attestation root of trust are honest. Two distinct failure modes,
two distinct defenses:

1. **Unapproved / tampered code wants to join.**
   The admission gate requires every update to carry an attestation *quote* whose
   code *measurement* is on an allow-list and whose signature chains to a trusted
   root. Clients that fail are dropped before aggregation. Analogous to Intel
   SGX/TDX or AMD SEV-SNP remote attestation.

2. **Approved code still submits a harmful update.**
   Attestation cannot catch bad *data* or a compromised runtime that runs
   approved code. Robust aggregation bounds each update's influence:
   - **Coordinate median** — tolerates `f < n/2`.
   - **Trimmed mean** — drop the top/bottom `β` fraction per coordinate; needs
     `β·n ≥ f`.
   - **(Multi-)Krum** — pick the update(s) closest to their neighbors; needs
     `n > 2f + 2` (else falls back to median).

Neither layer alone is sufficient; together they cover both admission and
in-protocol faults.

## Design choices

- **numpy-only core.** The whole loop is auditable and runs anywhere. Swapping in
  a torch/sklearn model only changes `models/` and the parameter-vector packing.
- **Parameters as a flat vector.** Aggregators and attacks operate on a single
  1-D vector, so a new model just needs `get_params`/`set_params`.
- **Framework-neutral.** `aggregation/` and `attestation/` have no dependency on
  the rest of TrustFed and can be dropped into Flower / NVIDIA FLARE as a custom
  aggregator + client admission check.

## Roadmap

- [ ] Real TEE backends behind `Attestor` (SGX/TDX via DCAP, SEV-SNP).
- [ ] Secure aggregation / DP composition with the robust rules.
- [ ] Adapters for Flower and NVIDIA FLARE.
- [ ] Richer models (torch) and neuroimaging feature pipelines.
- [ ] Bulyan and centered-clipping aggregators; adaptive-attack evaluation.
