# TrustFed

**Software-supply-chain integrity for open-source clinical AI — attested admission and Byzantine-resilient aggregation that harden the collaborative-training infrastructure hospitals rely on. Demonstrated on multi-site Parkinson's-disease prognosis.**

[![CI](https://github.com/AnonymousUser08/Trusted-TEE-for-Data-Sharing/actions/workflows/ci.yml/badge.svg)](https://github.com/AnonymousUser08/Trusted-TEE-for-Data-Sharing/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

---

## Why this exists

Open-source clinical AI increasingly depends on **collaborative, multi-institution
model training** — several hospitals improving one shared model without moving
patient data across institutional boundaries. Mature platforms already provide the
plumbing ([Flower](https://flower.ai), [NVIDIA FLARE](https://github.com/NVIDIA/NVFlare),
[Fed-BioMed](https://fedbiomed.org), [NeuroFLAME/COINSTAC](https://neuroflame.org)).

**The gap is trust, not plumbing.** That infrastructure assumes **every
participating site is honest**. A single compromised, faulty, or malicious site
can silently poison the shared model or contribute from unapproved code — an
integrity and supply-chain problem for the whole ecosystem, and in a clinical
setting a patient-safety problem, not just an accuracy one.

**TrustFed is the trust layer that closes that gap.** It hardens the one step the
existing infrastructure leaves trusting-by-default — update admission and
aggregation — with two complementary security controls, and is designed to be
**adopted by** those platforms rather than replace them:

| Layer | Threat it addresses | Module |
|------|--------------------|--------|
| **Remote attestation** | An unapproved / tampered client trying to contribute | [`trustfed/attestation`](trustfed/attestation) |
| **Byzantine-robust aggregation** | Poisoned updates from clients that pass attestation (corrupted data, compromised host, hardware fault) | [`trustfed/aggregation`](trustfed/aggregation) |

The two layers are designed to be **adopted by** existing frameworks (as a robust
aggregator + an admission gate), not to compete with them.

## Quickstart

```bash
git clone https://github.com/AnonymousUser08/Trusted-TEE-for-Data-Sharing.git
cd trustfed
python -m pip install -e .          # only dependency: numpy
python examples/parkinson_decline/run_demo.py
```

Expected output (synthetic data, 6 sites, 2 malicious):

```
scenario                                  test AUC  accepted  rejected
--------------------------------------------------------------------------
Baseline: no attack, FedAvg                  0.861         6         0
Attack (scaling), FedAvg [naive]             0.727         6         0
Attack (scaling), Coordinate median          0.861         6         0
Attack (scaling), Trimmed mean               0.861         6         0
Attack (scaling), Multi-Krum                 0.861         6         0
Attack (sign-flip), FedAvg [naive]           0.253         6         0
Attack (sign-flip), Multi-Krum               0.852         6         0
Code tampering + attestation ON              0.859         4         2
```

Naive averaging collapses under attack (AUC 0.25–0.73); the robust rules hold at
the clean baseline (~0.86); and with attestation on, the two code-tampered sites
are rejected *before* aggregation.

## The Parkinson's use case

The demo trains a **rapid cognitive/motor decline** classifier across six
simulated hospitals, using features that mirror a real prognostic study
(regional structural-MRI atrophy rates + baseline clinical scores). This is the
motivating application: let movement-disorder centers pool statistical power on
rare, heterogeneous cohorts **without** sharing records, and **without** trusting
every site unconditionally.

> ⚠️ **No patient data ships with this project.** Everything under
> [`trustfed/data`](trustfed/data) is synthetic. Real cohorts (PPMI, PDBP,
> institutional registries) are governed by Data Use Agreements and must never be
> committed here. See [`docs/DATA.md`](docs/DATA.md).

## Minimal API

```python
import numpy as np
from trustfed.data.synthetic import make_federated_parkinson
from trustfed.aggregation.robust import multi_krum
from trustfed.federated.client import Client
from trustfed.federated.server import Server

sites, X_test, y_test = make_federated_parkinson(n_sites=6)
clients = [Client(s.site_id, s.X, s.y) for s in sites]

server = Server(X_test.shape[1], aggregator=multi_krum, n_byzantine=2)
server.fit(clients, X_test, y_test, rounds=40, verbose=True)
print("final AUC:", server.history[-1].auc)
```

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — how the pieces fit, and the threat model
- [`docs/DATA.md`](docs/DATA.md) — data policy and how to plug in real (controlled-access) cohorts
- [`SECURITY.md`](SECURITY.md) — what the mock attestor does and does **not** guarantee

## Status & roadmap

This is an early (v0.1) reference implementation. The attestation layer is a
**software mock**; real TEE backends (Intel SGX/TDX via DCAP, AMD SEV-SNP) plug
in behind the `Attestor` interface. See [ROADMAP](docs/ARCHITECTURE.md#roadmap).

Contributions welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

Apache-2.0 — see [`LICENSE`](LICENSE).

## Citation

If you use TrustFed, please cite it — see [`CITATION.cff`](CITATION.cff).
