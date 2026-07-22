"""End-to-end TrustFed demo: securing collaborative Parkinson's prognosis.

Story
-----
Six "hospitals" hold private, non-IID synthetic cohorts and jointly improve one
rapid-decline classifier without sharing patient data. The point of the demo is
not the learning -- it is the *integrity* of that collaboration: we show what a
compromised or malicious site can do to the shared model, and how TrustFed's two
security controls neutralize it:

  1. Attestation rejects a site running *tampered* (unapproved) code before its
     update is ever aggregated.
  2. Byzantine-robust aggregation absorbs *update-poisoning* from sites that pass
     attestation but submit harmful updates (corrupted data, compromised host,
     hardware fault).

Run:  python examples/parkinson_decline/run_demo.py
No patient data is used -- everything is synthetic (see trustfed/data).
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Allow running from a source checkout without installing the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from trustfed.aggregation.robust import get_aggregator  # noqa: E402
from trustfed.attack.poison import get_attack  # noqa: E402
from trustfed.attestation.attestor import MockSoftwareAttestor, measure_code  # noqa: E402
from trustfed.data.synthetic import make_federated_parkinson  # noqa: E402
from trustfed.federated.client import APPROVED_CODE_IDENTITY, Client  # noqa: E402
from trustfed.federated.server import Server  # noqa: E402

ROUNDS = 40
LOCAL_EPOCHS = 5
ROOT_KEY = b"tee-vendor-root-of-trust-DEMO-ONLY"
APPROVED_MEASUREMENT = measure_code(APPROVED_CODE_IDENTITY)


def build_clients(sites, *, n_malicious=0, attack_name="none", attestor=None,
                  tamper_code=False):
    """Turn site data into clients; optionally make the last ``n_malicious``
    ones adversarial."""
    attack = get_attack(attack_name)
    clients = []
    for i, s in enumerate(sites):
        is_mal = i >= len(sites) - n_malicious and n_malicious > 0
        code_identity = (
            "trustfed-client@TAMPERED" if (is_mal and tamper_code)
            else APPROVED_CODE_IDENTITY
        )
        clients.append(
            Client(
                s.site_id, s.X, s.y,
                attestor=attestor,
                code_identity=code_identity,
                malicious=is_mal,
                attack=attack if is_mal else None,
                seed=100 + i,
            )
        )
    return clients


def run(label, *, aggregator, n_malicious, attack_name, use_attestation,
        tamper_code, sites, X_test, y_test, n_features):
    attestor = (
        MockSoftwareAttestor(ROOT_KEY, {APPROVED_MEASUREMENT})
        if use_attestation else None
    )
    clients = build_clients(
        sites, n_malicious=n_malicious, attack_name=attack_name,
        attestor=attestor, tamper_code=tamper_code,
    )
    # Trimmed mean must drop at least ``n_malicious`` values from each end.
    trimmed_beta = min(0.49, (n_malicious + 0.5) / len(sites))
    server = Server(
        n_features,
        aggregator=get_aggregator(aggregator),
        attestor=attestor,
        n_byzantine=n_malicious,
        trimmed_beta=trimmed_beta,
    )
    server.fit(clients, X_test, y_test, rounds=ROUNDS, local_epochs=LOCAL_EPOCHS)
    final = server.history[-1]
    return final, label


def main():
    np.set_printoptions(precision=3, suppress=True)
    sites, X_test, y_test = make_federated_parkinson(
        n_sites=6, samples_per_site=220, test_size=1200, seed=7
    )
    n_features = X_test.shape[1]
    n_mal = 2  # 2 of 6 sites are malicious

    print("=" * 74)
    print("TrustFed demo -- integrity of collaborative Parkinson's prognosis")
    print(f"  sites=6  malicious={n_mal}  rounds={ROUNDS}  features={n_features}")
    print("  (synthetic data -- no patient records)")
    print("=" * 74)

    scenarios = [
        # label, aggregator, n_malicious, attack, attestation, tamper_code
        ("Baseline: no attack, FedAvg",        "fedavg",     0, "none",     False, False),
        ("Attack (scaling), FedAvg [naive]",   "fedavg",     n_mal, "scaling", False, False),
        ("Attack (scaling), Coordinate median","median",     n_mal, "scaling", False, False),
        ("Attack (scaling), Trimmed mean",     "trimmed_mean", n_mal, "scaling", False, False),
        ("Attack (scaling), Multi-Krum",       "multi_krum", n_mal, "scaling", False, False),
        ("Attack (sign-flip), FedAvg [naive]", "fedavg",     n_mal, "sign_flip", False, False),
        ("Attack (sign-flip), Multi-Krum",     "multi_krum", n_mal, "sign_flip", False, False),
        ("Code tampering + attestation ON",    "fedavg",     n_mal, "scaling", True,  True),
    ]

    results = []
    for label, agg, nm, atk, att, tamper in scenarios:
        final, lbl = run(
            label, aggregator=agg, n_malicious=nm, attack_name=atk,
            use_attestation=att, tamper_code=tamper,
            sites=sites, X_test=X_test, y_test=y_test, n_features=n_features,
        )
        results.append((lbl, final))

    print(f"\n{'scenario':<40}{'test AUC':>10}{'accepted':>10}{'rejected':>10}")
    print("-" * 74)
    for lbl, f in results:
        print(f"{lbl:<40}{f.auc:>10.3f}{f.n_accepted:>10}{f.n_rejected:>10}")

    print("\nTakeaways")
    print("-" * 74)
    print("* Under poisoning, naive FedAvg collapses toward AUC 0.5.")
    print("* Median / trimmed-mean / Multi-Krum keep AUC near the clean baseline.")
    print("* With attestation ON, the 2 code-tampered sites are rejected before")
    print("  aggregation (see 'rejected' column) -- FedAvg is safe again because")
    print("  the malicious updates never enter the average.")


if __name__ == "__main__":
    main()
