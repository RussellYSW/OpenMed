"""End-to-end TrustFed demo: securing collaborative Parkinson's prognosis.

Story
-----
Eight "hospitals" hold private, non-IID synthetic cohorts and jointly improve one
rapid-decline classifier without sharing patient data. The point of the demo is
not the learning -- it is the *integrity* of that collaboration: we show what a
compromised or malicious site can do to the shared model, and how TrustFed's
controls neutralize it.

  1. Attestation rejects a site running *tampered* (unapproved) code before its
     update is ever aggregated.
  2. Byzantine-robust aggregation absorbs *update-poisoning* from sites that pass
     attestation but submit harmful updates (corrupted data, compromised host,
     hardware fault).
  3. An *adaptive* adversary -- one that sees the honest updates and knows which
     aggregation rule is running -- shows which of those defenses survive a real
     threat model rather than a convenient one.
  4. Reputation-weighted client selection reduces how often a badly-behaving
     site is even asked to contribute.
  5. The quality analyzer produces the report a reviewer would read before the
     model is admitted to the commons.

Run:  python examples/parkinson_decline/run_demo.py
No patient data is used -- everything is synthetic (see trustfed/data).
Every number printed below is produced by this script on this machine.

**This script's output is the only valid source for any figure quoted about
this demo elsewhere in the repository** (README, docs, proposal text). The
scenario table is printed twice: once for reading, and once inside a
``README block`` fence in exactly the shape the README uses, so a maintainer can
regenerate that section verbatim instead of transcribing it. If the
configuration below changes (site count, attacker count, rounds, seed), the
numbers change with it and any figure copied earlier is stale.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Allow running from a source checkout without installing the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from trustfed.aggregation.robust import get_aggregator  # noqa: E402
from trustfed.aggregation.tolerance import (  # noqa: E402
    describe,
    recommended_trim_beta,
)
from trustfed.attack.adaptive import AdaptiveAdversary  # noqa: E402
from trustfed.attack.poison import get_attack  # noqa: E402
from trustfed.attestation.attestor import (  # noqa: E402
    MockSoftwareAttestor,
    measure_code,
)
from trustfed.data.cohort import CohortSpec  # noqa: E402
from trustfed.data.synthetic import make_cohort  # noqa: E402
from trustfed.federated.client import APPROVED_CODE_IDENTITY, Client  # noqa: E402
from trustfed.federated.selection import ReputationSelector  # noqa: E402
from trustfed.federated.server import Server  # noqa: E402
from trustfed.metrics import binary_metrics, subgroup_metrics  # noqa: E402
from trustfed.quality import QualityAnalyzer, QualityEvidence  # noqa: E402

ROUNDS = 40
LOCAL_EPOCHS = 5
N_SITES = 8
N_MALICIOUS = 3
ROOT_KEY = b"tee-vendor-root-of-trust-DEMO-ONLY"
APPROVED_MEASUREMENT = measure_code(APPROVED_CODE_IDENTITY)
SEED = 7


def build_clients(sites, *, n_malicious=0, attack_name="none", attestor=None,
                  tamper_code=False):
    """Turn site data into clients; optionally make the last ``n_malicious``
    ones adversarial."""
    attack = get_attack(attack_name) if attack_name != "none" else None
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


def run(*, aggregator, n_malicious, attack_name, use_attestation, tamper_code,
        sites, X_test, y_test, n_features, adaptive=False, selector=None):
    """Run one federated training configuration and return its final round."""
    attestor = (
        MockSoftwareAttestor(ROOT_KEY, {APPROVED_MEASUREMENT})
        if use_attestation else None
    )
    clients = build_clients(
        sites,
        n_malicious=0 if adaptive else n_malicious,
        attack_name=attack_name,
        attestor=attestor,
        tamper_code=tamper_code,
    )
    adversary = None
    byz_ids = ()
    if adaptive:
        adversary = AdaptiveAdversary(target_aggregator=aggregator, seed=SEED)
        byz_ids = [c.client_id for c in clients[len(clients) - n_malicious:]]

    server = Server(
        n_features,
        aggregator=get_aggregator(aggregator),
        attestor=attestor,
        n_byzantine=n_malicious,
        trimmed_beta=recommended_trim_beta(len(sites), n_malicious),
        adversary=adversary,
        byzantine_ids=byz_ids,
        selector=selector,
    )
    server.fit(clients, X_test, y_test, rounds=ROUNDS, local_epochs=LOCAL_EPOCHS)
    return server


def scenario_table(cohort):
    """Section 1-3: attacks and defenses, printed as a comparison table."""
    sites, X_test, y_test = cohort.as_tuple()
    n_features = cohort.n_features
    n_mal = N_MALICIOUS

    scenarios = [
        # label, aggregator, n_malicious, attack, attestation, tamper, adaptive
        ("Baseline: no attack, FedAvg", "fedavg", 0, "none", False, False, False),
        ("Scaling attack, FedAvg [naive]", "fedavg", n_mal, "scaling", False, False, False),
        ("Scaling attack, Coordinate median", "median", n_mal, "scaling", False, False, False),
        ("Scaling attack, Trimmed mean", "trimmed_mean", n_mal, "scaling", False, False, False),
        ("Scaling attack, Multi-Krum", "multi_krum", n_mal, "scaling", False, False, False),
        ("Scaling attack, Norm clipping", "norm_clip", n_mal, "scaling", False, False, False),
        ("Sign-flip attack, FedAvg [naive]", "fedavg", n_mal, "sign_flip", False, False, False),
        ("Sign-flip attack, Multi-Krum", "multi_krum", n_mal, "sign_flip", False, False, False),
        ("ADAPTIVE attack, FedAvg [naive]", "fedavg", n_mal, "adaptive", False, False, True),
        ("ADAPTIVE attack, Coordinate median", "median", n_mal, "adaptive", False, False, True),
        ("ADAPTIVE attack, Multi-Krum", "multi_krum", n_mal, "adaptive", False, False, True),
        ("ADAPTIVE attack, Norm clipping", "norm_clip", n_mal, "adaptive", False, False, True),
        ("Code tampering + attestation ON", "fedavg", n_mal, "scaling", True, True, False),
    ]

    header = f"{'scenario':<40}{'test AUC':>10}{'accepted':>10}{'rejected':>10}"
    lines = [header, "-" * 74]
    aucs = {}
    print(f"\n{header}")
    print("-" * 74)
    for label, agg, nm, atk, att, tamper, adaptive in scenarios:
        server = run(
            aggregator=agg, n_malicious=nm,
            attack_name="none" if adaptive else atk,
            use_attestation=att, tamper_code=tamper, adaptive=adaptive,
            sites=sites, X_test=X_test, y_test=y_test, n_features=n_features,
        )
        f = server.history[-1]
        row = f"{label:<40}{f.auc:>10.3f}{f.n_accepted:>10}{f.n_rejected:>10}"
        aucs[label] = f.auc
        lines.append(row)
        print(row)
    print_readme_block(lines, aucs)
    return None


def print_readme_block(lines, aucs):
    """Print the scenario table in the exact shape the README quotes.

    The README's "Expected output" section must be regenerated from this block,
    not hand-edited: a figure that is not in this output is not a figure this
    code produces.
    """
    print("\nREADME block -- copy verbatim; this run is the only source for")
    print("these numbers, and they are valid only for the configuration above:")
    print()
    print(f"Expected output (synthetic data, {N_SITES} sites, "
          f"{N_MALICIOUS} malicious):")
    print()
    print("```")
    for line in lines:
        print(line)
    print("```")
    baseline = aucs.get("Baseline: no attack, FedAvg")
    robust = [
        aucs[k] for k in aucs
        if k.startswith("Scaling attack,") and "FedAvg" not in k
    ]
    if baseline is not None and robust:
        print()
        print(
            f"Accompanying sentence, computed from this run: naive averaging "
            f"falls to {min(aucs.values()):.3f} at worst; under the static "
            f"scaling attack the robust rules land between {min(robust):.3f} "
            f"and {max(robust):.3f} against a clean baseline of "
            f"{baseline:.3f}, i.e. they contain the attack at a cost of up to "
            f"{baseline - min(robust):.3f} AUC rather than restoring the "
            f"baseline exactly."
        )


def reputation_demo(cohort):
    """Section 4: does behavioural reputation separate the poisoning sites?"""
    sites, X_test, y_test = cohort.as_tuple()
    server = run(
        aggregator="median", n_malicious=N_MALICIOUS, attack_name="sign_flip",
        use_attestation=False, tamper_code=False,
        sites=sites, X_test=X_test, y_test=y_test, n_features=cohort.n_features,
        selector=ReputationSelector(0.75, seed=SEED),
    )
    records = server.client_history.records
    malicious = {s.site_id for s in sites[len(sites) - N_MALICIOUS:]}
    print(f"\n{'site':<12}{'reputation':>12}{'rounds selected':>18}{'ground truth':>16}")
    print("-" * 60)
    for site_id in sorted(records):
        rec = records[site_id]
        truth = "poisoning" if site_id in malicious else "honest"
        print(f"{site_id:<12}{rec.reputation:>12.3f}{rec.rounds_selected:>18}{truth:>16}")
    honest_mean = np.mean(
        [r.reputation for k, r in records.items() if k not in malicious]
    )
    mal_mean = np.mean([r.reputation for k, r in records.items() if k in malicious])
    print(f"\n  mean reputation: honest {honest_mean:.3f} vs poisoning {mal_mean:.3f}")
    print("  (behavioural evidence, not proof: an honest site with unusual")
    print("   case-mix also drifts down. Reputation limits exposure; it does")
    print("   not identify attackers.)")
    return server


def quality_demo(cohort, server):
    """Section 5: the automated quality report a reviewer would receive."""
    test_scores = server.model.predict_proba(cohort.X_test)
    train_X = np.concatenate([s.X for s in cohort.sites])
    train_scores = server.model.predict_proba(train_X)
    overall = binary_metrics(cohort.y_test, test_scores)

    bundle = {
        "bundle_id": "pd-rapid-decline-demo",
        "model_card": {
            "model_details": "Logistic regression over 12 synthetic features.",
            "intended_use": "Demonstration of the TrustFed review pipeline.",
            "out_of_scope_use": "Not for clinical use; trained on synthetic data.",
            "training_data": "Synthetic non-IID multi-site cohort (trustfed.data).",
            "evaluation_data": f"Held-out synthetic sample, n={int(overall['n'])}.",
            "metrics": "ROC AUC and accuracy.",
            "limitations": "Synthetic data only; no external validation.",
            # 'ethical_considerations' deliberately omitted, to show the check firing.
            "contact": "see repository maintainers",
        },
        "metrics": {
            "auc": overall["auc"],
            "accuracy": overall["accuracy"],
            "n_eval": int(overall["n"]),
        },
        "attestation": {"measurement": APPROVED_MEASUREMENT, "signature": "demo"},
        "lineage": [{"round": r.round, "auc": r.auc} for r in server.history],
        "weights_path": None,
    }
    evidence = QualityEvidence(
        member_scores=train_scores,
        nonmember_scores=test_scores,
        subgroup_metrics=subgroup_metrics(
            cohort.y_test, test_scores, cohort.test_groups
        ),
        weights=server.model.get_params(),
        notes={"generated_by": "examples/parkinson_decline/run_demo.py"},
    )
    report = QualityAnalyzer().analyze(bundle, evidence)
    print(f"\noverall: {report.overall_status.value.upper()}   {report.counts}")
    print(f"\n{'check':<44}{'status':>8}")
    print("-" * 74)
    for r in report.results:
        print(f"{r.title:<44}{r.status.value.upper():>8}   {r.summary[:60]}")
    print("\nNarrative (deterministic template summarizer, no LLM required):\n")
    print(report.narrative)
    return report


def main():
    """Run all five demo sections."""
    np.set_printoptions(precision=3, suppress=True)
    cohort = make_cohort(
        CohortSpec(
            n_sites=N_SITES, samples_per_site=220, test_size=1200,
            label_skew=0.7, feature_shift=0.4, seed=SEED,
        )
    )

    print("=" * 74)
    print("TrustFed demo -- integrity of collaborative Parkinson's prognosis")
    print(f"  sites={N_SITES}  malicious={N_MALICIOUS}  rounds={ROUNDS}  "
          f"features={cohort.n_features}")
    print("  (synthetic data -- no patient records)")
    print("=" * 74)
    print("\nSite outcome prevalence (non-IID):",
          np.round(cohort.site_prevalences(), 3))

    print("\n[1-3] ATTACKS AND DEFENSES")
    scenario_table(cohort)

    print("\nDocumented tolerance of the rules used above:")
    for name in ("fedavg", "median", "multi_krum", "norm_clip"):
        print(f"  - {describe(name, N_SITES)}")

    print("\n[4] REPUTATION-WEIGHTED CLIENT SELECTION")
    server = reputation_demo(cohort)

    print("\n[5] AUTOMATED QUALITY REPORT")
    quality_demo(cohort, server)

    print("\nTakeaways")
    print("-" * 74)
    print("* Under naive FedAvg, poisoning collapses the shared model.")
    print("* Median / trimmed-mean / Multi-Krum contain the static attacks at a")
    print("  cost of a few hundredths of AUC against the clean baseline -- they")
    print("  bound the damage, they do not restore the baseline exactly -- and")
    print("  the adaptive rows show which of them still hold when the attacker")
    print("  knows the defense. Read the printed table, not this sentence.")
    print("* Norm clipping bounds an attacker's influence but does not remove it,")
    print("  which the adaptive row makes visible.")
    print("* With attestation ON, code-tampered sites are rejected before")
    print("  aggregation, so even FedAvg stays clean.")
    print("* The quality report is what a reviewer reads; it flags what is")
    print("  missing or uneven, and claims nothing about clinical validity.")
    print("\nReproduce the full defense grid with:  python -m trustfed.benchmark")


if __name__ == "__main__":
    main()
