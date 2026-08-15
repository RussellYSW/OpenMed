"""Synthetic inputs and wiring shared by the demo's six sections.

Everything here is generated in-process from a seed. There is no patient data,
no network call, and no persistent state outside the output directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from trustfed.attestation import (
    AttestationPolicy,
    MockSoftwareAttestor,
    NonceStore,
    measure_code,
)
from trustfed.certification import (
    CertificationAuthority,
    FineTuningManual,
    Reviewer,
    ReviewerKeyring,
    ThresholdPolicy,
)
from trustfed.incentives import CounterpartyRegistry, CreditLedger
from trustfed.ledger import FileLedger, HmacSigner
from trustfed.registry import ModelCard, ModelRegistry, PipelineAttestation

APPROVED_PIPELINE = "openmed-training-pipeline@v1"
UNAPPROVED_PIPELINE = "openmed-training-pipeline@patched-locally"
ROOT_KEY = b"demo-attestation-root-key-not-a-secret"
YEAR = 2026


def rule(title: str) -> None:
    """Print a section heading."""
    print()
    print(title)
    print("-" * 74)


def synthetic_weights(seed: int, n_features: int = 12) -> bytes:
    """Return deterministic synthetic 'weights' bytes for a seed."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=n_features).astype(np.float64).tobytes()


def synthetic_metrics(seed: int) -> Dict[str, float]:
    """Return plausible-looking but entirely synthetic evaluation metrics."""
    rng = np.random.default_rng(seed + 1000)
    auc = float(np.clip(0.78 + rng.normal(0, 0.02), 0.5, 0.95))
    return {"auc": round(auc, 3), "accuracy": round(auc - 0.05, 3)}


def make_card(
    name: str, version: str, owner: str, metrics: Dict[str, float]
) -> ModelCard:
    """Build a complete model card (Mitchell et al. 2019 sections)."""
    return ModelCard(
        model_details={
            "name": name,
            "version": version,
            "owner": owner,
            "date": f"{YEAR}-01-15",
            "model_type": "logistic regression on tabular features (numpy)",
            "license": "Apache-2.0",
        },
        intended_use={
            "primary_use": "research prototype: flag rapid decline risk",
            "out_of_scope": "diagnosis, treatment selection, deployment",
        },
        factors={"groups": ["site", "age_band"], "instrumentation": "synthetic"},
        metrics={"reported": sorted(metrics), "decision_threshold": 0.5},
        evaluation_data={"dataset": "synthetic held-out split", "n": 500},
        training_data={"dataset": "synthetic multi-site cohort", "n": 3000},
        quantitative_analyses=dict(metrics),
        ethical_considerations={
            "risks": "synthetic data only; no clinical validity established"
        },
        caveats_and_recommendations={
            "caveats": "demo artefact; numbers come from a seeded RNG"
        },
    )


def make_manual(site: str) -> FineTuningManual:
    """Build a complete clinician-readable fine-tuning manual."""
    return FineTuningManual(
        intended_use={
            "task": "flag rapid decline risk within 24 months",
            "population": "adults in the synthetic demo cohort",
            "care_setting": "research use only",
            "not_intended_for": "diagnosis, triage, or any clinical decision",
        },
        data={
            "sources": [f"synthetic cohort generated at {site}"],
            "n_records": 3000,
            "inclusion_criteria": "baseline plus 24-month synthetic follow-up",
            "label_definition": "threshold on the synthetic composite score",
        },
        preprocessing={
            "steps": ["drop incomplete visits", "z-score per site"],
            "normalization": "z-score with training-set statistics",
            "missing_data": "listwise deletion",
        },
        hyperparameters={
            "optimizer": "full-batch gradient descent",
            "learning_rate": 0.05,
            "epochs": 25,
            "batch_size": "full batch",
        },
        failure_modes={
            "known_failure_modes": ["site shift", "feature scaling drift"],
            "monitoring": "subgroup AUC review before any reuse",
        },
        clinical_caveats={
            "human_oversight": "advisory only; a clinician decides",
            "contraindications": "do not use outside the stated population",
            "escalation": "notify the site lead if drift is observed",
        },
    )


def make_attestor(nonces: NonceStore) -> MockSoftwareAttestor:
    """Build the demo attestor: nonce required, quotes expire after 5 minutes."""
    policy = AttestationPolicy(
        approved_measurements={measure_code(APPROVED_PIPELINE)},
        require_nonce=True,
        max_age_seconds=300,
    )
    return MockSoftwareAttestor(ROOT_KEY, policy=policy, nonce_store=nonces)


def attest(
    attestor: MockSoftwareAttestor, site: str, identity: str
) -> PipelineAttestation:
    """Run a full challenge-response and return the attestation record."""
    nonce = attestor.issue_nonce(site)
    quote = attestor.generate_quote(site, identity, "training-config-v1", nonce=nonce)
    return PipelineAttestation.from_quote(quote, attestor=attestor, expected_nonce=nonce)


def evaluator_keys() -> Tuple[CounterpartyRegistry, Dict[str, HmacSigner]]:
    """Return the counterparty registry and the per-site keys that back it.

    The keys are derived from fixed strings so the demo is reproducible; a
    committed key is a published key, which is fine here and nowhere else.
    """
    signers = {
        site: HmacSigner(f"demo-key-{site}".encode("utf-8"), key_label=site)
        for site in ("site_a", "site_b", "site_c")
    }
    registry = CounterpartyRegistry()
    for site, signer in signers.items():
        registry.register(site, signer.verifier())
    return registry, signers


def build(
    out_dir: Path, seed: int
) -> Tuple[ModelRegistry, CertificationAuthority, CreditLedger, Dict[str, HmacSigner]]:
    """Wire up the registry, authority and credit ledger on file-backed chains."""
    registry = ModelRegistry(FileLedger(out_dir / "registry.jsonl"))
    keyring = ReviewerKeyring(seed=b"demo-reviewer-keys")
    keyring.register(Reviewer("rev_a1", "INST_A", role="ml"))
    keyring.register(Reviewer("rev_b1", "INST_B", role="clinical"))
    keyring.register(Reviewer("rev_c1", "INST_C", role="clinical"))
    authority = CertificationAuthority(
        keyring,
        FileLedger(out_dir / "certification.jsonl"),
        policy=ThresholdPolicy(k=2, min_institutions=2),
    )
    counterparties, signers = evaluator_keys()
    credit = CreditLedger(
        FileLedger(out_dir / "credit.jsonl"), counterparties=counterparties
    )
    return registry, authority, credit, signers


__all__ = [
    "APPROVED_PIPELINE",
    "ROOT_KEY",
    "UNAPPROVED_PIPELINE",
    "YEAR",
    "attest",
    "build",
    "evaluator_keys",
    "make_attestor",
    "make_card",
    "make_manual",
    "rule",
    "synthetic_metrics",
    "synthetic_weights",
]
