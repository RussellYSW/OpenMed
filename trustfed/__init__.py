"""OpenMed / TrustFed: a governed commons for clinical AI models.

Hospitals train models they cannot share, review, or safely reuse. This package
implements the machinery a shared model commons needs in order to work between
institutions that do not trust each other: federated training that tolerates a
malicious minority, tamper-evident provenance, and certification that no single
site can grant itself.

The system assumes **mutually distrusting sites, any minority of which may be
actively malicious**. No site releases raw patient data, and no site is trusted
to self-report what code it ran.

Components
----------
The package is organised as two planes. The *learning plane* produces models
from data that never leaves its institution; the *trust plane* decides which
models are admitted, records what happened, and makes that record verifiable.

Learning plane:

* ``trustfed.data`` -- deterministic synthetic, non-IID multi-site cohorts.
  No patient data ships with this project and none is required to run it.
* ``trustfed.models`` -- the reference model (numpy-only logistic regression),
  deliberately small so the whole federated loop stays auditable.
* ``trustfed.federated`` -- the coordinator and client, plus pluggable client
  selection (random, loss-based, reputation-weighted).
* ``trustfed.aggregation`` -- Byzantine-robust aggregation rules (coordinate
  median, trimmed mean, Krum / Multi-Krum, norm clipping, centered clipping),
  each with its documented tolerance bound in ``tolerance_bound()``.
* ``trustfed.attack`` -- the attacks those defenses are measured against,
  including an adaptive adversary that targets the aggregator in use.
* ``trustfed.benchmark`` -- a shared harness so a contributed defense is
  compared on the same scenarios as everything else.

Trust plane:

* ``trustfed.attestation`` -- the admission gate. An update is aggregated only
  if it carries a quote over an approved code measurement. Nonce-based freshness
  is available but **off by default**: build the attestor with a ``NonceStore``
  and a policy with ``require_nonce=True``, or a captured quote replays
  indefinitely. The shipped parkinson demo does not enable it.
* ``trustfed.ledger`` -- an append-only signed hash chain.
  ``LedgerBackend.verify_chain()`` always detects modification and reordering.
  It detects *truncation* only against an anchor (``FileLedger``'s
  ``.head.json`` sidecar, or a ``Checkpoint`` passed as ``expected=``); an
  unanchored verdict reports ``anchored=False``.
* ``trustfed.registry`` -- content-addressed model bundles (weights, model
  card, pipeline attestation, evaluation report, fine-tuning manual). Derived
  models link to their parent, so the registry accumulates a lineage *graph*
  and ``ModelRegistry.verify_lineage()`` can walk a model's whole ancestry.
* ``trustfed.certification`` -- k-of-n sign-off across distinct institutions,
  conflict-of-interest rules, and an append-only decision log with an explicit
  dispute and appeal state machine.
* ``trustfed.quality`` -- automated analysis of a submission: model-card
  completeness, metric sanity, subgroup performance gaps, membership-inference
  leakage screening, parameter-norm outliers, lineage presence.
* ``trustfed.incentives`` -- credit and reciprocity: citable identifiers,
  release attribution, contributor standing, and a reciprocity rule with teeth
  (to receive an evaluation, a site serves as an evaluator).

Security caveats
----------------
Read these before treating any of it as a deployed control.

* The shipped attestor is a **software mock**. It exercises the protocol and
  the accept/reject logic; it is not a hardware root of trust and not a
  security boundary. Implement the ``Attestor`` interface against SGX/TDX or
  SEV-SNP quoting APIs for a real deployment.
* The ledger signs with Ed25519 when ``cryptography`` is importable and falls
  back to HMAC otherwise. The HMAC fallback gives integrity under a shared
  secret; it is not a signature and gives no non-repudiation. Check
  ``trustfed.ledger.HAVE_CRYPTOGRAPHY`` to see which you have.
* The reviewer keyring models the *policy* layer -- who may sign for whom, and
  what satisfies a quorum -- not a PKI.
* The default ledger signing key is derived from a constant in this source
  tree. It is for demos only: anyone reading the repository can forge a
  consistent chain signed with it. Supply your own signer.
* ``MembershipInferenceCheck`` implements the global-confidence-threshold
  attack and *skips* unless member/non-member scores are supplied. A pass means
  that attack failed, not that the model is private.

Quickstart
----------
Two runnable demonstrations, both on synthetic data::

    python examples/parkinson_decline/run_demo.py   # defenses under attack
    python examples/model_commons/run_demo.py       # publish -> certify -> derive
    python -m trustfed.benchmark                    # the full defense grid

See ``docs/`` for the documentation site, or
https://github.com/RussellYSW/OpenMed.
"""

from __future__ import annotations

__version__ = "0.1.0"

# --- learning plane -------------------------------------------------------
from trustfed.models.logistic import LogisticRegressionModel
from trustfed.federated.server import Server, RoundResult
from trustfed.federated.client import Client, Update

# --- trust plane ----------------------------------------------------------
from trustfed.attestation import Attestor, MockSoftwareAttestor, AttestationPolicy
from trustfed.ledger import LedgerBackend, InMemoryLedger, FileLedger
from trustfed.registry import ModelRegistry, ModelBundle, ModelCard
from trustfed.certification import CertificationAuthority, ThresholdPolicy, FineTuningManual
from trustfed.incentives import CreditLedger, ReciprocityPolicy

__all__ = [
    "__version__",
    # learning plane
    "LogisticRegressionModel",
    "Server",
    "RoundResult",
    "Client",
    "Update",
    # trust plane
    "Attestor",
    "MockSoftwareAttestor",
    "AttestationPolicy",
    "LedgerBackend",
    "InMemoryLedger",
    "FileLedger",
    "ModelRegistry",
    "ModelBundle",
    "ModelCard",
    "CertificationAuthority",
    "ThresholdPolicy",
    "FineTuningManual",
    "CreditLedger",
    "ReciprocityPolicy",
]


def __getattr__(name: str):
    """Lazily expose the heavier subpackages as attributes.

    ``trustfed.benchmark`` and ``trustfed.quality`` are not imported eagerly:
    they pull in the aggregation and attack registries and are only needed when
    a caller actually reaches for them. ``import trustfed.benchmark`` works as
    normal; this only makes ``trustfed.benchmark`` resolve after a bare
    ``import trustfed`` as well.
    """
    if name in ("benchmark", "quality", "aggregation", "attack", "data", "metrics"):
        import importlib

        return importlib.import_module(f"trustfed.{name}")
    raise AttributeError(f"module 'trustfed' has no attribute {name!r}")
