"""Tests for the registry's lineage graph and ``verify_lineage`` (Component 3).

Covers the derivation graph, the ancestry walk, every issue code the verdict can
carry, rebuilding from the ledger, and the shipped model-commons demo. Bundle
and model-card behaviour lives in ``test_registry_bundle.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from trustfed.attestation import AttestationPolicy, MockSoftwareAttestor, measure_code
from trustfed.ledger import FileLedger, HmacSigner
from trustfed.registry import (
    BundleNotFoundError,
    EvaluationReport,
    ModelBundle,
    ModelCard,
    ModelRegistry,
    PipelineAttestation,
    WeightsRef,
)

CODE_IDENTITY = "openmed-pipeline@v1"


def make_attestor() -> MockSoftwareAttestor:
    return MockSoftwareAttestor(
        b"registry-test-root",
        policy=AttestationPolicy(approved_measurements={measure_code(CODE_IDENTITY)}),
    )


def make_attestation(site: str = "site_a", identity: str = CODE_IDENTITY):
    att = make_attestor()
    quote = att.generate_quote(site, identity, "training-config")
    return PipelineAttestation.from_quote(quote, attestor=att)


def make_card(name: str = "pd-decline", version: str = "1.0", owner: str = "INST_A"):
    return ModelCard(
        model_details={
            "name": name,
            "version": version,
            "owner": owner,
            "date": "2026-01-01",
            "model_type": "logistic regression (numpy)",
            "license": "Apache-2.0",
        },
        intended_use={"primary_use": "research prototype on synthetic cohorts"},
        factors={"groups": ["site", "age_band"]},
        metrics={"reported": ["auc", "accuracy"]},
        evaluation_data={"dataset": "synthetic held-out split"},
        training_data={"dataset": "synthetic multi-site cohort"},
        quantitative_analyses={"auc": 0.81},
        ethical_considerations={"risks": "not clinically validated"},
        caveats_and_recommendations={"caveats": "synthetic data only"},
    )


def make_eval(auc: float = 0.81):
    return EvaluationReport(
        dataset_id="synthetic-holdout",
        n_samples=500,
        metrics={"auc": auc},
        subgroup_metrics={"site_a": {"auc": auc - 0.02}},
        protocol="fixed split, seed 7",
        evaluated_by="INST_B",
        seed=7,
    )


def make_weights(blob: bytes = b"weights-v1"):
    return WeightsRef.from_bytes("file://weights/v1.npz", blob)


def publish_root(registry: ModelRegistry, **overrides):
    kwargs = dict(
        name="pd-decline",
        version="1.0",
        weights=make_weights(),
        model_card=make_card(),
        evaluation=make_eval(),
        attestation=make_attestation(),
        published_by="INST_A",
    )
    kwargs.update(overrides)
    return registry.publish(**kwargs)
# --------------------------------------------------------------------- lineage


def test_derived_publish_auto_links_to_parent():
    registry = ModelRegistry()
    root = publish_root(registry)
    child = registry.publish_derived(
        root.bundle_id,
        name="pd-decline",
        version="1.1",
        weights=make_weights(b"weights-v2"),
        model_card=make_card(version="1.1"),
        evaluation=make_eval(0.83),
        attestation=make_attestation("site_b"),
        published_by="INST_B",
    )
    assert child.parents == (root.bundle_id,)
    assert registry.children(root.bundle_id) == (child.bundle_id,)
    assert registry.ancestors(child.bundle_id) == (root.bundle_id,)
    assert registry.descendants(root.bundle_id) == (child.bundle_id,)
    assert registry.roots() == (root.bundle_id,)


def test_registry_accumulates_a_graph_not_a_list():
    registry = ModelRegistry()
    root = publish_root(registry)
    branch_a = registry.publish_derived(
        root.bundle_id,
        name="pd-decline",
        version="1.1-a",
        weights=make_weights(b"a"),
        model_card=make_card(version="1.1-a"),
        evaluation=make_eval(0.82),
        attestation=make_attestation("site_b"),
        published_by="INST_B",
    )
    branch_b = registry.publish_derived(
        root.bundle_id,
        name="pd-decline",
        version="1.1-b",
        weights=make_weights(b"b"),
        model_card=make_card(version="1.1-b"),
        evaluation=make_eval(0.80),
        attestation=make_attestation("site_c"),
        published_by="INST_C",
    )
    merged = registry.publish(
        name="pd-decline",
        version="2.0",
        weights=make_weights(b"merged"),
        model_card=make_card(version="2.0"),
        evaluation=make_eval(0.85),
        attestation=make_attestation("site_d"),
        published_by="INST_D",
        parents=(branch_a.bundle_id, branch_b.bundle_id),
    )
    graph = registry.lineage_graph(merged.bundle_id)
    assert set(graph.nodes) == {
        merged.bundle_id,
        branch_a.bundle_id,
        branch_b.bundle_id,
        root.bundle_id,
    }
    assert len(graph.edges) == 4
    assert graph.roots() == (root.bundle_id,)
    assert "graph TD" in graph.to_mermaid()

    verdict = registry.verify_lineage(merged.bundle_id)
    assert verdict.ok is True
    assert verdict.depth == 2
    assert verdict.roots == (root.bundle_id,)
    assert len(verdict.ancestry) == 4


def test_verify_lineage_reports_unknown_bundle():
    registry = ModelRegistry()
    with pytest.raises(BundleNotFoundError):
        registry.verify_lineage("openmed:bundle:sha256:" + "0" * 64)


def test_verify_lineage_detects_tampered_ancestor_content():
    registry = ModelRegistry()
    root = publish_root(registry)
    child = registry.publish_derived(
        root.bundle_id,
        name="pd-decline",
        version="1.1",
        weights=make_weights(b"v2"),
        model_card=make_card(version="1.1"),
        evaluation=make_eval(0.83),
        attestation=make_attestation("site_b"),
        published_by="INST_B",
    )
    assert registry.verify_lineage(child.bundle_id).ok is True

    # Someone edits the parent's evaluation in the registry's view.
    data = root.to_dict()
    data["evaluation"]["metrics"]["auc"] = 0.99
    registry._bundles[root.bundle_id] = ModelBundle.from_dict(data)

    verdict = registry.verify_lineage(child.bundle_id)
    assert verdict.ok is False
    assert "content_hash_mismatch" in verdict.codes
    assert "ledger_content_mismatch" in verdict.codes
    assert "FAILED" in verdict.summary()


def test_verify_lineage_detects_a_tampered_ledger(tmp_path: Path):
    path = tmp_path / "registry.jsonl"
    ledger = FileLedger(path, signer=HmacSigner(b"k"))
    registry = ModelRegistry(ledger)
    bundle = publish_root(registry)
    assert registry.verify_lineage(bundle.bundle_id).ok is True

    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["payload"]["published_by"] = "INST_IMPOSTOR"
    lines[0] = json.dumps(record, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    verdict = registry.verify_lineage(bundle.bundle_id)
    assert verdict.ok is False
    assert verdict.ledger_verified is False
    assert "ledger_chain_invalid" in verdict.codes


def test_verify_lineage_flags_a_missing_ledger_record():
    registry = ModelRegistry()
    bundle = publish_root(registry)
    orphan = ModelBundle.create(
        name="orphan",
        version="9.9",
        weights=make_weights(b"orphan"),
        model_card=make_card(name="orphan", version="9.9"),
        evaluation=make_eval(),
        attestation=make_attestation("site_x"),
        published_by="INST_X",
    )
    registry._bundles[orphan.bundle_id] = orphan  # inserted without publishing
    verdict = registry.verify_lineage(orphan.bundle_id)
    assert verdict.ok is False
    assert "missing_ledger_record" in verdict.codes
    assert registry.verify_lineage(bundle.bundle_id).ok is True


def test_verify_lineage_flags_a_missing_parent():
    registry = ModelRegistry()
    root = publish_root(registry)
    child = registry.publish_derived(
        root.bundle_id,
        name="pd-decline",
        version="1.1",
        weights=make_weights(b"v2"),
        model_card=make_card(version="1.1"),
        evaluation=make_eval(),
        attestation=make_attestation("site_b"),
        published_by="INST_B",
    )
    del registry._bundles[root.bundle_id]  # parent disappears from the registry
    verdict = registry.verify_lineage(child.bundle_id)
    assert verdict.ok is False
    assert "missing_parent" in verdict.codes


def test_registry_can_be_rebuilt_from_the_ledger(tmp_path: Path):
    path = tmp_path / "registry.jsonl"
    ledger = FileLedger(path, signer=HmacSigner(b"k"))
    registry = ModelRegistry(ledger)
    root = publish_root(registry)
    child = registry.publish_derived(
        root.bundle_id,
        name="pd-decline",
        version="1.1",
        weights=make_weights(b"v2"),
        model_card=make_card(version="1.1"),
        evaluation=make_eval(),
        attestation=make_attestation("site_b"),
        published_by="INST_B",
    )

    reopened = FileLedger(path, signer=HmacSigner(b"k"))
    rebuilt = ModelRegistry.rebuild_from_ledger(reopened)
    assert len(rebuilt) == 2
    assert rebuilt.get(child.bundle_id).parents == (root.bundle_id,)
    assert rebuilt.verify_lineage(child.bundle_id).ok is True


def test_bundle_carries_and_validates_a_fine_tuning_manual():
    from trustfed.certification import FineTuningManual

    manual = FineTuningManual(
        intended_use={
            "task": "t",
            "population": "p",
            "care_setting": "s",
            "not_intended_for": "n",
        },
        data={
            "sources": ["synthetic"],
            "n_records": 500,
            "inclusion_criteria": "c",
            "label_definition": "l",
        },
        preprocessing={"steps": ["z-score"], "normalization": "z", "missing_data": "drop"},
        hyperparameters={
            "optimizer": "sgd",
            "learning_rate": 0.05,
            "epochs": 10,
            "batch_size": 32,
        },
        failure_modes={"known_failure_modes": ["shift"], "monitoring": "monthly"},
        clinical_caveats={
            "human_oversight": "required",
            "contraindications": "none stated",
            "escalation": "clinical lead",
        },
    )
    registry = ModelRegistry()
    bundle = publish_root(registry, fine_tuning_manual=manual.to_dict())
    assert bundle.fine_tuning_manual is not None
    assert bundle.validate_manual().ok is True

    without = ModelBundle.create(
        name="m",
        version="1",
        weights=make_weights(),
        model_card=make_card(),
        evaluation=make_eval(),
        attestation=make_attestation(),
        published_by="INST_A",
    )
    assert without.validate_manual() is None



def test_model_commons_demo_runs(tmp_path: Path, capsys):
    """The shipped example must actually run, since the README points at it."""
    import importlib.util

    demo_path = Path(__file__).resolve().parents[1] / "examples" / "model_commons" / "run_demo.py"
    spec = importlib.util.spec_from_file_location("model_commons_demo", demo_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.main(["--out", str(tmp_path), "--seed", "3"]) == 0
    out = capsys.readouterr().out
    assert "certified base" in out
    assert "lineage OK" in out
    assert "after editing block 0   : ok=False" in out
    assert (tmp_path / "registry.jsonl").exists()
    assert (tmp_path / "credit.jsonl.head.json").exists()


def test_damage_elsewhere_in_a_shared_ledger_does_not_fail_this_lineage(
    tmp_path: Path,
):
    """A registry often shares a chain with certification and credit.

    One bad block belonging to another component must not be reported as
    though this bundle's own publish record had been touched.
    """
    path = tmp_path / "shared.jsonl"
    ledger = FileLedger(path, signer=HmacSigner(b"k"))
    registry = ModelRegistry(ledger)
    bundle = publish_root(registry)
    ledger.append({"event": "certification_event", "action": "certified", "n": 1})
    assert registry.verify_lineage(bundle.bundle_id).ok is True

    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[1])  # the certification block, not the publish block
    record["payload"]["action"] = "revoked"
    lines[1] = json.dumps(record, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    verdict = registry.verify_lineage(bundle.bundle_id)
    assert verdict.ledger_verified is False  # the log is damaged...
    assert "shared_ledger_unverified" in verdict.codes
    assert "ledger_chain_invalid" not in verdict.codes
    assert verdict.ok is True  # ...but not in a way that implicates this bundle
    assert "blocks=[1]" in verdict.issues[0].detail


def test_a_tampered_publish_block_is_attributed_to_the_lineage(tmp_path: Path):
    path = tmp_path / "shared.jsonl"
    ledger = FileLedger(path, signer=HmacSigner(b"k"))
    registry = ModelRegistry(ledger)
    bundle = publish_root(registry)
    ledger.append({"event": "credit_entry", "actor": "site_a"})

    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])  # this bundle's own publish record
    record["payload"]["bundle"]["published_by"] = "INST_IMPOSTOR"
    lines[0] = json.dumps(record, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    verdict = registry.verify_lineage(bundle.bundle_id)
    assert verdict.ok is False
    assert "ledger_chain_invalid" in verdict.codes
    assert "blocks=[0]" in verdict.issues[0].detail


def test_a_missing_anchor_on_a_shared_ledger_implicates_the_lineage(tmp_path: Path):
    """An unattributable failure gets no benefit of the doubt."""
    path = tmp_path / "shared.jsonl"
    ledger = FileLedger(path, signer=HmacSigner(b"k"))
    registry = ModelRegistry(ledger)
    bundle = publish_root(registry)
    ledger.checkpoint_path.unlink()

    verdict = registry.verify_lineage(bundle.bundle_id)
    assert verdict.ok is False
    assert "ledger_chain_invalid" in verdict.codes
    assert "missing_checkpoint" in verdict.issues[0].detail
