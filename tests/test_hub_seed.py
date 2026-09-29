"""The demo seeder exercises every workflow; it must produce the promised state."""

from __future__ import annotations

from openmed_hub.app import create_app
from openmed_hub.config import HubSettings
from openmed_hub.seed import seed_demo
from openmed_hub.services import HubServices


import pytest


@pytest.mark.parametrize("mode", ["nodekey", "mock"])
def test_seed_demo_produces_certified_root_and_blocked_gate(tmp_path, mode):
    settings = HubSettings(data_dir=tmp_path / "hub", secret_key="s", attestation_root_key=b"r", attestation_mode=mode)
    settings.data_dir.mkdir()
    app = create_app(settings)
    svc = HubServices(app.state.trustplane, settings, app.state.session_factory())
    summary = seed_demo(svc)
    assert summary["attestation_mode"] == mode
    assert summary["root_model"]["state"] == "certified"
    assert summary["root_model"]["identifier"].startswith("oid:")
    assert summary["derived_model"]["state"] == "under_review"
    assert summary["gate_blocked_model"]["state"] == "blocked"
    assert summary["evaluation_task"]["state"] == "acknowledged"
    metrics = svc.metrics()
    assert metrics["certified_models"] == 1
    assert metrics["independent_installations"]["numerator"] == 1  # the external lab's node
    assert all(v["ok"] for v in svc.tp.verify_ledgers().values())
