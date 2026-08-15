"""Tests for the shared defense benchmark harness."""

import json

import numpy as np
import pytest

from trustfed.benchmark import (
    BenchmarkError,
    BenchmarkReport,
    Runner,
    Scenario,
    compare_reports,
    default_grid,
    grid,
    load_report,
)
from trustfed.benchmark.__main__ import build_parser, build_scenarios, main


def small_grid(seed=0):
    """A three-row grid that runs in well under a second."""
    return [
        Scenario(aggregator="fedavg", attack="none", n_byzantine=0, rounds=8, seed=seed),
        Scenario(aggregator="fedavg", attack="sign_flip", rounds=8, seed=seed),
        Scenario(aggregator="multi_krum", attack="sign_flip", rounds=8, seed=seed),
    ]


# -- scenario validation ----------------------------------------------------


def test_scenario_names_are_stable_and_descriptive():
    s = Scenario(aggregator="median", attack="adaptive", n_sites=8, n_byzantine=2, seed=4)
    assert s.name == Scenario(
        aggregator="median", attack="adaptive", n_sites=8, n_byzantine=2, seed=4
    ).name
    assert "median" in s.name and "adaptive" in s.name
    assert s.byzantine_fraction == pytest.approx(0.25)


def test_scenario_rejects_incoherent_configurations():
    with pytest.raises(BenchmarkError):
        Scenario(n_sites=4, n_byzantine=4)  # nobody honest
    with pytest.raises(BenchmarkError):
        Scenario(attack="sign_flip", n_byzantine=0)  # attack with no attacker
    with pytest.raises(BenchmarkError):
        Scenario(rounds=0)


def test_clean_twin_removes_the_attack():
    twin = Scenario(aggregator="krum", attack="scaling", n_byzantine=3).clean_twin()
    assert twin.attack == "none" and twin.n_byzantine == 0


def test_grid_emits_one_clean_row_per_aggregator_and_seed():
    scenarios = grid(
        aggregators=("fedavg", "median"),
        attacks=("none", "sign_flip"),
        n_byzantine=(1, 2),
        seeds=(0, 1),
    )
    clean = [s for s in scenarios if s.attack == "none"]
    assert len(clean) == 4  # 2 aggregators x 2 seeds
    assert len(scenarios) == 4 + 8  # plus 2 aggregators x 2 f x 2 seeds


def test_default_grid_covers_every_aggregator_and_attack():
    scenarios = default_grid(seed=0)
    assert {s.aggregator for s in scenarios} >= {"fedavg", "median", "multi_krum"}
    assert {s.attack for s in scenarios} >= {"none", "sign_flip", "label_flip", "adaptive"}


# -- running ----------------------------------------------------------------


def test_runner_reproduces_identical_numbers_from_the_same_seed():
    a = Runner().run(small_grid(seed=3))
    b = Runner().run(small_grid(seed=3))
    assert compare_reports(json.loads(a.to_json()), json.loads(b.to_json())) == []


def test_changing_the_seed_changes_the_numbers():
    a = Runner().run(small_grid(seed=3))
    b = Runner().run(small_grid(seed=4))
    diffs = compare_reports(json.loads(a.to_json()), json.loads(b.to_json()))
    assert diffs  # different cohorts must not coincidentally agree


def test_robust_rule_beats_fedavg_under_the_same_attack():
    report = Runner().run(small_grid(seed=5))
    by_name = {r.scenario.aggregator: r for r in report if r.scenario.attack == "sign_flip"}
    assert by_name["multi_krum"].final_auc > by_name["fedavg"].final_auc + 0.1


def test_runner_reports_documented_tolerance_per_row():
    report = Runner().run(small_grid(seed=6))
    fedavg_attacked = [
        r for r in report if r.scenario.aggregator == "fedavg" and r.scenario.n_byzantine
    ][0]
    assert fedavg_attacked.tolerance_max_f == 0
    assert fedavg_attacked.within_tolerance is False
    krum_attacked = [r for r in report if r.scenario.aggregator == "multi_krum"][0]
    assert krum_attacked.within_tolerance is True


def test_runner_rejects_an_unknown_attack_name():
    with pytest.raises(BenchmarkError):
        Runner().run_one(Scenario(attack="teleportation", n_byzantine=1, rounds=2))


def test_runner_refuses_an_empty_scenario_list():
    with pytest.raises(BenchmarkError):
        Runner().run([])


def test_data_and_adaptive_attacks_both_execute():
    scenarios = [
        Scenario(aggregator="median", attack="label_flip", rounds=5, seed=1),
        Scenario(aggregator="median", attack="adaptive", rounds=5, seed=1),
    ]
    report = Runner().run(scenarios)
    assert len(report) == 2
    assert all(0.0 <= r.final_auc <= 1.0 for r in report)


def test_attestation_scenario_rejects_tampered_sites():
    scenario = Scenario(
        aggregator="fedavg",
        attack="scaling",
        n_byzantine=2,
        rounds=6,
        attestation=True,
        tamper_code=True,
        seed=2,
    )
    result = Runner().run_one(scenario)
    assert result.n_rejected_total == 2 * result.rounds_run
    assert result.final_auc > 0.7  # the poison never reaches the average


def test_default_grid_completes_quickly():
    import time

    started = time.perf_counter()
    report = Runner().run(default_grid(seed=0))
    elapsed = time.perf_counter() - started
    assert len(report) == 49
    assert elapsed < 55.0, f"default grid took {elapsed:.1f}s"


# -- reporting --------------------------------------------------------------


def test_report_json_is_stable_and_omits_timing_by_default():
    report = Runner().run(small_grid(seed=7))
    text = report.to_json()
    assert "wall_time_s" not in text
    assert report.to_json() == text
    parsed = json.loads(text)
    assert parsed["n_results"] == 3
    assert "synthetic" in parsed["provenance"]


def test_report_markdown_flags_out_of_tolerance_rows():
    report = Runner().run(small_grid(seed=8))
    md = report.to_markdown()
    assert "| `fedavg` | sign_flip |" in md
    assert "| NO |" in md  # fedavg tolerates no attackers
    assert "Documented Byzantine tolerance" in md
    assert "not clinical performance claims" in md


def test_report_delta_vs_clean_uses_the_matching_baseline():
    report = Runner().run(small_grid(seed=9))
    attacked = [r for r in report if r.scenario.aggregator == "fedavg" and r.scenario.n_byzantine][0]
    delta = report.delta_vs_clean(attacked)
    assert delta is not None and delta < 0


def test_report_write_round_trips_to_disk(tmp_path):
    report = Runner().run(small_grid(seed=10))
    written = report.write(tmp_path)
    assert written["json"].exists() and written["markdown"].exists()
    reloaded = load_report(written["json"])
    assert compare_reports(reloaded, json.loads(report.to_json())) == []
    assert written["json"].read_text(encoding="utf-8").endswith("\n")


def test_empty_report_refuses_to_write(tmp_path):
    with pytest.raises(BenchmarkError):
        BenchmarkReport().write(tmp_path)


def test_worst_case_summary_picks_the_lowest_attacked_auc():
    report = Runner().run(small_grid(seed=11))
    worst = report.worst_case()
    assert set(worst) == {"fedavg", "multi_krum"}
    assert worst["fedavg"] <= min(
        r.final_auc for r in report if r.scenario.aggregator == "fedavg" and r.scenario.n_byzantine
    )


# -- command line -----------------------------------------------------------


def test_cli_builds_the_requested_grid():
    args = build_parser().parse_args(
        ["--seed", "2", "--sites", "6", "--byzantine", "1", "--aggregators", "median"]
    )
    scenarios = build_scenarios(args)
    assert {s.aggregator for s in scenarios} == {"median"}
    assert {s.n_sites for s in scenarios} == {6}
    assert {s.seed for s in scenarios} == {2}


def test_cli_writes_both_artifacts(tmp_path, capsys):
    code = main(
        [
            "--seed", "1",
            "--rounds", "4",
            "--aggregators", "median",
            "--attacks", "none", "sign_flip",
            "--byzantine", "2",
            "--out", str(tmp_path),
            "--quiet",
        ]
    )
    assert code == 0
    assert (tmp_path / "benchmark.json").exists()
    assert (tmp_path / "benchmark.md").exists()
    assert np.isfinite(load_report(tmp_path / "benchmark.json")["results"][0]["final_auc"])
