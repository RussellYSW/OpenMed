"""Executes benchmark scenarios and collects results.

The runner is the piece that makes contributed defenses comparable: it owns the
wiring from a :class:`~trustfed.benchmark.scenario.Scenario` to a concrete
federation, so that two aggregation rules differ only in the rule itself and not
in how their experiment was set up.

Determinism: every stochastic component (cohort, client attack randomness,
selector, adversary) is seeded from ``scenario.seed``, so re-running a scenario
reproduces its numbers exactly on the same numpy version.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import numpy as np

from trustfed.aggregation.robust import get_aggregator
from trustfed.aggregation.tolerance import (
    bounded_influence_bound,
    max_byzantine,
    recommended_trim_beta,
)
from trustfed.attack.adaptive import AdaptiveAdversary
from trustfed.attack.data_poison import DATA_ATTACKS, get_data_attack
from trustfed.attack.poison import ATTACKS, get_attack
from trustfed.benchmark.errors import BenchmarkError
from trustfed.benchmark.scenario import ADAPTIVE, Scenario, default_grid
from trustfed.data.synthetic import make_cohort
from trustfed.data.cohort import CohortSpec
from trustfed.federated.client import APPROVED_CODE_IDENTITY, Client
from trustfed.federated.selection import get_selector
from trustfed.federated.server import Server

if TYPE_CHECKING:  # pragma: no cover - import cycle broken at runtime
    from trustfed.benchmark.report import BenchmarkReport

TAMPERED_CODE_IDENTITY = "trustfed-client@TAMPERED"


@dataclass
class ScenarioResult:
    """Measured outcome of one scenario.

    ``final_auc`` is the headline number; ``mean_auc_tail`` averages the last
    few rounds so a single lucky round cannot flatter a defense.

    ``tolerance_max_f`` and ``within_tolerance`` describe the rule's
    *exact-recovery* bound only: ``within_tolerance`` says the scenario's ``f``
    is inside the range where the rule is claimed to recover the honest signal.
    For a bounded-influence rule such as ``norm_clip`` that range is ``f = 0``,
    so any attacked scenario is outside it; ``guarantee_kind`` names which claim
    is being tested and ``influence_bound`` carries the ``2*tau*f/n``
    displacement bound that applies instead (``None`` for threshold rules).
    """

    scenario: Scenario
    final_auc: float
    final_accuracy: float
    best_auc: float
    mean_auc_tail: float
    n_rejected_total: int
    rounds_run: int
    tolerance_max_f: int
    within_tolerance: bool
    wall_time_s: float = 0.0
    round_auc: List[float] = field(default_factory=list)
    guarantee_kind: str = "exact recovery"
    influence_bound: Optional[float] = None

    def to_dict(self, *, include_timing: bool = False) -> Dict[str, Any]:
        """JSON-serializable view.

        Timing is excluded by default so that two runs of the same grid produce
        byte-identical JSON.
        """
        out: Dict[str, Any] = {
            "scenario": self.scenario.to_dict(),
            "final_auc": round(float(self.final_auc), 6),
            "final_accuracy": round(float(self.final_accuracy), 6),
            "best_auc": round(float(self.best_auc), 6),
            "mean_auc_tail": round(float(self.mean_auc_tail), 6),
            "n_rejected_total": int(self.n_rejected_total),
            "rounds_run": int(self.rounds_run),
            "exact_recovery_max_f": int(self.tolerance_max_f),
            "f_within_exact_recovery_bound": bool(self.within_tolerance),
            "guarantee_kind": self.guarantee_kind,
            "influence_bound_2_tau_f_over_n": (
                None
                if self.influence_bound is None
                else round(float(self.influence_bound), 6)
            ),
            "round_auc": [round(float(a), 6) for a in self.round_auc],
        }
        if include_timing:
            out["wall_time_s"] = round(float(self.wall_time_s), 4)
        return out


class Runner:
    """Runs scenarios and returns results.

    Parameters
    ----------
    tail:
        How many trailing rounds ``mean_auc_tail`` averages over.
    verbose:
        Print one line per scenario as it completes.

    Notes
    -----
    The runner reports what it measured on **synthetic** data. It says nothing
    about clinical performance, and a defense that wins here has been shown to
    win here only.
    """

    def __init__(self, *, tail: int = 5, verbose: bool = False):
        if tail < 1:
            raise BenchmarkError(f"tail must be >= 1, got {tail}")
        self.tail = int(tail)
        self.verbose = bool(verbose)

    # -- wiring -------------------------------------------------------------

    def _cohort(self, scenario: Scenario):
        """Build the synthetic cohort for a scenario."""
        return make_cohort(
            CohortSpec(
                n_sites=scenario.n_sites,
                samples_per_site=scenario.samples_per_site,
                test_size=scenario.test_size,
                feature_shift=scenario.feature_shift,
                label_skew=scenario.label_skew,
                seed=scenario.seed,
            )
        )

    def _resolve_attack(self, scenario: Scenario) -> str:
        """Classify the scenario's attack as none/update/data/adaptive."""
        name = scenario.attack
        if name == "none":
            return "none"
        if name == ADAPTIVE:
            return "adaptive"
        if name in ATTACKS:
            return "update"
        if name in DATA_ATTACKS:
            return "data"
        raise BenchmarkError(
            f"unknown attack '{name}'. update-space: {sorted(ATTACKS)}; "
            f"data-space: {sorted(DATA_ATTACKS)}; plus '{ADAPTIVE}'"
        )

    def _build_clients(
        self, scenario: Scenario, sites: Sequence[Any], attestor: Optional[Any]
    ) -> List[Client]:
        """Instantiate clients, arming the last ``n_byzantine`` of them."""
        kind = self._resolve_attack(scenario)
        malicious_ids = {s.site_id for s in sites[len(sites) - scenario.n_byzantine :]}
        if scenario.n_byzantine == 0:
            malicious_ids = set()

        clients: List[Client] = []
        for i, site in enumerate(sites):
            is_mal = site.site_id in malicious_ids
            attack = None
            data_attack = None
            if is_mal and kind == "update":
                attack = get_attack(scenario.attack)
            elif is_mal and kind == "data":
                data_attack = get_data_attack(scenario.attack)
            code = (
                TAMPERED_CODE_IDENTITY
                if (is_mal and scenario.tamper_code)
                else APPROVED_CODE_IDENTITY
            )
            clients.append(
                Client(
                    site.site_id,
                    site.X,
                    site.y,
                    attestor=attestor,
                    code_identity=code,
                    malicious=is_mal,
                    attack=attack,
                    attack_kwargs=dict(scenario.attack_kwargs) if attack else None,
                    data_attack=data_attack,
                    data_attack_kwargs=(
                        dict(scenario.attack_kwargs) if data_attack else None
                    ),
                    seed=scenario.seed * 1000 + i,
                )
            )
        return clients

    def _attestor(self, scenario: Scenario) -> Optional[Any]:
        """Build a mock attestor when the scenario enables attestation."""
        if not scenario.attestation:
            return None
        # Imported here so the benchmark does not hard-depend on the attestation
        # package at import time.
        from trustfed.attestation.attestor import MockSoftwareAttestor, measure_code

        root_key = f"benchmark-root-key-seed-{scenario.seed}".encode("utf-8")
        return MockSoftwareAttestor(root_key, {measure_code(APPROVED_CODE_IDENTITY)})

    # -- execution ----------------------------------------------------------

    def run_one(self, scenario: Scenario) -> ScenarioResult:
        """Execute a single scenario and return its measured result."""
        started = time.perf_counter()
        cohort = self._cohort(scenario)
        attestor = self._attestor(scenario)
        clients = self._build_clients(scenario, cohort.sites, attestor)

        kind = self._resolve_attack(scenario)
        adversary = None
        byz_ids: Sequence[str] = ()
        if kind == "adaptive":
            adversary = AdaptiveAdversary(
                target_aggregator=scenario.aggregator,
                seed=scenario.seed,
                **dict(scenario.attack_kwargs),
            )
            byz_ids = [c.client_id for c in clients if c.malicious]

        server = Server(
            cohort.n_features,
            aggregator=get_aggregator(scenario.aggregator),
            attestor=attestor,
            n_byzantine=scenario.n_byzantine,
            trimmed_beta=recommended_trim_beta(scenario.n_sites, scenario.n_byzantine),
            selector=get_selector(
                scenario.selector,
                fraction=scenario.selection_fraction,
                seed=scenario.seed,
            ),
            adversary=adversary,
            byzantine_ids=byz_ids,
        )
        history = server.fit(
            clients,
            cohort.X_test,
            cohort.y_test,
            rounds=scenario.rounds,
            local_epochs=scenario.local_epochs,
        )

        aucs = [float(r.auc) for r in history if not np.isnan(r.auc)]
        if not aucs:
            raise BenchmarkError(
                f"scenario '{scenario.name}' produced no finite AUC; "
                "the test set may contain a single class"
            )
        tail = aucs[-self.tail :]
        max_f = max_byzantine(scenario.aggregator, scenario.n_sites)
        influence = bounded_influence_bound(
            scenario.aggregator, scenario.n_sites, scenario.n_byzantine
        )
        result = ScenarioResult(
            scenario=scenario,
            final_auc=aucs[-1],
            final_accuracy=float(history[-1].accuracy),
            best_auc=max(aucs),
            mean_auc_tail=float(np.mean(tail)),
            n_rejected_total=sum(r.n_rejected for r in history),
            rounds_run=len(history),
            tolerance_max_f=max_f,
            within_tolerance=scenario.n_byzantine <= max_f,
            guarantee_kind=(
                "bounded influence" if influence is not None else "exact recovery"
            ),
            influence_bound=influence,
            wall_time_s=time.perf_counter() - started,
            round_auc=aucs,
        )
        if self.verbose:
            print(
                f"{scenario.name:<48} AUC={result.final_auc:.3f} "
                f"({result.wall_time_s * 1000:.0f} ms)"
            )
        return result

    def run(
        self, scenarios: Optional[Sequence[Scenario]] = None
    ) -> "BenchmarkReport":
        """Run a list of scenarios (default: :func:`default_grid`).

        Returns
        -------
        BenchmarkReport
            Collected results, ready to serialize.
        """
        from trustfed.benchmark.report import BenchmarkReport

        todo = list(scenarios) if scenarios is not None else default_grid()
        if not todo:
            raise BenchmarkError("no scenarios to run")
        results = [self.run_one(s) for s in todo]
        return BenchmarkReport(results=results)


def run_default(seed: int = 0, *, verbose: bool = False) -> "BenchmarkReport":
    """Convenience: run the default grid at ``seed`` and return the report."""
    return Runner(verbose=verbose).run(default_grid(seed))


__all__ = ["Runner", "ScenarioResult", "run_default", "TAMPERED_CODE_IDENTITY"]
