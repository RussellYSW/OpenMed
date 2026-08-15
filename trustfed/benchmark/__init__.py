"""A shared benchmark harness for federated defenses.

Contributed defenses are only comparable if they are measured the same way. This
package provides that common ground: a declarative
:class:`~trustfed.benchmark.scenario.Scenario`, a
:class:`~trustfed.benchmark.runner.Runner` that executes a grid of them
deterministically from a seed, and a
:class:`~trustfed.benchmark.report.BenchmarkReport` that serializes the results
to JSON and Markdown.

Run the default grid::

    python -m trustfed.benchmark --out results/

All cohorts are synthetic (:mod:`trustfed.data`). Results characterize the
simulation, not clinical performance.
"""

from __future__ import annotations

from trustfed.benchmark.errors import BenchmarkError
from trustfed.benchmark.report import (
    PROVENANCE,
    BenchmarkReport,
    compare_reports,
    load_report,
)
from trustfed.benchmark.runner import Runner, ScenarioResult, run_default
from trustfed.benchmark.scenario import (
    DEFAULT_AGGREGATORS,
    DEFAULT_ATTACKS,
    Scenario,
    default_grid,
    grid,
)

__all__ = [
    "Scenario",
    "grid",
    "default_grid",
    "DEFAULT_AGGREGATORS",
    "DEFAULT_ATTACKS",
    "Runner",
    "ScenarioResult",
    "run_default",
    "BenchmarkReport",
    "PROVENANCE",
    "load_report",
    "compare_reports",
    "BenchmarkError",
]
