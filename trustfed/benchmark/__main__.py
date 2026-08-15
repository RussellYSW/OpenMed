"""Command-line entry point: ``python -m trustfed.benchmark``.

Runs the default grid (or a filtered subset) and writes ``benchmark.json`` and
``benchmark.md``. Kept separate from ``trustfed/cli.py`` so the benchmark can be
run without the rest of the CLI.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

from trustfed.benchmark.runner import Runner
from trustfed.benchmark.scenario import (
    DEFAULT_AGGREGATORS,
    DEFAULT_ATTACKS,
    Scenario,
    grid,
)


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the benchmark CLI."""
    p = argparse.ArgumentParser(
        prog="python -m trustfed.benchmark",
        description=(
            "Run the TrustFed defense benchmark on synthetic cohorts and write "
            "JSON + Markdown results."
        ),
    )
    p.add_argument("--seed", type=int, default=0, help="master seed (default: 0)")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="directory to write benchmark.json / benchmark.md into",
    )
    p.add_argument("--rounds", type=int, default=15, help="federated rounds per run")
    p.add_argument("--sites", type=int, default=8, help="number of sites")
    p.add_argument(
        "--byzantine",
        type=int,
        nargs="+",
        default=[2, 3],
        help="attacker counts to sweep",
    )
    p.add_argument(
        "--aggregators",
        nargs="+",
        default=list(DEFAULT_AGGREGATORS),
        help="aggregation rules to evaluate",
    )
    p.add_argument(
        "--attacks", nargs="+", default=list(DEFAULT_ATTACKS), help="attacks to run"
    )
    p.add_argument(
        "--selector",
        default="all",
        help="client-selection policy: all, random, loss, reputation",
    )
    p.add_argument("--quiet", action="store_true", help="suppress per-scenario lines")
    return p


def build_scenarios(args: argparse.Namespace) -> List[Scenario]:
    """Translate parsed arguments into the scenario grid."""
    return grid(
        aggregators=args.aggregators,
        attacks=args.attacks,
        n_byzantine=tuple(args.byzantine),
        seeds=(args.seed,),
        n_sites=args.sites,
        rounds=args.rounds,
        selector=args.selector,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the benchmark; return a process exit code."""
    args = build_parser().parse_args(argv)
    runner = Runner(verbose=not args.quiet)
    report = runner.run(build_scenarios(args))
    if args.out is not None:
        written = report.write(args.out)
        print(f"wrote {written['json']}")
        print(f"wrote {written['markdown']}")
    else:
        print(report.to_markdown())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
