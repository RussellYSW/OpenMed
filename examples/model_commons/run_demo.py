"""End-to-end model-commons demo: publish -> certify -> derive -> verify -> credit.

Runs the whole trust plane on **synthetic data generated in-process from a seed**.
No patient data, no network calls, no persistent state outside the output
directory (a temporary directory unless ``--out`` is given).

What it shows, in order -- one module per section under ``openmed_demo/``:

1. an attested publish (a fresh challenge nonce, a policy over approved code
   measurements, and a refusal for a site running unapproved code);
2. multi-institution certification, with the owner institution derived from the
   bundle rather than declared, including the refusal of a site trying to
   certify its own model and a threshold that needs two institutions;
3. a derived model that is auto-linked to its certified parent;
4. ``verify_lineage`` walking the attested chain back to the root;
5. credit with counterparty-signed evaluation service, citable release
   attribution, contributor standing, and a reciprocity refusal with a
   machine-readable reason;
6. tamper evidence: editing one byte of the ledger on disk makes verification
   fail, and truncating it without its anchor makes verification refuse to pass.

Nothing here is clinically validated, and the attestor is a software mock.

Usage::

    python examples/model_commons/run_demo.py [--out DIR] [--seed 7]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))
sys.path.insert(0, str(_HERE.parent))

from trustfed.attestation import NonceStore  # noqa: E402

from openmed_demo import (  # noqa: E402
    build,
    certify,
    credit_and_reciprocity,
    derive,
    make_attestor,
    publish_root,
    rule,
    show_lineage,
    show_tamper_evidence,
    show_truncation_evidence,
    show_unapproved_publish,
)


def main(argv: Any = None) -> int:
    """Run the demo end to end."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    parser.add_argument("--seed", type=int, default=7, help="synthetic-data seed")
    args = parser.parse_args(argv)

    tmp = None
    if args.out is None:
        tmp = tempfile.TemporaryDirectory(prefix="openmed-demo-")
        out_dir = Path(tmp.name)
    else:
        out_dir = args.out
        out_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("OpenMed model-commons demo (synthetic data, software-mock attestor)")
        print(f"output directory        : {out_dir}")
        registry, authority, credit, signers = build(out_dir, args.seed)
        attestor = make_attestor(NonceStore())

        rule("1. Attested publish")
        root = publish_root(registry, attestor, args.seed)
        show_unapproved_publish(registry, attestor)

        rule("2. Multi-party certification")
        certify(authority, root)

        rule("3. Derive from the certified base")
        derived = derive(registry, attestor, root.bundle_id, args.seed)

        rule("4. Verify lineage")
        show_lineage(registry, derived.bundle_id)

        rule("5. Credit and reciprocity")
        credit_and_reciprocity(credit, signers, root.bundle_id)

        rule("6. Tamper evidence")
        show_truncation_evidence(out_dir / "certification.jsonl", out_dir)
        show_tamper_evidence(out_dir / "registry.jsonl")

        print()
        print("All records are synthetic. The attestor is a software mock and is")
        print("not a security boundary; see trustfed/attestation/attestor.py.")
    finally:
        if tmp is not None:
            tmp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
