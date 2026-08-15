"""Section 6: what tampering with the ledger on disk actually looks like.

Two cases, because they fail differently. Editing a recorded field breaks the
hash chain and is caught by structure alone. Cutting the tail off the file does
*not* break anything structural -- the surviving blocks still link -- so it is
caught only against an anchor. Deleting the anchor as well does not turn the
verdict green; it turns it into an explicit refusal to judge.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from trustfed.ledger import FileLedger


def show_tamper_evidence(path: Path) -> None:
    """Edit one recorded field on disk and show that verification fails."""
    ledger = FileLedger(path)
    before = ledger.verify_chain()
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    payload = record["payload"]
    key = "published_by" if "published_by" in payload else "actor"
    payload[key] = "INST_IMPOSTOR"
    lines[0] = json.dumps(record, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    after = FileLedger(path).verify_chain()
    print(f"before tampering        : ok={before.ok}")
    print(f"after editing block 0   : ok={after.ok} codes={list(after.codes)}")


def show_truncation_evidence(source: Path, work_dir: Path) -> None:
    """Truncate a copy of the chain, with and without its anchor."""
    path = work_dir / "truncated.jsonl"
    shutil.copyfile(source, path)
    anchor = FileLedger(path).checkpoint_path
    shutil.copyfile(source.with_name(source.name + ".head.json"), anchor)

    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        return
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    with_anchor = FileLedger(path).verify_chain()
    print(
        f"after dropping the tail : ok={with_anchor.ok} "
        f"codes={list(with_anchor.codes)}"
    )
    anchor.unlink()
    without = FileLedger(path).verify_chain()
    print(
        f"  and deleting anchor   : ok={without.ok} codes={list(without.codes)} "
        f"anchored={without.anchored}"
    )


__all__ = ["show_tamper_evidence", "show_truncation_evidence"]
