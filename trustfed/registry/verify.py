"""Lineage verification: walk a bundle's ancestry and judge every hop.

Split out of :mod:`trustfed.registry.registry` so the walk can be read (and
extended) on its own. ``verify_lineage`` is exposed as the registry method of
the same name; call it through :class:`~trustfed.registry.registry.ModelRegistry`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Sequence, Set

from trustfed.ledger.errors import LedgerError
from trustfed.ledger.verification import ChainVerification
from trustfed.registry.lineage import (
    ISSUE_CONTENT_HASH_MISMATCH,
    ISSUE_CYCLE,
    ISSUE_LEDGER_CHAIN_INVALID,
    ISSUE_LEDGER_CONTENT_MISMATCH,
    ISSUE_MISSING_ATTESTATION,
    ISSUE_MISSING_LEDGER_RECORD,
    ISSUE_MISSING_PARENT,
    ISSUE_SHARED_LEDGER_UNVERIFIED,
    ISSUE_UNVERIFIED_ATTESTATION,
    LineageIssue,
    LineageVerdict,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type checkers
    from trustfed.registry.registry import ModelRegistry

#: Ledger payload ``event`` value for a publication.
#: Defined here (not in ``registry``) so the verifier does not import the registry.
EVENT_MODEL_PUBLISHED = "model_published"


def verify_lineage(registry: "ModelRegistry", bundle_id: str) -> LineageVerdict:
    """Walk the full attested chain from ``bundle_id`` to its root(s).

    Checks, for the bundle and every ancestor: the content still hashes to
    its id, a pipeline attestation is present and verified, a matching
    publish record exists on the ledger, and no parent is missing. Also
    verifies the ledger chain itself, and detects derivation cycles.

    A registry frequently shares one :class:`~trustfed.ledger.backend.LedgerBackend`
    with the certification authority and the credit ledger, so a failing chain
    does not necessarily implicate *this* bundle. The failure is therefore
    attributed: blocks that carry a publish record for a bundle in this
    ancestry raise the fatal :data:`ISSUE_LEDGER_CHAIN_INVALID`, while damage
    confined to blocks belonging to other components raises the non-fatal
    :data:`ISSUE_SHARED_LEDGER_UNVERIFIED`, which does not by itself fail the
    verdict. ``ledger_verified`` still reports the chain's own verdict either
    way, so a caller can see that something is wrong with the log even when
    this lineage is clean.

    Returns a structured :class:`LineageVerdict`; it never raises for a
    *failed* lineage, only for an unknown bundle id.
    """
    registry.get(bundle_id)
    issues: List[LineageIssue] = []

    chain_result = registry.ledger.verify_chain()
    order, depth, missing = registry.walk_ancestry(bundle_id)
    if not chain_result.ok:
        issues.extend(_ledger_issues(registry, bundle_id, chain_result, set(order)))
    for absent in missing:
        issues.append(
            LineageIssue(
                ISSUE_MISSING_PARENT, absent, "declared parent is not registered"
            )
        )
    if _has_cycle(registry, order):
        issues.append(
            LineageIssue(ISSUE_CYCLE, bundle_id, "derivation graph contains a cycle")
        )

    ledger_by_bundle = {
        b.payload.get("bundle_id"): b
        for b in registry.ledger
        if b.payload.get("event") == EVENT_MODEL_PUBLISHED
    }

    for node in order:
        bundle = registry.get(node)
        if not bundle.id_is_intact():
            issues.append(
                LineageIssue(
                    ISSUE_CONTENT_HASH_MISMATCH,
                    node,
                    "bundle content does not hash to its recorded id",
                )
            )
        attestation = bundle.attestation
        if not attestation.measurement:
            issues.append(
                LineageIssue(
                    ISSUE_MISSING_ATTESTATION, node, "no pipeline attestation"
                )
            )
        elif not attestation.verified:
            issues.append(
                LineageIssue(
                    ISSUE_UNVERIFIED_ATTESTATION,
                    node,
                    f"attestation not verified ({attestation.verifier_reason})",
                )
            )
        block = ledger_by_bundle.get(node)
        if block is None:
            issues.append(
                LineageIssue(
                    ISSUE_MISSING_LEDGER_RECORD, node, "no publish event on ledger"
                )
            )
        elif block.payload.get("bundle") != bundle.to_dict():
            issues.append(
                LineageIssue(
                    ISSUE_LEDGER_CONTENT_MISMATCH,
                    node,
                    "registry content differs from the ledger record",
                )
            )

    roots = tuple(
        n for n in order if n in registry and not registry.get(n).parents
    )
    fatal = [i for i in issues if i.code != ISSUE_SHARED_LEDGER_UNVERIFIED]
    return LineageVerdict(
        bundle_id=bundle_id,
        ok=not fatal,
        ancestry=tuple(order),
        roots=roots,
        depth=max(depth.values()) if depth else 0,
        ledger_verified=chain_result.ok,
        issues=tuple(issues),
    )


def _ledger_issues(
    registry: "ModelRegistry",
    bundle_id: str,
    chain_result: "ChainVerification",
    ancestry: Set[str],
) -> List[LineageIssue]:
    """Attribute a failing chain to this ancestry, or to a shared co-tenant.

    Walks the blocks the chain verifier complained about and asks, for each,
    whether it is a publish record for a bundle in ``ancestry``. Blocks whose
    index the verifier could not attribute (``index is None``, e.g. a malformed
    checkpoint or an unparsable line) are treated as implicating this lineage,
    because there is no basis for excusing them.
    """
    implicated: List[int] = []
    unattributed: List[str] = []
    ours = False
    for issue in chain_result.issues:
        if issue.index is None:
            unattributed.append(issue.code)
            ours = True
            continue
        implicated.append(issue.index)
        try:
            block = registry.ledger.get(issue.index)
        except LedgerError:
            ours = True
            continue
        if (
            block.payload.get("event") == EVENT_MODEL_PUBLISHED
            and block.payload.get("bundle_id") in ancestry
        ):
            ours = True

    detail = (
        f"ledger chain failed verification: codes={list(chain_result.codes)} "
        f"blocks={implicated or 'unattributed'}"
    )
    if unattributed:
        detail += f" unattributed={unattributed}"
    if ours:
        return [LineageIssue(ISSUE_LEDGER_CHAIN_INVALID, bundle_id, detail)]
    return [
        LineageIssue(
            ISSUE_SHARED_LEDGER_UNVERIFIED,
            bundle_id,
            detail
            + "; no offending block is a publish record for this ancestry, so "
            "this bundle's own lineage is unaffected -- but the shared log "
            "needs attention",
        )
    ]


def _has_cycle(registry: "ModelRegistry", nodes: Sequence[str]) -> bool:
    """Return ``True`` iff the ancestry sub-graph over ``nodes`` has a cycle."""
    state: Dict[str, int] = {}

    def visit(node: str) -> bool:
        if state.get(node) == 1:
            return True
        if state.get(node) == 2:
            return False
        state[node] = 1
        if node in registry:
            for parent in registry.get(node).parents:
                if parent in registry and visit(parent):
                    return True
        state[node] = 2
        return False

    return any(visit(n) for n in nodes)


__all__ = ["EVENT_MODEL_PUBLISHED", "verify_lineage"]
