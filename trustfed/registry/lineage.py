"""Lineage graph structures and the structured verdict of a lineage walk.

Derivation in OpenMed is a directed acyclic graph, not a list: a model may be
fine-tuned from one parent, merged from several, and may itself have many
children. These records describe that graph and the outcome of verifying one
node's full ancestry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Tuple

#: Issue codes emitted by :meth:`trustfed.registry.registry.ModelRegistry.verify_lineage`.
ISSUE_CONTENT_HASH_MISMATCH = "content_hash_mismatch"
ISSUE_MISSING_PARENT = "missing_parent"
ISSUE_MISSING_ATTESTATION = "missing_attestation"
ISSUE_UNVERIFIED_ATTESTATION = "unverified_attestation"
ISSUE_MISSING_LEDGER_RECORD = "missing_ledger_record"
ISSUE_LEDGER_CONTENT_MISMATCH = "ledger_content_mismatch"
ISSUE_LEDGER_CHAIN_INVALID = "ledger_chain_invalid"
#: The shared ledger failed verification, but no offending block belongs to the
#: ancestry under examination. Non-fatal: it does not by itself fail a verdict.
ISSUE_SHARED_LEDGER_UNVERIFIED = "shared_ledger_unverified"
ISSUE_CYCLE = "cycle_detected"


@dataclass(frozen=True)
class LineageIssue:
    """One problem found while walking a lineage."""

    code: str
    bundle_id: str
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {"code": self.code, "bundle_id": self.bundle_id, "detail": self.detail}


@dataclass(frozen=True)
class LineageVerdict:
    """Result of verifying the attested chain from a bundle to its roots.

    Attributes
    ----------
    ok:
        True only when every ancestor exists, hashes to its recorded id, has a
        verified pipeline attestation, and has a matching ledger record. The
        one issue code that does *not* clear this flag is
        :data:`ISSUE_SHARED_LEDGER_UNVERIFIED`, which reports damage elsewhere
        in a ledger shared with other components; check ``ledger_verified`` and
        ``codes`` as well before treating a passing verdict as a clean log.
    ancestry:
        Bundle ids visited, breadth-first from the queried bundle upward.
    roots:
        Ancestors with no parents of their own.
    depth:
        Longest parent-hop distance from the queried bundle to a root.
    ledger_verified:
        Whether the underlying hash chain itself verified.
    """

    bundle_id: str
    ok: bool
    ancestry: Tuple[str, ...] = ()
    roots: Tuple[str, ...] = ()
    depth: int = 0
    ledger_verified: bool = False
    issues: Tuple[LineageIssue, ...] = ()

    @property
    def codes(self) -> Tuple[str, ...]:
        """Return the issue codes in the order they were found."""
        return tuple(issue.code for issue in self.issues)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict of the verdict."""
        return {
            "bundle_id": self.bundle_id,
            "ok": self.ok,
            "ancestry": list(self.ancestry),
            "roots": list(self.roots),
            "depth": self.depth,
            "ledger_verified": self.ledger_verified,
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def summary(self) -> str:
        """Return a one-line human summary of the verdict."""
        if self.ok:
            return (
                f"lineage OK: {len(self.ancestry)} attested bundle(s), depth "
                f"{self.depth}, root(s) {len(self.roots)}"
            )
        return "lineage FAILED: " + "; ".join(
            f"[{i.code}] {i.bundle_id[-12:]} {i.detail}" for i in self.issues
        )


@dataclass(frozen=True)
class LineageGraph:
    """A directed acyclic graph of bundles, edges pointing parent -> child."""

    nodes: Tuple[str, ...] = ()
    edges: Tuple[Tuple[str, str], ...] = ()
    labels: Mapping[str, str] = field(default_factory=dict)

    def parents_of(self, bundle_id: str) -> Tuple[str, ...]:
        """Return the direct parents of ``bundle_id`` within this graph."""
        return tuple(p for p, c in self.edges if c == bundle_id)

    def children_of(self, bundle_id: str) -> Tuple[str, ...]:
        """Return the direct children of ``bundle_id`` within this graph."""
        return tuple(c for p, c in self.edges if p == bundle_id)

    def roots(self) -> Tuple[str, ...]:
        """Return nodes with no parents in this graph."""
        return tuple(n for n in self.nodes if not self.parents_of(n))

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict of the graph."""
        return {
            "nodes": list(self.nodes),
            "edges": [list(e) for e in self.edges],
            "labels": dict(self.labels),
        }

    def to_mermaid(self) -> str:
        """Render the graph as a Mermaid ``graph TD`` block for docs."""
        lines: List[str] = ["graph TD"]
        alias = {node: f"n{i}" for i, node in enumerate(self.nodes)}
        for node in self.nodes:
            label = self.labels.get(node, node[-12:])
            lines.append(f'    {alias[node]}["{label}"]')
        for parent, child in self.edges:
            if parent in alias and child in alias:
                lines.append(f"    {alias[parent]} --> {alias[child]}")
        return "\n".join(lines) + "\n"


def topological_order(
    nodes: Iterable[str], edges: Iterable[Tuple[str, str]]
) -> Tuple[str, ...]:
    """Return a parents-before-children ordering of ``nodes``.

    Nodes that remain in a cycle are appended at the end in input order, so the
    caller can detect the cycle by comparing against its own expectations.
    """
    node_list = list(dict.fromkeys(nodes))
    edge_list = [e for e in edges if e[0] in node_list and e[1] in node_list]
    incoming = {n: sum(1 for p, c in edge_list if c == n) for n in node_list}
    ready = [n for n in node_list if incoming[n] == 0]
    ordered: List[str] = []
    while ready:
        current = ready.pop(0)
        ordered.append(current)
        for parent, child in edge_list:
            if parent == current:
                incoming[child] -= 1
                if incoming[child] == 0:
                    ready.append(child)
    ordered.extend(n for n in node_list if n not in ordered)
    return tuple(ordered)


__all__ = [
    "ISSUE_CONTENT_HASH_MISMATCH",
    "ISSUE_CYCLE",
    "ISSUE_LEDGER_CHAIN_INVALID",
    "ISSUE_LEDGER_CONTENT_MISMATCH",
    "ISSUE_MISSING_ATTESTATION",
    "ISSUE_MISSING_LEDGER_RECORD",
    "ISSUE_MISSING_PARENT",
    "ISSUE_SHARED_LEDGER_UNVERIFIED",
    "ISSUE_UNVERIFIED_ATTESTATION",
    "LineageGraph",
    "LineageIssue",
    "LineageVerdict",
    "topological_order",
]
