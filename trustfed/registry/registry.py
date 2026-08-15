"""The model registry: publish bundles, accumulate lineage, verify ancestry.

Every publish is written to a :class:`~trustfed.ledger.backend.LedgerBackend`,
so the registry's history is tamper-evident and can be replayed
(:meth:`ModelRegistry.rebuild_from_ledger`). Publishing a derived model links it
to its parents, which is what makes the registry a lineage *graph* rather than a
list of releases.

The registry stores metadata only. It never stores weights, never fetches a
URI, and cannot tell you that a model is safe -- it tells you what was claimed,
by whom, on top of what, and whether that record has been altered since.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from trustfed.ledger.backend import LedgerBackend
from trustfed.ledger.block import Block
from trustfed.ledger.memory import InMemoryLedger
from trustfed.registry.bundle import (
    EvaluationReport,
    ModelBundle,
    PipelineAttestation,
    WeightsRef,
)
from trustfed.registry.model_card import ModelCard
from trustfed.registry.errors import (
    AttestationRequiredError,
    BundleNotFoundError,
    DuplicateBundleError,
    LineageError,
    ValidationError,
)
from trustfed.registry.lineage import LineageGraph, LineageVerdict
from trustfed.registry.verify import EVENT_MODEL_PUBLISHED, verify_lineage


class ModelRegistry:
    """Content-addressed registry of model bundles with a lineage graph.

    Parameters
    ----------
    ledger:
        Where publish events are recorded. Defaults to a fresh
        :class:`~trustfed.ledger.memory.InMemoryLedger`, which is not durable.
    require_verified_attestation:
        Refuse to publish a bundle whose pipeline attestation did not verify.
        Default True; turning it off is only appropriate for importing history.
    require_complete_model_card:
        Refuse to publish a bundle whose model card is structurally incomplete.
        Default True.
    """

    def __init__(
        self,
        ledger: Optional[LedgerBackend] = None,
        *,
        require_verified_attestation: bool = True,
        require_complete_model_card: bool = True,
    ) -> None:
        self._ledger: LedgerBackend = ledger if ledger is not None else InMemoryLedger()
        self._require_attestation = require_verified_attestation
        self._require_card = require_complete_model_card
        self._bundles: Dict[str, ModelBundle] = {}
        self._children: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------ properties

    @property
    def ledger(self) -> LedgerBackend:
        """The ledger this registry writes publish events to."""
        return self._ledger

    def __len__(self) -> int:
        """Return the number of bundles registered."""
        return len(self._bundles)

    def __iter__(self) -> Iterator[ModelBundle]:
        """Iterate bundles in publication order."""
        return iter(list(self._bundles.values()))

    def __contains__(self, bundle_id: object) -> bool:
        """Return ``True`` iff ``bundle_id`` is registered."""
        return bundle_id in self._bundles

    # -------------------------------------------------------------------- publish

    def publish(
        self,
        *,
        name: str,
        version: str,
        weights: WeightsRef,
        model_card: ModelCard,
        evaluation: EvaluationReport,
        attestation: PipelineAttestation,
        published_by: str,
        parents: Sequence[str] = (),
        fine_tuning_manual: Optional[Mapping[str, Any]] = None,
        tags: Sequence[str] = (),
    ) -> ModelBundle:
        """Register a new bundle and record the event on the ledger.

        Raises
        ------
        AttestationRequiredError
            If the pipeline attestation did not verify and the registry
            requires verified attestations.
        ValidationError
            If the model card is incomplete (when required).
        BundleNotFoundError
            If a declared parent is not registered.
        DuplicateBundleError
            If byte-identical content was already published.
        """
        for parent in parents:
            if parent not in self._bundles:
                raise BundleNotFoundError(f"parent bundle not registered: {parent}")
        if self._require_attestation and not attestation.verified:
            raise AttestationRequiredError(
                f"pipeline attestation did not verify "
                f"({attestation.verifier_reason or 'unchecked'}); refusing to publish"
            )
        if self._require_card:
            model_card.validate().raise_if_invalid()

        bundle = ModelBundle.create(
            name=name,
            version=version,
            weights=weights,
            model_card=model_card,
            evaluation=evaluation,
            attestation=attestation,
            published_by=published_by,
            parents=tuple(parents),
            fine_tuning_manual=fine_tuning_manual,
            tags=tuple(tags),
        )
        if bundle.bundle_id in self._bundles:
            raise DuplicateBundleError(
                f"identical content is already registered as {bundle.bundle_id}"
            )
        self._record(bundle)
        return bundle

    def publish_derived(
        self, parent_id: str, **kwargs: Any
    ) -> ModelBundle:
        """Publish a bundle derived from ``parent_id``.

        The parent is prepended to any ``parents`` passed explicitly, so the
        lineage edge cannot be forgotten by the caller.
        """
        if parent_id not in self._bundles:
            raise BundleNotFoundError(f"parent bundle not registered: {parent_id}")
        extra = tuple(kwargs.pop("parents", ()))
        parents = (parent_id,) + tuple(p for p in extra if p != parent_id)
        return self.publish(parents=parents, **kwargs)

    def _record(self, bundle: ModelBundle) -> None:
        """Store the bundle, update the graph, and append the ledger event."""
        self._ledger.append(
            {
                "event": EVENT_MODEL_PUBLISHED,
                "bundle_id": bundle.bundle_id,
                "name": bundle.name,
                "version": bundle.version,
                "published_by": bundle.published_by,
                "parents": list(bundle.parents),
                "measurement": bundle.attestation.measurement,
                "bundle": bundle.to_dict(),
            }
        )
        self._bundles[bundle.bundle_id] = bundle
        self._children.setdefault(bundle.bundle_id, [])
        for parent in bundle.parents:
            self._children.setdefault(parent, []).append(bundle.bundle_id)

    @classmethod
    def rebuild_from_ledger(cls, ledger: LedgerBackend, **kwargs: Any) -> "ModelRegistry":
        """Replay publish events from ``ledger`` into a fresh registry view.

        The rebuilt registry shares the same ledger object; replaying does not
        append new events. Bundles whose recorded id does not match their
        recorded content are still loaded, so that
        :meth:`verify_lineage` can report the mismatch instead of hiding it.
        """
        registry = cls(ledger, **kwargs)
        for block in ledger:
            if block.payload.get("event") != EVENT_MODEL_PUBLISHED:
                continue
            data = block.payload.get("bundle")
            if not isinstance(data, Mapping):
                raise ValidationError(
                    f"publish event at index {block.index} carries no bundle"
                )
            bundle = ModelBundle.from_dict(data)
            registry._bundles[bundle.bundle_id] = bundle
            registry._children.setdefault(bundle.bundle_id, [])
            for parent in bundle.parents:
                registry._children.setdefault(parent, []).append(bundle.bundle_id)
        return registry

    # --------------------------------------------------------------------- lookup

    def get(self, bundle_id: str) -> ModelBundle:
        """Return the bundle with ``bundle_id``.

        Raises
        ------
        BundleNotFoundError
            If the id is unknown.
        """
        try:
            return self._bundles[bundle_id]
        except KeyError as exc:
            raise BundleNotFoundError(f"unknown bundle: {bundle_id}") from exc

    def find(self, *, name: Optional[str] = None, **fields: Any) -> Tuple[ModelBundle, ...]:
        """Return bundles matching ``name`` and any other exact field values."""
        out = []
        for bundle in self._bundles.values():
            if name is not None and bundle.name != name:
                continue
            if all(getattr(bundle, k, None) == v for k, v in fields.items()):
                out.append(bundle)
        return tuple(out)

    def ledger_records(self, bundle_id: str) -> Tuple[Block, ...]:
        """Return every ledger block that mentions ``bundle_id``."""
        return tuple(
            b for b in self._ledger if b.payload.get("bundle_id") == bundle_id
        )

    # --------------------------------------------------------------------- lineage

    def parents(self, bundle_id: str) -> Tuple[str, ...]:
        """Return the direct parents of ``bundle_id``."""
        return self.get(bundle_id).parents

    def children(self, bundle_id: str) -> Tuple[str, ...]:
        """Return the direct children of ``bundle_id``."""
        self.get(bundle_id)
        return tuple(self._children.get(bundle_id, ()))

    def ancestors(self, bundle_id: str) -> Tuple[str, ...]:
        """Return every ancestor of ``bundle_id``, breadth-first, excluding itself."""
        return tuple(self.walk_ancestry(bundle_id)[0][1:])

    def descendants(self, bundle_id: str) -> Tuple[str, ...]:
        """Return every descendant of ``bundle_id``, breadth-first."""
        self.get(bundle_id)
        seen: List[str] = []
        queue = list(self._children.get(bundle_id, ()))
        while queue:
            current = queue.pop(0)
            if current in seen:
                continue
            seen.append(current)
            queue.extend(self._children.get(current, ()))
        return tuple(seen)

    def roots(self) -> Tuple[str, ...]:
        """Return every registered bundle that has no parents."""
        return tuple(b.bundle_id for b in self._bundles.values() if not b.parents)

    def walk_ancestry(self, bundle_id: str) -> Tuple[List[str], Dict[str, int], List[str]]:
        """Breadth-first walk toward the roots.

        Returns the visit order, the depth of each visited node, and the ids of
        parents that are not registered.
        """
        self.get(bundle_id)
        order: List[str] = []
        depth: Dict[str, int] = {bundle_id: 0}
        missing: List[str] = []
        queue = [bundle_id]
        guard = 0
        while queue:
            guard += 1
            if guard > 10_000:
                raise LineageError("lineage walk exceeded 10000 steps")
            current = queue.pop(0)
            if current in order:
                continue
            order.append(current)
            bundle = self._bundles.get(current)
            if bundle is None:
                continue
            for parent in bundle.parents:
                if parent not in self._bundles:
                    if parent not in missing:
                        missing.append(parent)
                    continue
                depth[parent] = max(depth.get(parent, 0), depth[current] + 1)
                queue.append(parent)
        return order, depth, missing

    def lineage_graph(self, bundle_id: Optional[str] = None) -> LineageGraph:
        """Return the whole graph, or the ancestry sub-graph of ``bundle_id``."""
        if bundle_id is None:
            ids = list(self._bundles)
        else:
            ids = list(self.walk_ancestry(bundle_id)[0])
        edges: List[Tuple[str, str]] = []
        for node in ids:
            bundle = self._bundles.get(node)
            if bundle is None:
                continue
            for parent in bundle.parents:
                if parent in ids:
                    edges.append((parent, node))
        labels = {
            n: f"{self._bundles[n].name} v{self._bundles[n].version}"
            for n in ids
            if n in self._bundles
        }
        return LineageGraph(nodes=tuple(ids), edges=tuple(edges), labels=labels)

    def verify_lineage(self, bundle_id: str) -> LineageVerdict:
        """Walk the full attested chain from ``bundle_id`` to its root(s).

        Checks, for the bundle and every ancestor: the content still hashes to
        its id, a pipeline attestation is present and verified, a matching
        publish record exists on the ledger, and no parent is missing. Also
        verifies the ledger chain itself, and detects derivation cycles.

        Returns a structured
        :class:`~trustfed.registry.lineage.LineageVerdict`; it never raises for
        a *failed* lineage, only for an unknown bundle id. The walk itself lives
        in :mod:`trustfed.registry.verify`.
        """
        return verify_lineage(self, bundle_id)


__all__ = ["EVENT_MODEL_PUBLISHED", "ModelRegistry"]
