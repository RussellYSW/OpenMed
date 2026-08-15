"""Append-only hash-chain backend: the ABC and the shared verification logic.

The chain is local and dependency-free -- it is *not* a distributed ledger and
there is no consensus. Its guarantee is tamper *evidence* for a reader who
knows the chain's head: any edit to a payload, any reordering, and any removal
of blocks changes the head hash. A party with write access to the storage and
the signing key can rewrite the whole chain; the defence against that is
publishing the head hash somewhere the rewriter does not control (a witness,
another site's ledger, a commit hash). ``LedgerBackend`` exists so a Hyperledger
Fabric or cloud-KMS backend can be dropped in later without touching callers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple

from trustfed.ledger.block import GENESIS_PREV_HASH, Block, utc_now_iso
from trustfed.ledger.crypto import Signer, Verifier, default_signer, hash_payload
from trustfed.ledger.errors import BlockNotFoundError, LedgerError
from trustfed.ledger.verification import ChainIssue, ChainVerification, Checkpoint


class LedgerBackend(ABC):
    """Append-only hash chain over JSON payloads.

    Parameters
    ----------
    signer:
        Key used to sign appended blocks. Defaults to
        :func:`trustfed.ledger.crypto.default_signer` (Ed25519 when
        ``cryptography`` is importable, otherwise the HMAC fallback, which is
        explicitly not a security boundary).
    verifier:
        Key used to check signatures during verification. Defaults to
        ``signer.verifier()``. Pass a verify-only key when auditing a chain
        written by somebody else.
    clock:
        Callable returning the ISO-8601 timestamp for new blocks. Injectable so
        tests and demos are deterministic.

    Subclasses implement storage only: :meth:`_append_block`, :meth:`_get_block`,
    :meth:`_iter_blocks` and :meth:`__len__`.
    """

    def __init__(
        self,
        *,
        signer: Optional[Signer] = None,
        verifier: Optional[Verifier] = None,
        clock: Optional[Callable[[], str]] = None,
    ) -> None:
        self._signer: Signer = signer if signer is not None else default_signer()
        self._verifier: Verifier = (
            verifier if verifier is not None else self._signer.verifier()
        )
        self._clock: Callable[[], str] = clock if clock is not None else utc_now_iso
        self._payload_index: Dict[str, List[int]] = {}

    # ------------------------------------------------------------------ storage

    @abstractmethod
    def _append_block(self, block: Block) -> None:
        """Persist ``block`` as the new head. Called under the append path."""

    @abstractmethod
    def _get_block(self, index: int) -> Block:
        """Return the block at ``index`` or raise :class:`BlockNotFoundError`."""

    @abstractmethod
    def _iter_blocks(self) -> Iterator[Block]:
        """Yield every block in index order."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of blocks currently stored."""

    def __bool__(self) -> bool:
        """A ledger object is always truthy, even when it holds no blocks.

        Without this, ``ledger or default_ledger()`` would silently discard a
        freshly created empty ledger.
        """
        return True

    # ------------------------------------------------------------------- append

    def append(self, payload: Mapping[str, Any]) -> Block:
        """Sign ``payload`` into a new head block and return it.

        Assumes ``payload`` is JSON-serialisable. The ledger never mutates or
        deletes an existing block; this is the only way to add one.
        """
        if not isinstance(payload, Mapping):
            raise LedgerError("payload must be a mapping")
        payload = dict(payload)
        index = len(self)
        prev_hash = self.head_hash()
        block = Block(
            index=index,
            prev_hash=prev_hash,
            timestamp=self._clock(),
            payload=payload,
            payload_hash=hash_payload(payload),
            signature="",
            key_id=self._signer.key_id,
            algorithm=self._signer.algorithm,
        )
        signature = self._signer.sign(block.signing_preimage())
        signed = Block(
            index=block.index,
            prev_hash=block.prev_hash,
            timestamp=block.timestamp,
            payload=block.payload,
            payload_hash=block.payload_hash,
            signature=signature,
            key_id=block.key_id,
            algorithm=block.algorithm,
        )
        self._append_block(signed)
        self._index_payload(signed)
        return signed

    def _index_payload(self, block: Block) -> None:
        """Record ``block`` in the payload-hash lookup table."""
        self._payload_index.setdefault(block.payload_hash, []).append(block.index)

    def _reindex(self) -> None:
        """Rebuild the payload-hash lookup table from stored blocks."""
        self._payload_index = {}
        for block in self._iter_blocks():
            self._index_payload(block)

    # ------------------------------------------------------------------ reading

    def __iter__(self) -> Iterator[Block]:
        """Iterate blocks in index order."""
        return self._iter_blocks()

    def __getitem__(self, index: int) -> Block:
        """Return the block at ``index`` (supports negative indices)."""
        return self.get(index)

    def get(self, index: int) -> Block:
        """Return the block at ``index``.

        Negative indices count from the head, like a list.

        Raises
        ------
        BlockNotFoundError
            If ``index`` is out of range.
        """
        n = len(self)
        real = index + n if index < 0 else index
        if real < 0 or real >= n:
            raise BlockNotFoundError(f"no block at index {index} (chain length {n})")
        return self._get_block(real)

    def head(self) -> Optional[Block]:
        """Return the most recent block, or ``None`` for an empty chain."""
        if len(self) == 0:
            return None
        return self._get_block(len(self) - 1)

    def head_hash(self) -> str:
        """Return the head block hash, or the genesis constant if empty."""
        head = self.head()
        return GENESIS_PREV_HASH if head is None else head.block_hash()

    def checkpoint(self) -> Checkpoint:
        """Return an anchor for the current chain state.

        Store this somewhere the ledger's writer cannot reach; comparing
        against it later detects truncation and wholesale rewriting.
        """
        return Checkpoint(length=len(self), head_hash=self.head_hash())

    def find_by_payload_hash(self, payload_hash: str) -> Block:
        """Return the *first* block whose payload hash is ``payload_hash``.

        Raises
        ------
        BlockNotFoundError
            If no block carries that payload hash.
        """
        indices = self._payload_index.get(payload_hash)
        if not indices:
            raise BlockNotFoundError(f"no block with payload hash {payload_hash}")
        return self._get_block(indices[0])

    def find_all_by_payload_hash(self, payload_hash: str) -> Tuple[Block, ...]:
        """Return every block with ``payload_hash``, in index order."""
        return tuple(
            self._get_block(i) for i in self._payload_index.get(payload_hash, [])
        )

    def find_payload(self, payload: Mapping[str, Any]) -> Block:
        """Return the first block whose payload equals ``payload``."""
        return self.find_by_payload_hash(hash_payload(dict(payload)))

    def filter(self, **equals: Any) -> Tuple[Block, ...]:
        """Return blocks whose payload matches every ``key=value`` given."""
        return tuple(
            b
            for b in self._iter_blocks()
            if all(b.payload.get(k) == v for k, v in equals.items())
        )

    # ------------------------------------------------------------- verification

    def verify_chain(
        self,
        *,
        expected: Optional[Checkpoint] = None,
        check_signatures: bool = True,
    ) -> ChainVerification:
        """Verify structure, hash linkage, payload hashes and signatures.

        Detects: a mutated payload (``payload_mutated``), a broken or forged
        link (``broken_link``), a missing/reordered block (``index_mismatch``),
        an invalid signature (``bad_signature``), a key or algorithm swap
        (``unknown_key``), and -- when ``expected`` is supplied -- truncation or
        rewriting (``truncated``, ``head_mismatch``, ``length_mismatch``).

        Does **not** detect a consistent rewrite of the whole chain by a party
        holding the signing key and the storage, unless ``expected`` came from
        outside that party's control.

        Without ``expected`` the verdict is *unanchored*: truncation cannot be
        detected, because the remaining blocks still link correctly. The
        returned :attr:`ChainVerification.anchored` flag says which of the two
        questions was actually answered; do not read ``ok=True`` on an
        unanchored verdict as "no blocks are missing".
        """
        issues: List[ChainIssue] = []
        prev_hash = GENESIS_PREV_HASH
        count = 0
        head_hash = GENESIS_PREV_HASH
        signatures_checked = check_signatures

        for position, block in enumerate(self._iter_blocks()):
            count += 1
            if block.index != position:
                issues.append(
                    ChainIssue(
                        "index_mismatch",
                        position,
                        f"block at position {position} declares index {block.index}",
                    )
                )
            if block.prev_hash != prev_hash:
                issues.append(
                    ChainIssue(
                        "broken_link",
                        block.index,
                        f"prev_hash {block.prev_hash[:16]}... does not match the "
                        f"previous block hash {prev_hash[:16]}...",
                    )
                )
            recomputed = block.recompute_payload_hash()
            if recomputed != block.payload_hash:
                issues.append(
                    ChainIssue(
                        "payload_mutated",
                        block.index,
                        "stored payload does not hash to the recorded payload_hash",
                    )
                )
            if check_signatures:
                if block.key_id != self._verifier.key_id:
                    issues.append(
                        ChainIssue(
                            "unknown_key",
                            block.index,
                            f"block signed by {block.key_id[:16]}..., verifier holds "
                            f"{self._verifier.key_id[:16]}...",
                        )
                    )
                elif block.algorithm != self._verifier.algorithm:
                    issues.append(
                        ChainIssue(
                            "unknown_key",
                            block.index,
                            f"algorithm {block.algorithm!r} is not "
                            f"{self._verifier.algorithm!r}",
                        )
                    )
                elif not self._verifier.verify(
                    block.signing_preimage(), block.signature
                ):
                    issues.append(
                        ChainIssue(
                            "bad_signature",
                            block.index,
                            "signature does not verify under the ledger key",
                        )
                    )
            prev_hash = block.block_hash()
            head_hash = prev_hash

        if expected is not None:
            if count < expected.length:
                issues.append(
                    ChainIssue(
                        "truncated",
                        count,
                        f"chain has {count} blocks, checkpoint recorded "
                        f"{expected.length}",
                    )
                )
            elif count > expected.length:
                issues.append(
                    ChainIssue(
                        "length_mismatch",
                        count,
                        f"chain has {count} blocks, checkpoint recorded "
                        f"{expected.length}; re-anchor before trusting it",
                    )
                )
            elif head_hash != expected.head_hash:
                issues.append(
                    ChainIssue(
                        "head_mismatch",
                        count - 1 if count else None,
                        "head hash differs from the checkpointed head",
                    )
                )

        return ChainVerification(
            ok=not issues,
            n_blocks=count,
            head_hash=head_hash,
            signatures_checked=signatures_checked,
            issues=tuple(issues),
            anchored=expected is not None,
        )


__all__ = [
    "ChainIssue",
    "ChainVerification",
    "Checkpoint",
    "LedgerBackend",
]
