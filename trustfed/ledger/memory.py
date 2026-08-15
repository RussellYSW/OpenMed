"""In-memory ledger backend: the chain lives in a Python list.

Useful for tests, demos, and short-lived processes. Nothing is persisted, so
restarting the process destroys the chain and any tamper evidence with it.
"""

from __future__ import annotations

from typing import Iterator, List

from trustfed.ledger.backend import LedgerBackend
from trustfed.ledger.block import Block
from trustfed.ledger.errors import BlockNotFoundError


class InMemoryLedger(LedgerBackend):
    """Append-only hash chain held in process memory.

    Not durable and not shared between processes. Carries exactly the same
    tamper-evidence properties as :class:`~trustfed.ledger.file.FileLedger`
    while the process lives; it protects against nothing once it exits.
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._blocks: List[Block] = []

    def _append_block(self, block: Block) -> None:
        self._blocks.append(block)

    def _get_block(self, index: int) -> Block:
        try:
            return self._blocks[index]
        except IndexError as exc:
            raise BlockNotFoundError(f"no block at index {index}") from exc

    def _iter_blocks(self) -> Iterator[Block]:
        return iter(tuple(self._blocks))

    def __len__(self) -> int:
        return len(self._blocks)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"InMemoryLedger(n_blocks={len(self._blocks)})"


__all__ = ["InMemoryLedger"]
