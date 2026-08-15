"""File-backed ledger: one JSON object per line (JSONL), append-only.

The file on disk is authoritative. :meth:`FileLedger.verify_chain` drops any
cached parse and re-reads the bytes on disk every time it is called, so
tampering performed outside this process is visible without restarting -- even
when the tamperer preserved the file's size and mtime. Ordinary reads (indexing,
iteration, ``len``) use a stat-keyed cache and re-read when the file looks
changed; that is a performance shortcut and not a tamper check.

A sidecar checkpoint file (``<ledger>.head.json``) records the chain length and
head hash after each append. A chain with its tail removed is internally
consistent, so length and head have to be anchored against something; the
sidecar is that anchor when no external one is supplied.

SECURITY NOTE: the sidecar is a *convenience* anchor, not a security boundary.
It sits next to the ledger, so anyone who can truncate the JSONL can also delete
or rewrite the sidecar. Deleting it does not make verification pass silently:
:meth:`FileLedger.verify_chain` reports ``missing_checkpoint`` and returns
``ok=False`` when it is asked to use the sidecar and there is none, because
without an anchor it has nothing to compare a truncated chain against. What the
sidecar genuinely defends against is accidental truncation and a tamperer who
edits only the JSONL. It does not defend against a full-directory rewrite. For
that, publish :meth:`checkpoint` output somewhere the ledger's writer does not
control and pass it back in as ``expected``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple

from trustfed.ledger.backend import (
    ChainIssue,
    ChainVerification,
    Checkpoint,
    LedgerBackend,
)
from trustfed.ledger.block import GENESIS_PREV_HASH, Block
from trustfed.ledger.errors import BlockNotFoundError, LedgerStorageError

#: Issue code: the sidecar anchor is absent, so truncation cannot be ruled out.
ISSUE_MISSING_CHECKPOINT = "missing_checkpoint"
#: Issue code: the sidecar exists but could not be parsed.
ISSUE_MALFORMED_CHECKPOINT = "malformed_checkpoint"
#: Issue code: a line of the JSONL file is not a decodable block.
ISSUE_MALFORMED_BLOCK = "malformed_block"


class FileLedger(LedgerBackend):
    """Append-only hash chain persisted as JSONL.

    Parameters
    ----------
    path:
        Path to the JSONL file. Created (with parents) if it does not exist.
    write_checkpoint:
        Write/refresh the ``.head.json`` sidecar on every append. Default True.
        With this off there is no local anchor, so :meth:`verify_chain` reports
        ``missing_checkpoint`` unless the caller supplies ``expected`` or asks
        for an explicitly unanchored check.

    The whole chain is parsed into memory on read; this reference
    implementation targets thousands of blocks, not millions.
    """

    def __init__(
        self,
        path: Path,
        *,
        write_checkpoint: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not isinstance(path, Path):
            raise LedgerStorageError("path must be a pathlib.Path")
        self._path = path
        self._checkpoint_path = path.with_name(path.name + ".head.json")
        self._write_checkpoint = write_checkpoint
        self._cache: Optional[List[Block]] = None
        self._cache_token: Optional[Tuple[int, int, int, int]] = None
        self._path.parent.mkdir(parents=True, exist_ok=True)
        created = not self._path.exists()
        if created:
            self._path.write_text("", encoding="utf-8")
        self._blocks()
        # Anchor a *newly created* chain at length 0 so that an empty ledger is
        # anchored rather than merely unverified. Never do this for a file that
        # already existed: re-anchoring there would bless whatever is on disk,
        # which is precisely the truncation an anchor is supposed to catch.
        if created and self._write_checkpoint and not self._checkpoint_path.exists():
            self._persist_checkpoint(self.checkpoint())

    # ---------------------------------------------------------------- properties

    @property
    def path(self) -> Path:
        """Path of the JSONL chain file."""
        return self._path

    @property
    def checkpoint_path(self) -> Path:
        """Path of the sidecar checkpoint file."""
        return self._checkpoint_path

    # ------------------------------------------------------------------- storage

    def _stat_token(self) -> Tuple[int, int, int, int]:
        """Return the cheap change-detection token for the chain file.

        Size and mtime alone are not enough: an attacker with write access to
        the JSONL file can make a same-length edit and restore the original
        mtime with :func:`os.utime`. The inode and the inode-change time are
        included so a rewritten, renamed or replaced file is caught as well.

        SECURITY NOTE: this token is still only a *cache* key, and an attacker
        who controls the filesystem can in principle forge all four fields.
        :meth:`verify_chain` therefore does not rely on it at all -- it drops
        the cache and re-reads the file unconditionally.
        """
        try:
            st = self._path.stat()
        except FileNotFoundError as exc:
            raise LedgerStorageError(f"ledger file disappeared: {self._path}") from exc
        return (st.st_size, st.st_mtime_ns, st.st_ino, st.st_ctime_ns)

    def _invalidate_cache(self) -> None:
        """Drop the parsed-chain cache so the next read re-reads the file."""
        self._cache = None
        self._cache_token = None

    def _blocks(self) -> List[Block]:
        """Return the parsed chain, re-reading the file if it changed on disk."""
        token = self._stat_token()
        if self._cache is not None and token == self._cache_token:
            return self._cache
        blocks: List[Block] = []
        with self._path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise LedgerStorageError(
                        f"{self._path}: line {lineno} is not valid JSON: {exc}"
                    ) from exc
                blocks.append(Block.from_dict(data))
        self._cache = blocks
        self._cache_token = token
        self._payload_index = {}
        for block in blocks:
            self._index_payload(block)
        return blocks

    def _append_block(self, block: Block) -> None:
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(block.to_json_line() + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        if self._cache is not None:
            self._cache.append(block)
            self._cache_token = self._stat_token()
        if self._write_checkpoint:
            self._persist_checkpoint(
                Checkpoint(length=len(self._cache or self._blocks()),
                           head_hash=block.block_hash())
            )

    def _persist_checkpoint(self, checkpoint: Checkpoint) -> None:
        self._checkpoint_path.write_text(
            json.dumps(checkpoint.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _get_block(self, index: int) -> Block:
        blocks = self._blocks()
        try:
            return blocks[index]
        except IndexError as exc:
            raise BlockNotFoundError(f"no block at index {index}") from exc

    def _iter_blocks(self) -> Iterator[Block]:
        return iter(tuple(self._blocks()))

    def __len__(self) -> int:
        return len(self._blocks())

    # -------------------------------------------------------------- checkpoints

    def stored_checkpoint(self) -> Optional[Checkpoint]:
        """Return the sidecar checkpoint, or ``None`` if there is not one."""
        if not self._checkpoint_path.exists():
            return None
        try:
            data = json.loads(self._checkpoint_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise LedgerStorageError(f"malformed checkpoint file: {exc}") from exc
        return Checkpoint.from_dict(data)

    def verify_chain(
        self,
        *,
        expected: Optional[Checkpoint] = None,
        check_signatures: bool = True,
        use_stored_checkpoint: bool = True,
    ) -> ChainVerification:
        """Verify the on-disk chain.

        When ``expected`` is not given and ``use_stored_checkpoint`` is True,
        the sidecar checkpoint is used as the anchor, which makes truncation of
        the JSONL file detectable.

        When the sidecar is *absent* in that mode there is no anchor at all, so
        this method refuses to return a passing verdict: it reports
        ``missing_checkpoint`` and ``ok=False``. Deleting the sidecar is exactly
        the move that hides a truncation, and silently downgrading to an
        unanchored check would tell a caller the chain is intact when nothing
        was compared. Pass ``use_stored_checkpoint=False`` to ask for an
        explicitly unanchored structural check instead, or supply ``expected``.

        A malformed or unparsable line makes verification fail with the
        ``malformed_block`` code rather than raising.

        The in-memory cache is dropped before anything is read, so this method
        always parses the bytes currently on disk. Ordinary reads (:meth:`get`,
        iteration, ``len``) re-read only when the stat token changed, which an
        attacker who can write the file can defeat with a same-length edit and
        :func:`os.utime`; verification is the one path where that shortcut
        would hide exactly what it exists to detect.
        """
        self._invalidate_cache()
        if expected is None and use_stored_checkpoint:
            try:
                expected = self.stored_checkpoint()
            except LedgerStorageError as exc:
                return ChainVerification(
                    ok=False,
                    n_blocks=0,
                    head_hash=GENESIS_PREV_HASH,
                    signatures_checked=False,
                    issues=(ChainIssue(ISSUE_MALFORMED_CHECKPOINT, None, str(exc)),),
                    anchored=False,
                )
            if expected is None:
                return self._unanchored_failure(check_signatures)
        try:
            return super().verify_chain(
                expected=expected, check_signatures=check_signatures
            )
        except LedgerStorageError as exc:
            return ChainVerification(
                ok=False,
                n_blocks=0,
                head_hash=GENESIS_PREV_HASH,
                signatures_checked=False,
                issues=(ChainIssue(ISSUE_MALFORMED_BLOCK, None, str(exc)),),
                anchored=False,
            )

    def _unanchored_failure(self, check_signatures: bool) -> ChainVerification:
        """Return a failing verdict for a chain with no anchor to compare to.

        The structural checks still run, so their issues are reported too; the
        ``missing_checkpoint`` issue is appended first so a caller reading only
        ``codes[0]`` sees the reason the verdict cannot be trusted.
        """
        try:
            structural = super().verify_chain(
                expected=None, check_signatures=check_signatures
            )
        except LedgerStorageError as exc:
            return ChainVerification(
                ok=False,
                n_blocks=0,
                head_hash=GENESIS_PREV_HASH,
                signatures_checked=False,
                issues=(ChainIssue(ISSUE_MALFORMED_BLOCK, None, str(exc)),),
                anchored=False,
            )
        issue = ChainIssue(
            ISSUE_MISSING_CHECKPOINT,
            None,
            f"no {self._checkpoint_path.name} anchor beside {self._path}; "
            "truncation cannot be detected, so this chain is unverified",
        )
        return ChainVerification(
            ok=False,
            n_blocks=structural.n_blocks,
            head_hash=structural.head_hash,
            signatures_checked=structural.signatures_checked,
            issues=(issue,) + structural.issues,
            anchored=False,
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"FileLedger(path={str(self._path)!r}, n_blocks={len(self)})"


__all__ = [
    "ISSUE_MALFORMED_BLOCK",
    "ISSUE_MALFORMED_CHECKPOINT",
    "ISSUE_MISSING_CHECKPOINT",
    "FileLedger",
]
