"""Component 2 -- tamper-evident lineage: a local append-only hash chain.

Every governance-relevant event in OpenMed (a model published, a review signed,
a certification granted or revoked, a credit entry) is written as a block whose
hash commits to the previous block. Re-reading the chain therefore reveals any
later edit, reordering, or deletion.

This is deliberately **not** a distributed ledger: there is no consensus, no
network, and no external dependency. :class:`LedgerBackend` is the seam where a
Hyperledger Fabric or cloud-KMS backend can be added without changing callers.
:class:`InMemoryLedger` and :class:`FileLedger` ship today.

SECURITY NOTE: a hash chain gives tamper *evidence*, not tamper *prevention*.
Anyone holding both the storage and the signing key can rewrite history
consistently. The defence is to publish :meth:`LedgerBackend.checkpoint` output
where the writer cannot reach it. When ``cryptography`` is unavailable the
signature falls back to HMAC, which is symmetric and therefore not a security
boundary at all -- see :mod:`trustfed.ledger.crypto`.

SCOPE NOTE: *deletion* is only detectable against an anchor. A chain with its
tail cut off still links correctly, so a verification run without a
:class:`Checkpoint` answers "are these blocks internally consistent?" and not
"are any blocks missing?". :attr:`ChainVerification.anchored` says which
question was answered, and :class:`FileLedger` refuses to return a passing
verdict when it was asked for an anchored check and its sidecar is gone.
"""

from __future__ import annotations

from trustfed.ledger.backend import LedgerBackend
from trustfed.ledger.block import GENESIS_PREV_HASH, Block, utc_now_iso
from trustfed.ledger.crypto import (
    HAVE_CRYPTOGRAPHY,
    Ed25519Signer,
    Ed25519Verifier,
    HmacSigner,
    Signer,
    Verifier,
    canonical_json,
    default_signer,
    hash_payload,
)
from trustfed.ledger.errors import (
    BlockNotFoundError,
    ChainVerificationError,
    LedgerError,
    LedgerStorageError,
    SignatureError,
)
from trustfed.ledger.file import (
    ISSUE_MALFORMED_BLOCK,
    ISSUE_MALFORMED_CHECKPOINT,
    ISSUE_MISSING_CHECKPOINT,
    FileLedger,
)
from trustfed.ledger.memory import InMemoryLedger
from trustfed.ledger.verification import ChainIssue, ChainVerification, Checkpoint

__all__ = [
    "GENESIS_PREV_HASH",
    "HAVE_CRYPTOGRAPHY",
    "ISSUE_MALFORMED_BLOCK",
    "ISSUE_MALFORMED_CHECKPOINT",
    "ISSUE_MISSING_CHECKPOINT",
    "Block",
    "BlockNotFoundError",
    "ChainIssue",
    "ChainVerification",
    "ChainVerificationError",
    "Checkpoint",
    "Ed25519Signer",
    "Ed25519Verifier",
    "FileLedger",
    "HmacSigner",
    "InMemoryLedger",
    "LedgerBackend",
    "LedgerError",
    "LedgerStorageError",
    "SignatureError",
    "Signer",
    "Verifier",
    "canonical_json",
    "default_signer",
    "hash_payload",
    "utc_now_iso",
]
