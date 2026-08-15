"""The block record of the append-only hash chain.

A block is ``{index, prev_hash, timestamp, payload, payload_hash, signature}``
plus the key id and algorithm used, so a reader can tell which key to check the
signature against without out-of-band information.

The block hash covers every field except the signature; the signature is taken
over the block hash. Truncating or reordering blocks therefore breaks the
``prev_hash`` linkage, and editing a payload breaks ``payload_hash``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping

from trustfed.ledger.crypto import canonical_json, hash_payload
from trustfed.ledger.errors import LedgerStorageError

#: ``prev_hash`` of the first block in a chain.
GENESIS_PREV_HASH = "0" * 64

_FIELDS = (
    "index",
    "prev_hash",
    "timestamp",
    "payload",
    "payload_hash",
    "signature",
    "key_id",
    "algorithm",
)


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with ``Z`` suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass(frozen=True)
class Block:
    """One immutable entry in the chain.

    Attributes
    ----------
    index:
        Zero-based position; must be contiguous within a chain.
    prev_hash:
        :meth:`block_hash` of the previous block, or :data:`GENESIS_PREV_HASH`.
    timestamp:
        ISO-8601 UTC string. Assumed to come from the appending process's
        clock; it is *not* a trusted timestamp and proves nothing about when
        the event really happened.
    payload:
        Arbitrary JSON-serialisable record.
    payload_hash:
        SHA-256 over :func:`canonical_json` of ``payload``.
    signature:
        Hex signature over :meth:`block_hash`, produced by ``key_id``.
    """

    index: int
    prev_hash: str
    timestamp: str
    payload: Mapping[str, Any]
    payload_hash: str
    signature: str
    key_id: str
    algorithm: str

    def signing_preimage(self) -> bytes:
        """Return the exact bytes the signature is computed over."""
        return self.block_hash().encode("utf-8")

    def block_hash(self) -> str:
        """Return the hex SHA-256 over every field except ``signature``."""
        material = {
            "index": self.index,
            "prev_hash": self.prev_hash,
            "timestamp": self.timestamp,
            "payload_hash": self.payload_hash,
            "key_id": self.key_id,
            "algorithm": self.algorithm,
        }
        return hashlib.sha256(canonical_json(material)).hexdigest()

    def recompute_payload_hash(self) -> str:
        """Return the payload hash implied by the payload actually stored."""
        return hash_payload(self.payload)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict of the block."""
        return {name: getattr(self, name) for name in _FIELDS}

    def to_json_line(self) -> str:
        """Return the one-line JSON encoding used by ``FileLedger``."""
        return canonical_json(self.to_dict()).decode("utf-8")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Block":
        """Rebuild a block from :meth:`to_dict` output.

        Raises
        ------
        LedgerStorageError
            If a required field is missing or has the wrong type.
        """
        missing = [name for name in _FIELDS if name not in data]
        if missing:
            raise LedgerStorageError(f"block is missing fields: {sorted(missing)}")
        if not isinstance(data["index"], int):
            raise LedgerStorageError("block index must be an int")
        if not isinstance(data["payload"], Mapping):
            raise LedgerStorageError("block payload must be a JSON object")
        return cls(
            index=int(data["index"]),
            prev_hash=str(data["prev_hash"]),
            timestamp=str(data["timestamp"]),
            payload=dict(data["payload"]),
            payload_hash=str(data["payload_hash"]),
            signature=str(data["signature"]),
            key_id=str(data["key_id"]),
            algorithm=str(data["algorithm"]),
        )


__all__ = ["Block", "GENESIS_PREV_HASH", "utc_now_iso"]
