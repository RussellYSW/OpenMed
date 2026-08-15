"""Tests for the append-only hash chain (Component 2)."""

from __future__ import annotations

import pytest

from trustfed.ledger import (
    Block,
    BlockNotFoundError,
    ChainVerificationError,
    Checkpoint,
    HmacSigner,
    InMemoryLedger,
    LedgerError,
    hash_payload,
)
from trustfed.ledger.block import GENESIS_PREV_HASH


def _clock():
    """Deterministic monotone clock so block bytes are reproducible."""
    state = {"t": 0}

    def tick() -> str:
        state["t"] += 1
        return f"2026-01-01T00:00:{state['t']:02d}.000000Z"

    return tick


def make_memory_ledger() -> InMemoryLedger:
    return InMemoryLedger(signer=HmacSigner(b"test-key"), clock=_clock())

# --------------------------------------------------------------------- basics


def test_append_links_blocks_and_sets_genesis():
    led = make_memory_ledger()
    assert led.head() is None
    assert led.head_hash() == GENESIS_PREV_HASH

    b0 = led.append({"event": "publish", "n": 1})
    b1 = led.append({"event": "certify", "n": 2})

    assert (b0.index, b1.index) == (0, 1)
    assert b0.prev_hash == GENESIS_PREV_HASH
    assert b1.prev_hash == b0.block_hash()
    assert led.head() == b1
    assert len(led) == 2


def test_empty_ledger_is_truthy():
    """``ledger or fallback()`` must not silently discard an empty ledger."""
    assert bool(make_memory_ledger()) is True
    assert len(make_memory_ledger()) == 0


def test_payload_hash_matches_canonical_hash():
    led = make_memory_ledger()
    block = led.append({"b": 2, "a": 1})
    assert block.payload_hash == hash_payload({"a": 1, "b": 2})
    assert block.recompute_payload_hash() == block.payload_hash


def test_lookup_by_index_and_payload_hash():
    led = make_memory_ledger()
    led.append({"event": "a"})
    target = led.append({"event": "b"})
    led.append({"event": "c"})

    assert led.get(1) == target
    assert led[1] == target
    assert led[-1].payload["event"] == "c"
    assert led.find_by_payload_hash(target.payload_hash) == target
    assert led.find_payload({"event": "b"}) == target
    assert [b.payload["event"] for b in led] == ["a", "b", "c"]
    assert led.filter(event="c")[0].payload["event"] == "c"


def test_duplicate_payloads_are_all_retrievable():
    led = make_memory_ledger()
    led.append({"event": "dup"})
    led.append({"event": "dup"})
    matches = led.find_all_by_payload_hash(hash_payload({"event": "dup"}))
    assert [b.index for b in matches] == [0, 1]


def test_missing_lookups_raise_typed_errors():
    led = make_memory_ledger()
    led.append({"event": "a"})
    with pytest.raises(BlockNotFoundError):
        led.get(5)
    with pytest.raises(BlockNotFoundError):
        led.find_by_payload_hash("00" * 32)
    with pytest.raises(LedgerError):
        led.append("not-a-mapping")  # type: ignore[arg-type]


def test_clean_chain_verifies():
    led = make_memory_ledger()
    for i in range(5):
        led.append({"i": i})
    result = led.verify_chain()
    assert result.ok is True
    assert result.n_blocks == 5
    assert result.issues == ()
    result.raise_if_invalid()
    assert result.to_dict()["ok"] is True


# ------------------------------------------------------------ adversarial paths


def test_in_memory_payload_mutation_is_detected():
    led = make_memory_ledger()
    led.append({"amount": 1})
    led.append({"amount": 2})
    original = led.get(0)
    forged = Block(
        index=original.index,
        prev_hash=original.prev_hash,
        timestamp=original.timestamp,
        payload={"amount": 999},
        payload_hash=original.payload_hash,
        signature=original.signature,
        key_id=original.key_id,
        algorithm=original.algorithm,
    )
    led._blocks[0] = forged  # simulate an attacker with memory access

    result = led.verify_chain()
    assert result.ok is False
    assert "payload_mutated" in result.codes
    with pytest.raises(ChainVerificationError):
        result.raise_if_invalid()


def test_rehashed_mutation_still_breaks_the_link():
    """Fixing payload_hash to match a forged payload breaks prev_hash instead."""
    led = make_memory_ledger()
    led.append({"amount": 1})
    led.append({"amount": 2})
    original = led.get(0)
    forged_payload = {"amount": 999}
    led._blocks[0] = Block(
        index=original.index,
        prev_hash=original.prev_hash,
        timestamp=original.timestamp,
        payload=forged_payload,
        payload_hash=hash_payload(forged_payload),
        signature=original.signature,
        key_id=original.key_id,
        algorithm=original.algorithm,
    )
    result = led.verify_chain()
    assert result.ok is False
    assert "broken_link" in result.codes


def test_signature_forgery_is_detected():
    led = make_memory_ledger()
    led.append({"x": 1})
    original = led.get(0)
    led._blocks[0] = Block(
        index=0,
        prev_hash=original.prev_hash,
        timestamp=original.timestamp,
        payload=original.payload,
        payload_hash=original.payload_hash,
        signature="ab" * 32,
        key_id=original.key_id,
        algorithm=original.algorithm,
    )
    result = led.verify_chain()
    assert result.ok is False
    assert "bad_signature" in result.codes


def test_chain_written_by_another_key_is_rejected():
    writer = InMemoryLedger(signer=HmacSigner(b"attacker-key"), clock=_clock())
    writer.append({"x": 1})
    auditor_view = InMemoryLedger(signer=HmacSigner(b"test-key"))
    auditor_view._blocks = list(writer)
    result = auditor_view.verify_chain()
    assert result.ok is False
    assert "unknown_key" in result.codes


def test_truncation_is_detected_against_a_checkpoint():
    led = make_memory_ledger()
    for i in range(4):
        led.append({"i": i})
    anchor = led.checkpoint()
    del led._blocks[-1]  # attacker drops the last block

    assert led.verify_chain().ok is True  # internally consistent...
    result = led.verify_chain(expected=anchor)  # ...but not against the anchor
    assert result.ok is False
    assert "truncated" in result.codes


def test_checkpoint_head_mismatch_is_detected():
    led = make_memory_ledger()
    led.append({"i": 0})
    stale = Checkpoint(length=1, head_hash="ff" * 32)
    result = led.verify_chain(expected=stale)
    assert result.ok is False
    assert "head_mismatch" in result.codes


# ---------------------------------------------------------------------- crypto


def test_ed25519_signer_when_cryptography_is_available():
    pytest.importorskip("cryptography")
    from trustfed.ledger import Ed25519Signer, Ed25519Verifier

    signer = Ed25519Signer.from_seed(b"seed-1")
    led = InMemoryLedger(signer=signer, clock=_clock())
    led.append({"x": 1})
    assert led.verify_chain().ok is True

    # A verify-only party can audit without being able to sign.
    public_only = Ed25519Verifier.from_public_hex(signer.key_id)
    auditor = InMemoryLedger(signer=signer, verifier=public_only)
    auditor._blocks = list(led)
    assert auditor.verify_chain().ok is True
    assert not hasattr(public_only, "sign")

    other = Ed25519Signer.from_seed(b"seed-2")
    assert other.key_id != signer.key_id
    assert other.verify(b"msg", signer.sign(b"msg")) is False


def test_hmac_signer_rejects_empty_key():
    from trustfed.ledger import SignatureError

    with pytest.raises(SignatureError):
        HmacSigner(b"")
