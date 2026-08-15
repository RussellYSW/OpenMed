"""Tests for the file-backed chain: persistence, tampering and anchoring.

The JSONL ledger's distinctive risk is that a reader trusts a file another
process can rewrite, so the adversarial cases here are on-disk edits,
truncation, and the absence of the anchor that makes truncation visible.
In-memory chain behaviour and the signing primitives live in
``test_ledger.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from trustfed.ledger import FileLedger, HmacSigner, InMemoryLedger, hash_payload


def _clock():
    """Deterministic monotone clock so block bytes are reproducible."""
    state = {"t": 0}

    def tick() -> str:
        state["t"] += 1
        return f"2026-01-01T00:00:{state['t']:02d}.000000Z"

    return tick


def make_memory_ledger() -> InMemoryLedger:
    return InMemoryLedger(signer=HmacSigner(b"test-key"), clock=_clock())

# ----------------------------------------------------------------- file ledger


def test_file_ledger_round_trips_and_persists(tmp_path: Path):
    path = tmp_path / "chain.jsonl"
    led = FileLedger(path, signer=HmacSigner(b"test-key"), clock=_clock())
    led.append({"event": "publish", "bundle": "abc"})
    led.append({"event": "certify", "bundle": "abc"})

    reopened = FileLedger(path, signer=HmacSigner(b"test-key"))
    assert len(reopened) == 2
    assert reopened.get(1).payload["event"] == "certify"
    assert reopened.verify_chain().ok is True
    assert reopened.stored_checkpoint() == led.checkpoint()

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["index"] == 0


def test_mutating_a_block_on_disk_fails_verification(tmp_path: Path):
    """The required adversarial test: edit the JSONL and verification must fail."""
    path = tmp_path / "chain.jsonl"
    led = FileLedger(path, signer=HmacSigner(b"test-key"), clock=_clock())
    led.append({"event": "certify", "model": "m1", "decision": "denied"})
    led.append({"event": "release", "model": "m1"})
    assert led.verify_chain().ok is True

    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["payload"]["decision"] = "approved"  # retroactive edit
    lines[0] = json.dumps(record, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Same live instance re-reads the file: tampering is visible without restart.
    result = led.verify_chain()
    assert result.ok is False
    assert "payload_mutated" in result.codes
    assert result.issues[0].index == 0

    reopened = FileLedger(path, signer=HmacSigner(b"test-key"))
    assert reopened.verify_chain().ok is False


def test_a_stealth_edit_that_preserves_size_and_mtime_is_still_caught(
    tmp_path: Path,
):
    """A tamperer controls the file's metadata as well as its bytes.

    The read cache is keyed on stat output, and an attacker with write access
    can make a same-length substitution and put the original mtime back with
    ``os.utime``. If verification trusted that cache it would report a clean
    chain on tampered bytes, which is the one thing this component exists to
    prevent.
    """
    path = tmp_path / "chain.jsonl"
    led = FileLedger(path, signer=HmacSigner(b"test-key"), clock=_clock())
    led.append({"event": "certify", "actor": "site_a"})
    led.append({"event": "release", "actor": "site_a"})
    assert led.verify_chain().ok is True

    before = path.stat()
    clean_parse = list(led)
    original = path.read_text(encoding="utf-8")
    tampered = original.replace('"actor":"site_a"', '"actor":"site_z"', 1)
    assert tampered != original
    assert len(tampered) == len(original)  # same length: size is unchanged
    path.write_text(tampered, encoding="utf-8")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_size == before.st_size
    assert path.stat().st_mtime_ns == before.st_mtime_ns

    result = led.verify_chain()  # the *same* live object, not a fresh one
    assert result.ok is False
    assert "payload_mutated" in result.codes
    assert result.issues[0].index == 0
    # And the tampered value is what a subsequent read now returns.
    assert led.get(0).payload["actor"] == "site_z"

    # Now the stronger property, stated white-box because it is the one an
    # attacker attacks: even if the cache held the pre-tamper parse under a
    # token that matches the tampered file exactly -- an adversary who forged
    # every stat field -- verify_chain does not consult it. It always re-reads.
    led._cache = clean_parse
    led._cache_token = led._stat_token()
    assert led.get(0).payload["actor"] == "site_a"  # the stale cache is in use
    assert led.verify_chain().ok is False


def test_truncating_the_file_is_detected_via_the_sidecar(tmp_path: Path):
    path = tmp_path / "chain.jsonl"
    led = FileLedger(path, signer=HmacSigner(b"test-key"), clock=_clock())
    for i in range(3):
        led.append({"i": i})
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")

    reopened = FileLedger(path, signer=HmacSigner(b"test-key"))
    result = reopened.verify_chain()
    assert result.ok is False
    assert "truncated" in result.codes
    # Without the anchor the truncated chain looks fine -- that is the point.
    assert reopened.verify_chain(use_stored_checkpoint=False).ok is True


def test_corrupt_line_is_reported_not_swallowed(tmp_path: Path):
    from trustfed.ledger import LedgerStorageError

    path = tmp_path / "chain.jsonl"
    led = FileLedger(path, signer=HmacSigner(b"test-key"), clock=_clock())
    led.append({"i": 0})
    path.write_text("{not json}\n", encoding="utf-8")

    # A live instance turns the parse failure into a failed verdict...
    result = led.verify_chain()
    assert result.ok is False
    assert "malformed_block" in result.codes
    # ...and opening a corrupt file raises a typed storage error.
    with pytest.raises(LedgerStorageError):
        FileLedger(path, signer=HmacSigner(b"test-key"))


def test_appending_after_reopen_continues_the_chain(tmp_path: Path):
    path = tmp_path / "chain.jsonl"
    first = FileLedger(path, signer=HmacSigner(b"k"), clock=_clock())
    b0 = first.append({"i": 0})
    second = FileLedger(path, signer=HmacSigner(b"k"), clock=_clock())
    b1 = second.append({"i": 1})
    assert b1.index == 1
    assert b1.prev_hash == b0.block_hash()
    assert second.verify_chain().ok is True


# ------------------------------------------------- anchoring and its absence


def test_deleting_the_sidecar_does_not_hide_a_truncation(tmp_path: Path):
    """The adversarial case that matters: truncate the chain AND drop the anchor.

    The remaining blocks are internally consistent, so nothing structural
    objects. Verification must therefore refuse to pass rather than report an
    intact chain it had nothing to compare against.
    """
    path = tmp_path / "chain.jsonl"
    led = FileLedger(path, signer=HmacSigner(b"test-key"), clock=_clock())
    for i in range(5):
        led.append({"i": i})
    assert led.verify_chain().ok is True

    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")
    led.checkpoint_path.unlink()

    reopened = FileLedger(path, signer=HmacSigner(b"test-key"))
    result = reopened.verify_chain()
    assert result.ok is False
    assert result.anchored is False
    assert "missing_checkpoint" in result.codes
    assert result.n_blocks == 2
    # Reopening must not silently re-anchor the truncated file either.
    assert not reopened.checkpoint_path.exists()
    assert FileLedger(path, signer=HmacSigner(b"test-key")).verify_chain().ok is False


def test_an_unanchored_check_can_still_be_asked_for_explicitly(tmp_path: Path):
    path = tmp_path / "chain.jsonl"
    led = FileLedger(path, signer=HmacSigner(b"test-key"), clock=_clock())
    led.append({"i": 0})
    led.checkpoint_path.unlink()

    unanchored = led.verify_chain(use_stored_checkpoint=False)
    assert unanchored.ok is True
    assert unanchored.anchored is False  # ...and it says so
    assert led.verify_chain(expected=led.checkpoint()).anchored is True


def test_a_fresh_empty_ledger_is_anchored_at_zero(tmp_path: Path):
    path = tmp_path / "fresh.jsonl"
    led = FileLedger(path, signer=HmacSigner(b"test-key"))
    assert led.checkpoint_path.exists()
    result = led.verify_chain()
    assert (result.ok, result.n_blocks, result.anchored) == (True, 0, True)


def test_verify_chain_without_a_checkpoint_still_reports_structural_damage(
    tmp_path: Path,
):
    path = tmp_path / "chain.jsonl"
    led = FileLedger(path, signer=HmacSigner(b"test-key"), clock=_clock())
    led.append({"decision": "denied"})
    led.append({"i": 1})
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["payload"]["decision"] = "approved"
    lines[0] = json.dumps(record, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    led.checkpoint_path.unlink()

    result = led.verify_chain()
    assert result.ok is False
    assert result.codes[0] == "missing_checkpoint"
    assert "payload_mutated" in result.codes


def test_in_memory_verification_reports_that_it_is_unanchored():
    led = make_memory_ledger()
    led.append({"i": 0})
    assert led.verify_chain().anchored is False
    assert led.verify_chain(expected=led.checkpoint()).anchored is True
    assert led.verify_chain().to_dict()["anchored"] is False


# ----------------------------------------------- canonical encoding is injective


def test_canonical_json_refuses_to_coerce_non_json_values():
    """A lossy encoder would let two distinct payloads share one hash."""
    from trustfed.ledger import LedgerStorageError
    from trustfed.ledger.crypto import canonical_json

    with pytest.raises(LedgerStorageError):
        hash_payload({"x": Path("/a")})
    with pytest.raises(LedgerStorageError):
        canonical_json({"x": {1, 2}})
    assert hash_payload({"x": "/a"}) == hash_payload({"x": "/a"})

    led = make_memory_ledger()
    with pytest.raises(LedgerStorageError):
        led.append({"path": Path("/tmp/weights.npz")})
    assert len(led) == 0
    led.append({"path": str(Path("/tmp/weights.npz"))})
    assert led.verify_chain(expected=led.checkpoint()).ok is True
