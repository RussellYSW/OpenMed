"""Attestation: the admission gate that decides who may contribute.

A client presents a *quote* over its code measurement, its configuration, and a
challenge nonce; the verifier checks the signature, the freshness, and an
allow-list policy before any update is aggregated or any bundle is published.

The shipped backend, :class:`MockSoftwareAttestor`, is a **software mock and not
a security boundary** -- it signs whatever code identity the caller declares.
:class:`NodeKeyAttestor` is the software-only backend for deployments without
attestation hardware: the client measures its actual code
(:func:`software_measurement`) and signs the quote with its own registered
key, so the verifier authenticates the *site*, not the hardware. Real TEE
backends (SGX/TDX via DCAP, SEV-SNP) implement the same :class:`Attestor`
interface. See :mod:`trustfed.attestation.attestor`.
"""

from __future__ import annotations

from trustfed.attestation.attestor import (
    AttestationResult,
    Attestor,
    MockSoftwareAttestor,
    Quote,
    measure_code,
)
from trustfed.attestation.errors import AttestationError, NonceError, PolicyError
from trustfed.attestation.measure import SoftwareMeasurement, software_measurement
from trustfed.attestation.nodekey import NODE_KEY_ATTESTOR_ID, NodeKeyAttestor
from trustfed.attestation.policy import AttestationPolicy, NonceStore

__all__ = [
    "AttestationError",
    "AttestationPolicy",
    "AttestationResult",
    "Attestor",
    "MockSoftwareAttestor",
    "NODE_KEY_ATTESTOR_ID",
    "NodeKeyAttestor",
    "NonceError",
    "NonceStore",
    "PolicyError",
    "Quote",
    "SoftwareMeasurement",
    "measure_code",
    "software_measurement",
]
