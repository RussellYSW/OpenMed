# Security policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately to the maintainers (open a
GitHub security advisory or email the address in `CITATION.cff`). Do not open a
public issue for undisclosed vulnerabilities. We aim to acknowledge reports
within a few business days.

## Scope and honest limitations of the reference attestor

`trustfed.attestation.MockSoftwareAttestor` is a **software mock** intended to
demonstrate the protocol and to unit-test the accept/reject logic. It is **not a
real security boundary**:

- The "code measurement" is a hash of a *declared* identity string, not a
  hardware measurement of a loaded enclave. A malicious client can declare the
  approved identity.
- The HMAC "quote" relies on a shared secret in process memory, not a hardware
  root of trust.

To obtain real guarantees, implement the `Attestor` interface against a genuine
TEE (Intel SGX/TDX via DCAP, AMD SEV-SNP) so that measurements are hardware-rooted
and quotes are verifiable off-host. Until then, treat the admission gate as a
functional placeholder.

## What the Byzantine-robust aggregators do and do not guarantee

The aggregators (Krum, coordinate median, trimmed mean) tolerate a **bounded**
fraction of malicious updates under stated conditions (see
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)). They do not defend against:

- More than the assumed number of Byzantine clients `f`.
- Adaptive attacks specifically crafted against the chosen rule (an active
  research area; see the roadmap).
- Privacy leakage from shared updates — combine with secure aggregation and/or
  differential privacy for confidentiality guarantees.
