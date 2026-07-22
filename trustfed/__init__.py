"""TrustFed: a zero-trust security layer for open-source clinical AI.

TrustFed brings software-supply-chain integrity to collaborative, multi-site
model training. The technical setting is cross-silo federated learning, but the
contribution is *security*, not learning: it does not replace general-purpose
federated frameworks (Flower, NVIDIA FLARE, Fed-BioMed, NeuroFLAME) -- it hardens
the admission and aggregation step those frameworks leave trusting-by-default,
so the shared model stays correct even when some sites are compromised, faulty,
or malicious:

* **Attestation** (``trustfed.attestation``) gates which clients may contribute
  an update, by verifying a remote-attestation quote over the code measurement
  and configuration of each participant. The reference implementation is a
  software mock; real TEEs (Intel SGX/TDX, AMD SEV-SNP) plug in behind the same
  interface.
* **Byzantine-robust aggregation** (``trustfed.aggregation``) tolerates a
  bounded number of malicious or faulty updates that survive attestation
  (corrupted data, compromised runtime, hardware faults) via Krum, coordinate
  median, and trimmed mean.

The ``examples/parkinson_decline`` demo shows the two layers defending a
multi-site Parkinson's-disease rapid-decline classifier trained on *synthetic*
data (no patient data ships with this project).
"""

__version__ = "0.1.0"

from trustfed.models.logistic import LogisticRegressionModel
from trustfed.federated.server import Server
from trustfed.federated.client import Client, Update

__all__ = [
    "LogisticRegressionModel",
    "Server",
    "Client",
    "Update",
    "__version__",
]
