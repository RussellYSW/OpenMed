"""OpenMed Hub -- the running service around the ``trustfed`` trust plane.

``trustfed`` implements the six components (co-training, ledger, registry,
certification, quality analysis, credit) as in-process objects. The hub is the
deployment that makes them a *commons*: institutions and users register, sites
stand up a node and complete a verification handshake, contributors upload
model bundles, the automated gate screens each submission, reviewers from other
institutions sign certifications, consumers download certified base models
after lightweight registration, and every action lands on a tamper-evident
ledger.

Run it with ``openmed serve`` (see :mod:`openmed_hub.cli`).
"""

__all__ = ["create_app"]

from openmed_hub.app import create_app  # noqa: E402
