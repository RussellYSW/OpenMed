"""The six sections of the OpenMed model-commons demo, one module each.

Split out of ``run_demo.py`` so each step of the trust plane can be read on its
own; ``run_demo.py`` is the driver that calls them in order. Nothing here is a
library API -- it is example code, and it imports ``trustfed`` the same way a
reader's own script would.
"""

from __future__ import annotations

from openmed_demo.certify import certify
from openmed_demo.credit import credit_and_reciprocity
from openmed_demo.derive import derive, show_lineage
from openmed_demo.fixtures import build, make_attestor, rule
from openmed_demo.publish import publish_root, show_unapproved_publish
from openmed_demo.tamper import show_tamper_evidence, show_truncation_evidence

__all__ = [
    "build",
    "certify",
    "credit_and_reciprocity",
    "derive",
    "make_attestor",
    "publish_root",
    "rule",
    "show_lineage",
    "show_tamper_evidence",
    "show_truncation_evidence",
    "show_unapproved_publish",
]
