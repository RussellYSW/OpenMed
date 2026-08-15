"""Reference model used throughout TrustFed.

Deliberately one small, transparent model (numpy-only logistic regression) so
that the federated protocol, the attacks and the defenses can be read and
audited without a deep-learning stack in the way. The aggregation, attack and
quality modules operate on a flat parameter vector and do not depend on this
particular model.
"""

from __future__ import annotations

from trustfed.models.errors import ModelError
from trustfed.models.logistic import LogisticRegressionModel, sigmoid

__all__ = ["LogisticRegressionModel", "sigmoid", "ModelError"]
