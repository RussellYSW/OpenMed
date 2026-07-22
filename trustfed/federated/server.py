"""Federated coordinator.

Each round the server: (1) broadcasts the global model, (2) collects updates,
(3) rejects any update that fails attestation, (4) aggregates the survivors with
a Byzantine-robust rule, and (5) evaluates the new global model on a held-out
test set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from trustfed.aggregation.robust import fedavg
from trustfed.attestation.attestor import Attestor
from trustfed.federated.client import Client, Update
from trustfed.metrics import accuracy, roc_auc
from trustfed.models.logistic import LogisticRegressionModel


@dataclass
class RoundResult:
    round: int
    auc: float
    accuracy: float
    n_accepted: int
    n_rejected: int
    rejected_ids: List[str] = field(default_factory=list)


class Server:
    def __init__(
        self,
        n_features: int,
        *,
        aggregator: Callable = fedavg,
        attestor: Optional[Attestor] = None,
        l2: float = 1e-3,
        lr: float = 0.2,
        n_byzantine: int = 0,
        trimmed_beta: float = 0.2,
    ):
        self.model = LogisticRegressionModel(n_features=n_features, l2=l2, lr=lr)
        self.aggregator = aggregator
        self.attestor = attestor
        self.n_byzantine = int(n_byzantine)
        self.trimmed_beta = float(trimmed_beta)
        self.history: List[RoundResult] = []

    def _accept(self, updates: List[Update]):
        accepted, rejected = [], []
        for u in updates:
            if self.attestor is not None:
                if u.quote is None or not self.attestor.verify(u.quote):
                    rejected.append(u)
                    continue
            accepted.append(u)
        return accepted, rejected

    def _aggregate(self, updates: List[Update]) -> np.ndarray:
        thetas = [u.theta for u in updates]
        weights = np.array([u.n_samples for u in updates], dtype=float)
        return self.aggregator(
            thetas,
            weights=weights,
            n_byzantine=self.n_byzantine,
            beta=self.trimmed_beta,
        )

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        proba = self.model.predict_proba(X_test)
        return roc_auc(y_test, proba), accuracy(y_test, proba)

    def fit(
        self,
        clients: List[Client],
        X_test: np.ndarray,
        y_test: np.ndarray,
        *,
        rounds: int = 30,
        local_epochs: int = 5,
        verbose: bool = False,
    ) -> List[RoundResult]:
        for r in range(1, rounds + 1):
            global_theta = self.model.get_params()
            updates = [c.train(global_theta, local_epochs) for c in clients]

            accepted, rejected = self._accept(updates)
            if not accepted:
                raise RuntimeError(
                    f"round {r}: every client failed attestation; nothing to aggregate"
                )

            self.model.set_params(self._aggregate(accepted))
            auc, acc = self.evaluate(X_test, y_test)

            result = RoundResult(
                round=r,
                auc=auc,
                accuracy=acc,
                n_accepted=len(accepted),
                n_rejected=len(rejected),
                rejected_ids=[u.client_id for u in rejected],
            )
            self.history.append(result)
            if verbose:
                rej = f" rejected={result.rejected_ids}" if rejected else ""
                print(
                    f"  round {r:>2}  AUC={auc:.3f}  acc={acc:.3f}  "
                    f"accepted={len(accepted)}{rej}"
                )
        return self.history
