"""A minimal L2-regularized logistic regression trained with gradient descent.

Kept deliberately small (numpy only) so the whole federated loop is transparent
and easy to audit. The model exposes its parameters as a single flat vector,
which is what the aggregation and attack modules operate on.
"""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


class LogisticRegressionModel:
    def __init__(self, n_features: int, l2: float = 1e-3, lr: float = 0.2):
        self.n_features = int(n_features)
        self.w = np.zeros(self.n_features, dtype=float)
        self.b = 0.0
        self.l2 = float(l2)
        self.lr = float(lr)

    @property
    def n_params(self) -> int:
        return self.n_features + 1

    def get_params(self) -> np.ndarray:
        """Return parameters as a flat vector ``[w_0, ..., w_{d-1}, b]``."""
        return np.concatenate([self.w, [self.b]])

    def set_params(self, theta: np.ndarray) -> None:
        theta = np.asarray(theta, dtype=float)
        if theta.shape[0] != self.n_params:
            raise ValueError(
                f"expected {self.n_params} params, got {theta.shape[0]}"
            )
        self.w = theta[:-1].copy()
        self.b = float(theta[-1])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(X @ self.w + self.b)

    def local_train(self, X: np.ndarray, y: np.ndarray, epochs: int = 5) -> np.ndarray:
        """Run full-batch gradient descent locally and return the new params."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = X.shape[0]
        if n == 0:
            return self.get_params()
        for _ in range(int(epochs)):
            p = self.predict_proba(X)
            err = p - y
            grad_w = X.T @ err / n + self.l2 * self.w
            grad_b = float(np.mean(err))
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self.get_params()
