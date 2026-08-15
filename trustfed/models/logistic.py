"""A minimal L2-regularized logistic regression trained with gradient descent.

Kept deliberately small (numpy only) so the whole federated loop is transparent
and easy to audit. The model exposes its parameters as a single flat vector,
which is what the aggregation and attack modules operate on.
"""

from __future__ import annotations

import numpy as np

from trustfed.models.errors import ModelError


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically-safe logistic function (input clipped to +/-30)."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


class LogisticRegressionModel:
    """L2-regularized logistic regression trained by full-batch gradient descent.

    Parameters
    ----------
    n_features:
        Input dimensionality.
    l2:
        L2 penalty on the weight vector (the intercept is not penalized).
    lr:
        Gradient-descent step size.

    Assumes features are on a comparable scale (the synthetic generator produces
    standardized features); no internal normalization is performed.
    """

    def __init__(self, n_features: int, l2: float = 1e-3, lr: float = 0.2):
        self.n_features = int(n_features)
        self.w = np.zeros(self.n_features, dtype=float)
        self.b = 0.0
        self.l2 = float(l2)
        self.lr = float(lr)

    @property
    def n_params(self) -> int:
        """Length of the flat parameter vector (weights plus intercept)."""
        return self.n_features + 1

    def get_params(self) -> np.ndarray:
        """Return parameters as a flat vector ``[w_0, ..., w_{d-1}, b]``."""
        return np.concatenate([self.w, [self.b]])

    def set_params(self, theta: np.ndarray) -> None:
        """Load a flat parameter vector.

        Raises
        ------
        ModelError
            If ``theta`` has the wrong length or contains non-finite values.
        """
        theta = np.asarray(theta, dtype=float)
        if theta.ndim != 1 or theta.shape[0] != self.n_params:
            raise ModelError(
                f"expected {self.n_params} params, got shape {theta.shape}"
            )
        if not np.all(np.isfinite(theta)):
            raise ModelError("parameter vector contains non-finite values")
        self.w = theta[:-1].copy()
        self.b = float(theta[-1])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicted probability of the positive class for each row of ``X``."""
        return sigmoid(X @ self.w + self.b)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Mean binary cross-entropy plus the L2 penalty.

        Returns ``0.0`` for an empty dataset. Probabilities are clipped away
        from 0/1 so the logarithm stays finite.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.shape[0] == 0:
            return 0.0
        p = np.clip(self.predict_proba(X), 1e-12, 1.0 - 1e-12)
        nll = -float(np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
        return nll + 0.5 * self.l2 * float(np.dot(self.w, self.w))

    def local_train(self, X: np.ndarray, y: np.ndarray, epochs: int = 5) -> np.ndarray:
        """Run ``epochs`` full-batch gradient steps locally; return new params.

        A no-op for an empty dataset.

        Raises
        ------
        ModelError
            If ``X`` does not have ``n_features`` columns or the label vector
            length does not match.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = X.shape[0]
        if n == 0:
            return self.get_params()
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ModelError(
                f"expected X with {self.n_features} columns, got shape {X.shape}"
            )
        if y.shape[0] != n:
            raise ModelError(
                f"X has {n} rows but y has {y.shape[0]} entries"
            )
        for _ in range(int(epochs)):
            p = self.predict_proba(X)
            err = p - y
            grad_w = X.T @ err / n + self.l2 * self.w
            grad_b = float(np.mean(err))
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self.get_params()


__all__ = ["LogisticRegressionModel", "sigmoid"]
