"""Federated client: holds local (private) data, trains locally, and returns an
attested parameter update. A client may be honest or Byzantine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from trustfed.attestation.attestor import Attestor, Quote
from trustfed.models.logistic import LogisticRegressionModel

# The code identity a client declares during attestation. In a real deployment
# this corresponds to the measured enclave image; here it is a constant string
# for honest clients and something different for tampered ones.
APPROVED_CODE_IDENTITY = "trustfed-client@0.1.0"


@dataclass
class Update:
    """What a client sends to the server each round."""

    client_id: str
    theta: np.ndarray
    n_samples: int
    quote: Optional[Quote] = None


class Client:
    def __init__(
        self,
        client_id: str,
        X: np.ndarray,
        y: np.ndarray,
        *,
        l2: float = 1e-3,
        lr: float = 0.2,
        attestor: Optional[Attestor] = None,
        code_identity: str = APPROVED_CODE_IDENTITY,
        config: str = "default",
        malicious: bool = False,
        attack: Optional[Callable] = None,
        attack_kwargs: Optional[dict] = None,
        seed: int = 0,
    ):
        self.client_id = client_id
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.l2 = l2
        self.lr = lr
        self.attestor = attestor
        self.code_identity = code_identity
        self.config = config
        self.malicious = malicious
        self.attack = attack
        self.attack_kwargs = attack_kwargs or {}
        self._rng = np.random.default_rng(seed)

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    def train(self, global_theta: np.ndarray, local_epochs: int = 5) -> Update:
        model = LogisticRegressionModel(
            n_features=self.X.shape[1], l2=self.l2, lr=self.lr
        )
        model.set_params(global_theta)
        theta = model.local_train(self.X, self.y, epochs=local_epochs)

        if self.malicious and self.attack is not None:
            theta = self.attack(theta, self._rng, **self.attack_kwargs)

        quote = None
        if self.attestor is not None:
            quote = self.attestor.generate_quote(
                self.client_id, self.code_identity, self.config
            )

        return Update(
            client_id=self.client_id,
            theta=np.asarray(theta, dtype=float),
            n_samples=self.n_samples,
            quote=quote,
        )
