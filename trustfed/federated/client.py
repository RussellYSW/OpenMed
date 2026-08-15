"""Federated client: holds local (private) data, trains locally, and returns an
attested parameter update. A client may be honest or Byzantine.

Two independent corruption channels are modelled, matching the two threat models
in :mod:`trustfed.attack`:

* ``data_attack`` corrupts the local training set once, at construction time
  (mislabelled export, broken instrument). The client then trains honestly, so
  its update has an entirely normal norm.
* ``attack`` rewrites the parameter vector after training (compromised host,
  Byzantine fault).

A client that is *not* marked ``malicious`` never runs either channel, so the
adversarial machinery cannot fire by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np

from trustfed.attestation.attestor import Attestor, Quote
from trustfed.federated.errors import ClientConfigError
from trustfed.models.logistic import LogisticRegressionModel

# The code identity a client declares during attestation. In a real deployment
# this corresponds to the measured enclave image; here it is a constant string
# for honest clients and something different for tampered ones.
APPROVED_CODE_IDENTITY = "trustfed-client@0.1.0"

__all__ = ["Client", "Update", "APPROVED_CODE_IDENTITY"]


@dataclass
class Update:
    """What a client sends to the server each round.

    Attributes
    ----------
    client_id:
        Identifier of the submitting site.
    theta:
        Flat parameter vector after local training (and after any attack).
    n_samples:
        Self-reported local sample count, used for FedAvg weighting. It is
        *unverified*: a malicious site can inflate it to buy influence, which is
        one reason weighted averaging is not a safe default.
    quote:
        Attestation quote, when the client was configured with an attestor.
    loss:
        Self-reported local training loss after the round. Used by
        :class:`~trustfed.federated.selection.LossBasedSelector`; also
        unverified.
    metadata:
        Free-form, non-authoritative annotations for logging.
    """

    client_id: str
    theta: np.ndarray
    n_samples: int
    quote: Optional[Quote] = None
    loss: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Client:
    """A simulated participating site.

    Parameters
    ----------
    client_id:
        Site identifier; must be unique within a federation.
    X, y:
        The site's private training data. Never leaves the client object.
    l2, lr:
        Local optimizer hyperparameters.
    attestor:
        If given, the client produces a quote with each update.
    code_identity:
        The code measurement the client declares. Set to something outside the
        server's allow-list to simulate a tampered client.
    config:
        Configuration string included in the quote.
    malicious:
        Master switch. Attacks only run when this is ``True``.
    attack:
        Update-space attack callable ``(theta, rng, **kwargs) -> theta``.
    attack_kwargs:
        Extra keyword arguments for ``attack``.
    data_attack:
        Data-space attack callable ``(X, y, rng, **kwargs) -> (X, y)``, applied
        once at construction.
    data_attack_kwargs:
        Extra keyword arguments for ``data_attack``.
    seed:
        Seed for this client's attack randomness.

    Raises
    ------
    ClientConfigError
        If ``X``/``y`` shapes disagree, or an attack is configured on a client
        that is not marked malicious.
    """

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
        attack: Optional[Callable[..., np.ndarray]] = None,
        attack_kwargs: Optional[Dict[str, Any]] = None,
        data_attack: Optional[Callable[..., Any]] = None,
        data_attack_kwargs: Optional[Dict[str, Any]] = None,
        seed: int = 0,
    ):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if X_arr.ndim != 2:
            raise ClientConfigError(
                f"{client_id}: X must be 2-D, got shape {X_arr.shape}"
            )
        if y_arr.shape[0] != X_arr.shape[0]:
            raise ClientConfigError(
                f"{client_id}: X has {X_arr.shape[0]} rows but y has {y_arr.shape[0]}"
            )
        if not malicious and (attack is not None or data_attack is not None):
            raise ClientConfigError(
                f"{client_id}: an attack was configured but malicious=False; "
                "set malicious=True to arm it (this guard exists so a stray "
                "keyword cannot silently poison an honest site)"
            )

        self.client_id = client_id
        self.l2 = float(l2)
        self.lr = float(lr)
        self.attestor = attestor
        self.code_identity = code_identity
        self.config = config
        self.malicious = bool(malicious)
        self.attack = attack
        self.attack_kwargs = dict(attack_kwargs or {})
        self.data_attack = data_attack
        self.data_attack_kwargs = dict(data_attack_kwargs or {})
        self.seed = int(seed)
        self._rng = np.random.default_rng(seed)

        if self.malicious and self.data_attack is not None:
            X_arr, y_arr = self.data_attack(
                X_arr, y_arr, self._rng, **self.data_attack_kwargs
            )
            X_arr = np.asarray(X_arr, dtype=float)
            y_arr = np.asarray(y_arr, dtype=float)
        self.X = X_arr
        self.y = y_arr

    @property
    def n_samples(self) -> int:
        """Number of local training samples."""
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        """Local feature dimensionality."""
        return int(self.X.shape[1])

    def train(
        self,
        global_theta: np.ndarray,
        local_epochs: int = 5,
        *,
        n_clients: int = 1,
        nonce: str = "",
    ) -> Update:
        """Train locally from ``global_theta`` and return the update to submit.

        Parameters
        ----------
        global_theta:
            The broadcast global parameter vector.
        local_epochs:
            Number of local gradient steps.
        n_clients:
            Size of the current cohort, passed through to update-space attacks
            that need it (model replacement scales by the cohort size).
        nonce:
            Challenge issued by the verifier for *this* round, echoed into the
            attestation quote. An empty string produces a quote with no
            freshness binding, which a verifier whose policy sets
            ``require_nonce`` will refuse; the server
            (:meth:`trustfed.federated.server.Server.fit`) supplies one
            whenever its attestor can issue challenges.
        """
        model = LogisticRegressionModel(
            n_features=self.n_features, l2=self.l2, lr=self.lr
        )
        model.set_params(global_theta)
        theta = model.local_train(self.X, self.y, epochs=local_epochs)
        loss = model.loss(self.X, self.y)

        if self.malicious and self.attack is not None:
            theta = self.attack(
                theta,
                self._rng,
                global_theta=np.asarray(global_theta, dtype=float),
                n_clients=int(n_clients),
                **self.attack_kwargs,
            )

        quote = None
        if self.attestor is not None:
            if nonce:
                quote = self.attestor.generate_quote(
                    self.client_id, self.code_identity, self.config, nonce=nonce
                )
            else:
                # Keep the two-field call for backends that predate the
                # challenge protocol; such a quote is a bearer token and the
                # verifier is expected to refuse it under require_nonce.
                quote = self.attestor.generate_quote(
                    self.client_id, self.code_identity, self.config
                )

        return Update(
            client_id=self.client_id,
            theta=np.asarray(theta, dtype=float),
            n_samples=self.n_samples,
            quote=quote,
            loss=float(loss),
        )
