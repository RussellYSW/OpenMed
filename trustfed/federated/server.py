"""Federated coordinator.

Each round the server: (1) selects a cohort, (2) broadcasts the global model,
(3) collects updates, (4) rejects any update that fails attestation, (5)
aggregates the survivors with a Byzantine-robust rule, (6) evaluates the new
global model on a held-out test set, and (7) folds what it observed into the
per-client history that reputation-weighted selection consumes.

The server trusts nothing a client says about itself: sample counts and losses
are recorded as *self-reported*, and only the attestation quote and the
aggregation rule stand between a malicious update and the global model.

Attestation is challenge-response, not bearer-token: each round the server asks
its attestor for a fresh nonce per selected client, the client echoes that nonce
into its quote, and the server verifies the quote *against the nonce it issued*.
A quote captured in one round therefore cannot be replayed in another. When the
configured attestor cannot issue challenges (no nonce store), the server falls
back to nonce-free verification and the freshness property is simply absent --
which is why a deployment should configure a
:class:`~trustfed.attestation.policy.NonceStore` and a policy with
``require_nonce=True``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from trustfed.aggregation.robust import aggregator_name, fedavg
from trustfed.attestation.attestor import Attestor
from trustfed.attestation.errors import PolicyError
from trustfed.federated.client import Client, Update
from trustfed.federated.errors import NoAcceptedUpdatesError
from trustfed.federated.history import ClientHistory, RoundObservation
from trustfed.federated.selection import AllClientsSelector, Selector
from trustfed.metrics import accuracy, roc_auc
from trustfed.models.logistic import LogisticRegressionModel


@dataclass
class RoundResult:
    """Outcome of a single federated round.

    ``rejected_reasons`` maps a rejected client id to the machine-readable
    reason code the attestor returned (see
    :mod:`trustfed.attestation.attestor`), so a caller can tell a replayed quote
    (``nonce_mismatch`` / ``nonce_rejected``) from tampered code
    (``unapproved_measurement``).
    """

    round: int
    auc: float
    accuracy: float
    n_accepted: int
    n_rejected: int
    rejected_ids: List[str] = field(default_factory=list)
    selected_ids: List[str] = field(default_factory=list)
    mean_loss: Optional[float] = None
    max_deviation: Optional[float] = None
    rejected_reasons: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable view of this round."""
        return {
            "round": self.round,
            "auc": None if np.isnan(self.auc) else round(float(self.auc), 6),
            "accuracy": round(float(self.accuracy), 6),
            "n_accepted": self.n_accepted,
            "n_rejected": self.n_rejected,
            "rejected_ids": list(self.rejected_ids),
            "selected_ids": list(self.selected_ids),
            "mean_loss": None if self.mean_loss is None else round(self.mean_loss, 6),
            "max_deviation": (
                None if self.max_deviation is None else round(self.max_deviation, 6)
            ),
            "rejected_reasons": dict(self.rejected_reasons),
        }


class Server:
    """Coordinator for one federated training run.

    Parameters
    ----------
    n_features:
        Feature dimensionality of the global model.
    aggregator:
        Aggregation rule (see :mod:`trustfed.aggregation`). Defaults to FedAvg,
        which has **no** Byzantine tolerance and is only a baseline.
    attestor:
        If given, every update must present a verifying quote or it is
        rejected. The server drives the challenge-response protocol: it calls
        ``attestor.issue_nonce(client_id)`` each round and verifies each quote
        against the nonce issued to that client, so quotes cannot be replayed.
        An attestor without a nonce store degrades to nonce-free verification.
    l2, lr:
        Global model hyperparameters, mirrored to the clients' local optimizer.
    n_byzantine:
        The defender's assumed attacker count, forwarded to rules that need it
        (Krum). Over-estimating wastes honest updates; under-estimating breaks
        the guarantee.
    trimmed_beta:
        Trim fraction for the trimmed mean.
    selector:
        Client-selection policy; defaults to "every client, every round".
    history:
        Behavioural :class:`~trustfed.federated.history.ClientHistory`; one is
        created if not supplied. Exposed afterwards as ``server.client_history``
        (``server.history`` is the per-round log).
    adversary:
        **Simulation hook.** An object with
        ``craft(honest_updates, n_malicious, ...)`` (see
        :class:`trustfed.attack.adaptive.AdaptiveAdversary`) used to model
        colluding attackers that observe the honest updates of the round before
        submitting. Real deployments have no such hook; it exists so the
        benchmark can evaluate defenses against a worst-case adaptive attacker.
    byzantine_ids:
        Ids of the clients the ``adversary`` controls.
    weighted:
        Whether to pass self-reported sample counts as aggregation weights.
        Defaults to ``True`` for FedAvg compatibility; note that a lying client
        gains influence this way.
    """

    def __init__(
        self,
        n_features: int,
        *,
        aggregator: Callable[..., np.ndarray] = fedavg,
        attestor: Optional[Attestor] = None,
        l2: float = 1e-3,
        lr: float = 0.2,
        n_byzantine: int = 0,
        trimmed_beta: float = 0.2,
        selector: Optional[Selector] = None,
        history: Optional[ClientHistory] = None,
        adversary: Optional[Any] = None,
        byzantine_ids: Sequence[str] = (),
        weighted: bool = True,
    ):
        self.model = LogisticRegressionModel(n_features=n_features, l2=l2, lr=lr)
        self.aggregator = aggregator
        self.attestor = attestor
        self.n_byzantine = int(n_byzantine)
        self.trimmed_beta = float(trimmed_beta)
        self.selector: Selector = selector or AllClientsSelector()
        self.client_history = history if history is not None else ClientHistory()
        self.adversary = adversary
        self.byzantine_ids = set(byzantine_ids)
        self.weighted = bool(weighted)
        self.rounds_run = 0
        self.history: List[RoundResult] = []
        self._challenge_response = False

    @property
    def challenge_response(self) -> bool:
        """Whether the attestor issued freshness challenges in the last round.

        ``False`` before the first round, when no attestor is configured, or
        when the configured attestor has no challenge store -- in which case
        quotes are bearer tokens and replay is not detected here.
        """
        return bool(self._challenge_response)

    def _issue_nonces(self, client_ids: Sequence[str]) -> Dict[str, Optional[str]]:
        """Issue one challenge nonce per selected client for this round.

        Returns ``{client_id: nonce}``. A value of ``None`` means "this attestor
        cannot issue challenges", in which case verification proceeds without a
        freshness check and quotes are, in the strict sense, replayable. That
        degradation is deliberate -- it keeps attestors that predate the
        challenge protocol usable -- and it is visible in :meth:`describe` as
        ``challenge_response: False``.
        """
        if self.attestor is None:
            self._challenge_response = False
            return {}
        out: Dict[str, Optional[str]] = {}
        for cid in client_ids:
            try:
                out[cid] = self.attestor.issue_nonce(cid)
            except PolicyError:
                out[cid] = None
        self._challenge_response = any(v for v in out.values())
        return out

    def _accept(
        self,
        updates: List[Update],
        expected_nonces: Optional[Dict[str, Optional[str]]] = None,
    ) -> Tuple[List[Update], List[Update]]:
        """Split updates into (accepted, rejected) by attestation verdict.

        Each update is checked against the nonce that was issued to *that*
        client for *this* round, so a quote lifted from another round or another
        client fails. The reason code is recorded on the rejected update's
        ``metadata['attestation_reason']``.
        """
        expected = expected_nonces or {}
        accepted, rejected = [], []
        for u in updates:
            if self.attestor is not None:
                if u.quote is None:
                    u.metadata["attestation_reason"] = "missing_quote"
                    rejected.append(u)
                    continue
                if u.quote.client_id != u.client_id:
                    # A valid quote bound to a different client is a replay.
                    u.metadata["attestation_reason"] = "client_id_mismatch"
                    rejected.append(u)
                    continue
                result = self.attestor.check(
                    u.quote, expected_nonce=expected.get(u.client_id)
                )
                if not result.ok:
                    u.metadata["attestation_reason"] = result.reason_code
                    u.metadata["attestation_detail"] = result.detail
                    rejected.append(u)
                    continue
            accepted.append(u)
        return accepted, rejected

    def _aggregate(self, updates: List[Update]) -> np.ndarray:
        """Apply the configured aggregation rule to the accepted updates."""
        thetas = [u.theta for u in updates]
        weights = (
            np.array([u.n_samples for u in updates], dtype=float)
            if self.weighted
            else None
        )
        return self.aggregator(
            thetas,
            weights=weights,
            n_byzantine=self.n_byzantine,
            beta=self.trimmed_beta,
            reference=self.model.get_params(),
        )

    def _apply_adversary(
        self, updates: List[Update], global_theta: np.ndarray
    ) -> List[Update]:
        """Replace controlled clients' updates with coordinated malicious ones.

        Simulation-only: models attackers that see the honest updates of the
        round. Returns the list unchanged when no adversary is configured.
        """
        if self.adversary is None or not self.byzantine_ids:
            return updates
        honest = [u.theta for u in updates if u.client_id not in self.byzantine_ids]
        targets = [u for u in updates if u.client_id in self.byzantine_ids]
        if not honest or not targets:
            return updates
        crafted = self.adversary.craft(
            honest,
            len(targets),
            global_theta=global_theta,
            n_byzantine_assumed=self.n_byzantine or len(targets),
            round_index=self.rounds_run + 1,
        )
        for update, theta in zip(targets, crafted):
            update.theta = np.asarray(theta, dtype=float)
            update.metadata["adaptive"] = True
        return updates

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
        """Return ``(roc_auc, accuracy)`` of the current global model."""
        proba = self.model.predict_proba(X_test)
        return roc_auc(y_test, proba), accuracy(y_test, proba)

    def _observe(
        self, accepted: List[Update], rejected: List[Update], theta: np.ndarray
    ) -> Tuple[Optional[float], Optional[float]]:
        """Record this round in the client history; return (mean loss, max deviation)."""
        observations = []
        deviations = []
        losses = []
        for u in accepted:
            dev = float(np.linalg.norm(u.theta - theta))
            deviations.append(dev)
            if u.loss is not None:
                losses.append(u.loss)
            observations.append(
                RoundObservation(
                    client_id=u.client_id, accepted=True, loss=u.loss, deviation=dev
                )
            )
        for u in rejected:
            observations.append(
                RoundObservation(client_id=u.client_id, accepted=False, loss=u.loss)
            )
        self.client_history.observe_round(observations)
        return (
            float(np.mean(losses)) if losses else None,
            float(np.max(deviations)) if deviations else None,
        )

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
        """Run ``rounds`` federated rounds and return the per-round log.

        Raises
        ------
        NoAcceptedUpdatesError
            If every update in a round is rejected. This is fatal by design:
            publishing a round with no attested contribution would defeat the
            point of attestation.
        """
        by_id = {c.client_id: c for c in clients}
        for _ in range(int(rounds)):
            r = self.rounds_run + 1
            global_theta = self.model.get_params()
            chosen = self.selector.select(
                list(by_id.keys()), round_index=r, history=self.client_history
            )
            cohort = [by_id[cid] for cid in chosen]
            nonces = self._issue_nonces([c.client_id for c in cohort])
            updates = [
                c.train(
                    global_theta,
                    local_epochs,
                    n_clients=len(cohort),
                    nonce=nonces.get(c.client_id) or "",
                )
                for c in cohort
            ]
            updates = self._apply_adversary(updates, global_theta)

            accepted, rejected = self._accept(updates, nonces)
            if not accepted:
                raise NoAcceptedUpdatesError(
                    f"round {r}: every client failed attestation; nothing to aggregate"
                )

            new_theta = self._aggregate(accepted)
            self.model.set_params(new_theta)
            auc, acc = self.evaluate(X_test, y_test)
            mean_loss, max_dev = self._observe(accepted, rejected, new_theta)

            result = RoundResult(
                round=r,
                auc=auc,
                accuracy=acc,
                n_accepted=len(accepted),
                n_rejected=len(rejected),
                rejected_ids=[u.client_id for u in rejected],
                selected_ids=list(chosen),
                mean_loss=mean_loss,
                max_deviation=max_dev,
                rejected_reasons={
                    u.client_id: str(u.metadata.get("attestation_reason", "rejected"))
                    for u in rejected
                },
            )
            self.history.append(result)
            self.rounds_run = r
            if verbose:
                rej = f" rejected={result.rejected_ids}" if rejected else ""
                print(
                    f"  round {r:>2}  AUC={auc:.3f}  acc={acc:.3f}  "
                    f"accepted={len(accepted)}{rej}"
                )
        return self.history

    def describe(self) -> Dict[str, Any]:
        """JSON-serializable description of this server's configuration."""
        return {
            "aggregator": aggregator_name(self.aggregator),
            "n_byzantine_assumed": self.n_byzantine,
            "trimmed_beta": self.trimmed_beta,
            "attestation": self.attestor is not None,
            "challenge_response": self.challenge_response,
            "selector": self.selector.describe(),
            "weighted": self.weighted,
            "rounds_run": self.rounds_run,
        }


__all__ = ["Server", "RoundResult"]
