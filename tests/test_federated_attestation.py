"""Attestation freshness in the federated plane (challenge-response, anti-replay).

The server issues a single-use nonce per client per round; a client echoes it
into its quote; the server verifies the quote against the nonce it issued. These
tests cover the property that gives: a quote is not a bearer token.
"""

import numpy as np

from trustfed.attestation.attestor import MockSoftwareAttestor, measure_code
from trustfed.attestation.policy import AttestationPolicy, NonceStore
from trustfed.data.synthetic import make_federated_parkinson
from trustfed.federated.client import APPROVED_CODE_IDENTITY, Client, Update
from trustfed.federated.server import Server

ROOT_KEY = b"integration-test-key"
APPROVED = measure_code(APPROVED_CODE_IDENTITY)


def nonce_attestor(**policy_kwargs):
    """A mock attestor that can issue single-use challenges."""
    policy = AttestationPolicy(approved_measurements={APPROVED}, **policy_kwargs)
    return MockSoftwareAttestor(ROOT_KEY, policy=policy, nonce_store=NonceStore())


class ReplayingClient(Client):
    """Captures its first quote and re-presents it in every later round."""

    def train(self, *args, **kwargs):
        update = super().train(*args, **kwargs)
        if getattr(self, "_captured", None) is None:
            self._captured = update.quote
        else:
            update.quote = self._captured
        return update


def test_a_quote_replayed_in_a_later_round_is_rejected():
    sites, X_test, y_test = make_federated_parkinson(seed=21, n_sites=4)
    att = nonce_attestor(require_nonce=True)
    clients = [
        Client(s.site_id, s.X, s.y, attestor=att, seed=i)
        for i, s in enumerate(sites[:3])
    ]
    replayer = ReplayingClient(
        sites[3].site_id, sites[3].X, sites[3].y, attestor=att, seed=3
    )
    server = Server(X_test.shape[1], attestor=att)
    server.fit(clients + [replayer], X_test, y_test, rounds=3)

    assert server.history[0].n_rejected == 0  # the first presentation is fresh
    for r in server.history[1:]:
        assert r.rejected_ids == [replayer.client_id]
        assert r.rejected_reasons[replayer.client_id] in {
            "nonce_mismatch",
            "nonce_rejected",
        }


def test_the_same_nonce_presented_twice_is_rejected_as_replay():
    # Exercises the admission gate directly: two updates carrying one captured
    # quote, checked against the challenge that quote answers.
    att = nonce_attestor(require_nonce=True)
    nonce = att.issue_nonce("site_00")
    quote = att.generate_quote("site_00", APPROVED_CODE_IDENTITY, "default", nonce=nonce)
    server = Server(3, attestor=att)
    first = Update("site_00", np.zeros(3), 10, quote=quote)
    second = Update("site_00", np.zeros(3), 10, quote=quote)

    accepted, rejected = server._accept([first, second], {"site_00": nonce})
    assert [u.client_id for u in accepted] == ["site_00"]
    assert rejected[0].metadata["attestation_reason"] == "nonce_rejected"


def test_honest_clients_still_train_under_a_require_nonce_policy():
    sites, X_test, y_test = make_federated_parkinson(seed=22, n_sites=5)
    att = nonce_attestor(require_nonce=True)
    server = Server(X_test.shape[1], attestor=att)
    clients = [
        Client(s.site_id, s.X, s.y, attestor=att, seed=i) for i, s in enumerate(sites)
    ]
    server.fit(clients, X_test, y_test, rounds=12)

    assert server.challenge_response is True
    assert server.describe()["challenge_response"] is True
    assert all(r.n_rejected == 0 for r in server.history)
    assert server.history[-1].auc > 0.7
