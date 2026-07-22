import numpy as np

from trustfed.aggregation.robust import fedavg, multi_krum
from trustfed.attack.poison import get_attack
from trustfed.attestation.attestor import MockSoftwareAttestor, measure_code
from trustfed.data.synthetic import make_federated_parkinson
from trustfed.federated.client import APPROVED_CODE_IDENTITY, Client
from trustfed.federated.server import Server

ROOT_KEY = b"integration-test-key"
APPROVED = measure_code(APPROVED_CODE_IDENTITY)


def build(sites, *, n_malicious=0, attack="none", attestor=None, tamper=False):
    atk = get_attack(attack)
    clients = []
    for i, s in enumerate(sites):
        is_mal = n_malicious > 0 and i >= len(sites) - n_malicious
        code = "trustfed-client@TAMPERED" if (is_mal and tamper) else APPROVED_CODE_IDENTITY
        clients.append(
            Client(s.site_id, s.X, s.y, attestor=attestor, code_identity=code,
                   malicious=is_mal, attack=atk if is_mal else None, seed=10 + i)
        )
    return clients


def test_clean_training_learns():
    sites, X_test, y_test = make_federated_parkinson(seed=1)
    server = Server(X_test.shape[1], aggregator=fedavg)
    server.fit(build(sites), X_test, y_test, rounds=25)
    assert server.history[-1].auc > 0.75


def test_multikrum_beats_fedavg_under_attack():
    sites, X_test, y_test = make_federated_parkinson(seed=2)
    n_features = X_test.shape[1]

    naive = Server(n_features, aggregator=fedavg, n_byzantine=2)
    naive.fit(build(sites, n_malicious=2, attack="sign_flip"), X_test, y_test, rounds=25)

    robust = Server(n_features, aggregator=multi_krum, n_byzantine=2)
    robust.fit(build(sites, n_malicious=2, attack="sign_flip"), X_test, y_test, rounds=25)

    assert robust.history[-1].auc > naive.history[-1].auc + 0.1


def test_attestation_rejects_tampered_sites():
    sites, X_test, y_test = make_federated_parkinson(seed=3)
    attestor = MockSoftwareAttestor(ROOT_KEY, {APPROVED})
    server = Server(X_test.shape[1], aggregator=fedavg, attestor=attestor, n_byzantine=2)
    clients = build(sites, n_malicious=2, attack="scaling", attestor=attestor, tamper=True)
    server.fit(clients, X_test, y_test, rounds=20)

    last = server.history[-1]
    assert last.n_rejected == 2  # both tampered sites excluded
    assert last.auc > 0.75       # clean model despite the attack
