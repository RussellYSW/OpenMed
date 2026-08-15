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


# --------------------------------------------------------------------------
# Synthetic cohort generator, client selection, and adaptive attacks.
# --------------------------------------------------------------------------

import pytest

from trustfed.aggregation.robust import coordinate_median, get_aggregator
from trustfed.aggregation.tolerance import recommended_trim_beta
from trustfed.attack.adaptive import AdaptiveAdversary, alie_z
from trustfed.attack.data_poison import get_data_attack, label_flip
from trustfed.data import CohortSpec, DataError, make_cohort
from trustfed.federated.errors import ClientConfigError, NoAcceptedUpdatesError
from trustfed.federated.history import ClientHistory, RoundObservation
from trustfed.federated.selection import (
    LossBasedSelector,
    RandomSelector,
    ReputationSelector,
    get_selector,
)


# -- data generator ---------------------------------------------------------


def test_cohort_generation_is_deterministic_given_a_seed():
    a = make_cohort(CohortSpec(seed=11))
    b = make_cohort(CohortSpec(seed=11))
    assert np.array_equal(a.sites[0].X, b.sites[0].X)
    assert np.array_equal(a.y_test, b.y_test)
    c = make_cohort(CohortSpec(seed=12))
    assert not np.array_equal(a.sites[0].X, c.sites[0].X)


def test_label_skew_produces_non_iid_site_prevalences():
    skewed = make_cohort(CohortSpec(seed=2, label_skew=1.0, n_sites=6))
    flat = make_cohort(CohortSpec(seed=2, label_skew=0.0, n_sites=6))
    assert np.std(skewed.site_prevalences()) > np.std(flat.site_prevalences())
    assert max(skewed.site_prevalences()) - min(skewed.site_prevalences()) > 0.2


def test_feature_shift_moves_site_feature_means():
    shifted = make_cohort(CohortSpec(seed=3, feature_shift=1.5, n_sites=5))
    means = np.stack([s.X.mean(axis=0) for s in shifted.sites])
    assert np.mean(np.std(means, axis=0)) > 0.5
    iid = make_cohort(CohortSpec(seed=3, feature_shift=0.0, n_sites=5))
    iid_means = np.stack([s.X.mean(axis=0) for s in iid.sites])
    assert np.mean(np.std(iid_means, axis=0)) < np.mean(np.std(means, axis=0))


def test_cohort_has_both_classes_and_subgroups_everywhere():
    cohort = make_cohort(CohortSpec(seed=4, n_sites=6))
    for site in cohort.sites:
        assert 0.0 < site.prevalence < 1.0
        assert set(np.unique(site.groups)) <= {0, 1}
    assert set(np.unique(cohort.test_groups)) == {0, 1}


def test_invalid_cohort_spec_raises_data_error():
    with pytest.raises(DataError):
        make_cohort(CohortSpec(n_sites=0))
    with pytest.raises(DataError):
        make_cohort(CohortSpec(label_skew=5.0))
    with pytest.raises(DataError):
        make_cohort(seed=1, no_such_field=3)


# -- client-side attacks ----------------------------------------------------


def test_label_flip_corrupts_only_the_requested_fraction():
    rng = np.random.default_rng(0)
    y = np.array([0.0, 1.0] * 50)
    _, flipped = label_flip(np.zeros((100, 2)), y, rng, fraction=0.3)
    assert 25 <= int(np.sum(flipped != y)) <= 35


def test_targeted_label_flip_only_touches_the_source_class():
    rng = np.random.default_rng(1)
    y = np.array([0.0, 1.0] * 50)
    _, flipped = label_flip(
        np.zeros((100, 2)), y, rng, fraction=1.0, source=1, target=0
    )
    assert flipped.sum() == 0
    assert np.all(flipped[y == 0] == 0.0)


def test_data_poisoned_site_keeps_a_normal_update_norm():
    # The point of data poisoning: it is invisible to a norm-based screen.
    sites, X_test, y_test = make_federated_parkinson(seed=6, n_sites=4)
    theta0 = np.zeros(X_test.shape[1] + 1)
    honest = Client("h", sites[0].X, sites[0].y, seed=1).train(theta0)
    poisoned = Client(
        "p",
        sites[0].X,
        sites[0].y,
        malicious=True,
        data_attack=get_data_attack("label_flip"),
        data_attack_kwargs={"fraction": 1.0},
        seed=1,
    ).train(theta0)
    assert not np.allclose(honest.theta, poisoned.theta)
    ratio = np.linalg.norm(poisoned.theta) / max(np.linalg.norm(honest.theta), 1e-12)
    assert 0.2 < ratio < 5.0


def test_arming_an_attack_without_marking_the_client_malicious_is_refused():
    sites, _, _ = make_federated_parkinson(seed=7, n_sites=3)
    with pytest.raises(ClientConfigError):
        Client("x", sites[0].X, sites[0].y, attack=get_attack("scaling"))


def test_client_rejects_mismatched_X_and_y():
    with pytest.raises(ClientConfigError):
        Client("x", np.zeros((10, 3)), np.zeros(9))


# -- adaptive adversary -----------------------------------------------------


def test_adaptive_adversary_picks_a_strategy_per_aggregator():
    assert AdaptiveAdversary("fedavg").resolved_strategy == "mean_shift"
    assert AdaptiveAdversary("multi_krum").resolved_strategy == "krum_search"
    assert AdaptiveAdversary("norm_clip").resolved_strategy == "clip_boundary"
    # An unknown rule must still produce a usable strategy, not an error.
    assert AdaptiveAdversary("some_new_rule").resolved_strategy


def test_adaptive_mean_shift_flips_the_sign_of_the_average():
    honest = [np.full(4, 1.0) for _ in range(6)]
    mal = AdaptiveAdversary("fedavg", seed=0).craft(honest, 2)
    combined = np.mean(np.stack(honest + mal), axis=0)
    assert np.all(combined < 0)


def test_adaptive_min_max_stays_inside_the_honest_spread():
    rng = np.random.default_rng(5)
    honest = [rng.normal(1.0, 0.2, size=6) for _ in range(8)]
    mal = AdaptiveAdversary("median", seed=0).craft(honest, 2)[0]
    U = np.stack(honest)
    budget = max(
        np.linalg.norm(a - b) for a in U for b in U
    )
    assert max(np.linalg.norm(mal - h) for h in honest) <= budget + 1e-6
    assert not np.allclose(mal, np.mean(U, axis=0))


def test_adaptive_attack_collapses_fedavg_but_not_multi_krum():
    sites, X_test, y_test = make_federated_parkinson(
        n_sites=8, samples_per_site=150, test_size=800, seed=9
    )
    mal_ids = [s.site_id for s in sites[-3:]]

    def final_auc(name, adversary):
        server = Server(
            X_test.shape[1],
            aggregator=get_aggregator(name),
            n_byzantine=3,
            adversary=adversary,
            byzantine_ids=mal_ids if adversary else (),
        )
        clients = [Client(s.site_id, s.X, s.y, seed=i) for i, s in enumerate(sites)]
        server.fit(clients, X_test, y_test, rounds=20)
        return server.history[-1].auc

    for name, max_damage in (("fedavg", None), ("multi_krum", 0.05)):
        clean = final_auc(name, None)
        attacked = final_auc(name, AdaptiveAdversary(target_aggregator=name, seed=2))
        damage = clean - attacked
        if max_damage is None:
            assert damage > 0.10, f"{name}: adaptive attack did nothing ({damage:.3f})"
        else:
            assert damage < max_damage, f"{name}: lost {damage:.3f} AUC"


def test_alie_z_is_zero_when_no_stealthy_deviation_exists():
    # n=8, f=2: the median leaves the attacker no room. Honest reporting of a
    # negative result matters more than a dramatic number.
    assert alie_z(8, 2) == pytest.approx(0.0, abs=1e-9)
    assert alie_z(50, 24) > 1.0


# -- selection and reputation ----------------------------------------------


def test_random_selector_is_deterministic_and_respects_the_fraction():
    ids = [f"site_{i:02d}" for i in range(10)]
    sel = RandomSelector(0.5, seed=7)
    first = sel.select(ids, round_index=1)
    assert len(first) == 5
    assert first == sel.select(ids, round_index=1)
    assert first != sel.select(ids, round_index=2)


def test_loss_selector_measures_unseen_clients_first():
    ids = [f"c{i}" for i in range(6)]
    hist = ClientHistory()
    hist.observe_round([RoundObservation("c0", True, loss=0.9, deviation=0.1)])
    chosen = LossBasedSelector(0.5, seed=1).select(ids, round_index=1, history=hist)
    assert "c0" not in chosen  # the five unseen clients fill the cohort first


def test_reputation_falls_for_clients_that_deviate_and_are_rejected():
    hist = ClientHistory(alpha=0.5)
    for _ in range(5):
        hist.observe_round(
            [
                RoundObservation("good", True, deviation=0.1),
                RoundObservation("far", True, deviation=5.0),
                RoundObservation("rejected", False),
            ]
        )
    assert hist.reputation("good") > hist.reputation("far")
    assert hist.reputation("far") > hist.reputation("rejected")
    assert hist.ranking()[0] == "good"


def test_reputation_selector_favours_high_reputation_clients():
    ids = ["good", "bad"]
    hist = ClientHistory(alpha=0.9)
    for _ in range(10):
        hist.observe_round(
            [
                RoundObservation("good", True, deviation=0.05),
                RoundObservation("bad", False),
            ]
        )
    sel = ReputationSelector(0.5, seed=3, exploration=0.01)
    picks = [sel.select(ids, round_index=r, history=hist)[0] for r in range(40)]
    assert picks.count("good") > picks.count("bad")
    weights = sel.weights(ids, hist)
    assert weights[0] > weights[1]


def test_reputation_recovers_when_a_client_behaves_again():
    hist = ClientHistory(alpha=0.5)
    for _ in range(3):
        hist.observe_round([RoundObservation("c", False)])
    low = hist.reputation("c")
    for _ in range(6):
        hist.observe_round([RoundObservation("c", True, deviation=0.0)])
    assert hist.reputation("c") > low


def test_selector_registry_and_bad_names():
    assert isinstance(get_selector("reputation", fraction=0.5), ReputationSelector)
    with pytest.raises(Exception):
        get_selector("nope")


def test_server_runs_with_partial_participation():
    sites, X_test, y_test = make_federated_parkinson(seed=8, n_sites=8)
    server = Server(
        X_test.shape[1],
        aggregator=coordinate_median,
        selector=get_selector("random", fraction=0.5, seed=1),
    )
    clients = [Client(s.site_id, s.X, s.y, seed=i) for i, s in enumerate(sites)]
    server.fit(clients, X_test, y_test, rounds=15)
    assert all(len(r.selected_ids) == 4 for r in server.history)
    assert server.history[-1].auc > 0.7
    assert len(server.client_history.records) == 8


def test_server_raises_when_every_update_is_rejected():
    sites, X_test, y_test = make_federated_parkinson(seed=10, n_sites=4)
    attestor = MockSoftwareAttestor(ROOT_KEY, {APPROVED})
    clients = [
        Client(
            s.site_id,
            s.X,
            s.y,
            attestor=attestor,
            code_identity="trustfed-client@TAMPERED",
            seed=i,
        )
        for i, s in enumerate(sites)
    ]
    server = Server(X_test.shape[1], attestor=attestor)
    with pytest.raises(NoAcceptedUpdatesError):
        server.fit(clients, X_test, y_test, rounds=1)


def test_trimmed_beta_recommendation_keeps_the_model_learning_under_attack():
    sites, X_test, y_test = make_federated_parkinson(seed=13, n_sites=8)
    server = Server(
        X_test.shape[1],
        aggregator=get_aggregator("trimmed_mean"),
        n_byzantine=2,
        trimmed_beta=recommended_trim_beta(8, 2),
    )
    clients = build(sites, n_malicious=2, attack="scaling")
    server.fit(clients, X_test, y_test, rounds=20)
    assert server.history[-1].auc > 0.7
