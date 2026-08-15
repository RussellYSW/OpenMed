import numpy as np

from trustfed.aggregation.robust import (
    coordinate_median,
    fedavg,
    krum,
    multi_krum,
    trimmed_mean,
)


def honest_and_poisoned():
    rng = np.random.default_rng(0)
    honest = [rng.normal(1.0, 0.05, size=8) for _ in range(7)]
    poison = [np.full(8, 50.0), np.full(8, 50.0)]  # 2 coordinated Byzantine
    return honest, poison


def test_fedavg_is_dragged_by_outliers():
    honest, poison = honest_and_poisoned()
    agg = fedavg(honest + poison)
    # A single huge update pulls the mean far from the honest center (~1.0).
    assert np.linalg.norm(agg - 1.0) > 1.0


def test_median_resists_poisoning():
    honest, poison = honest_and_poisoned()
    agg = coordinate_median(honest + poison)
    assert np.linalg.norm(agg - 1.0) < 0.2


def test_trimmed_mean_resists_poisoning():
    honest, poison = honest_and_poisoned()
    # Trim 2 from each end to cover the 2 Byzantine updates.
    agg = trimmed_mean(honest + poison, beta=2.5 / 9)
    assert np.linalg.norm(agg - 1.0) < 0.2


def test_krum_selects_honest_update():
    honest, poison = honest_and_poisoned()
    agg = krum(honest + poison, n_byzantine=2)
    assert np.linalg.norm(agg - 1.0) < 0.5


def test_multi_krum_resists_poisoning():
    honest, poison = honest_and_poisoned()
    agg = multi_krum(honest + poison, n_byzantine=2)
    assert np.linalg.norm(agg - 1.0) < 0.3


def test_krum_falls_back_when_too_few_clients():
    # n <= 2f + 2 -> should fall back to median rather than error.
    updates = [np.ones(4), np.full(4, 100.0)]
    agg = krum(updates, n_byzantine=1)
    assert np.allclose(agg, np.median(np.stack(updates), axis=0))


# --------------------------------------------------------------------------
# Bounded-influence rules, tolerance bookkeeping and failure paths.
# --------------------------------------------------------------------------

import pytest

from trustfed.aggregation import (
    AggregationError,
    UnknownAggregatorError,
    centered_clipping,
    get_aggregator,
    max_byzantine,
    norm_clipped_mean,
    recommended_trim_beta,
    tolerance_bound,
    tolerance_table,
)


def test_norm_clip_resists_poisoning():
    honest, poison = honest_and_poisoned()
    agg = norm_clipped_mean(honest + poison)
    assert np.linalg.norm(agg - 1.0) < 0.3


def test_norm_clip_bounds_attacker_influence_by_the_documented_amount():
    # The documented bound is ||agg - agg_without_attacker|| <= 2*tau*f/n.
    rng = np.random.default_rng(3)
    honest = [rng.normal(1.0, 0.05, size=8) for _ in range(9)]
    tau = 0.5
    clean = norm_clipped_mean(honest + [honest[0].copy()], clip=tau)
    attacked = norm_clipped_mean(honest + [np.full(8, 1e6)], clip=tau)
    n, f = 10, 1
    assert np.linalg.norm(attacked - clean) <= 2.0 * tau * f / n + 1e-9


def test_norm_clip_leaves_inlying_updates_untouched():
    # With a radius above every deviation, the rule must reduce to the mean.
    rng = np.random.default_rng(4)
    updates = [rng.normal(0.0, 0.01, size=5) for _ in range(6)]
    assert np.allclose(
        norm_clipped_mean(updates, clip=100.0), np.mean(np.stack(updates), axis=0)
    )


def test_centered_clipping_resists_poisoning_and_uses_reference():
    honest, poison = honest_and_poisoned()
    agg = centered_clipping(honest + poison, reference=np.ones(8), iters=3)
    assert np.linalg.norm(agg - 1.0) < 0.3


def test_centered_clipping_rejects_zero_iterations():
    with pytest.raises(AggregationError):
        centered_clipping([np.ones(3), np.zeros(3)], iters=0)


def test_aggregators_reject_nan_updates():
    # A single nan would otherwise poison every coordinate of a mean.
    bad = [np.ones(4), np.array([np.nan, 1.0, 1.0, 1.0])]
    for rule in (fedavg, coordinate_median, trimmed_mean, krum, norm_clipped_mean):
        with pytest.raises(AggregationError):
            rule(bad)


def test_aggregators_reject_ragged_and_empty_input():
    with pytest.raises(AggregationError):
        fedavg([np.ones(3), np.ones(4)])
    with pytest.raises(AggregationError):
        fedavg([])


def test_fedavg_rejects_negative_weights():
    # Negative weights would let one client subtract another's contribution.
    with pytest.raises(AggregationError):
        fedavg([np.ones(3), np.zeros(3)], weights=[-1.0, 2.0])


def test_fedavg_weights_are_honoured():
    agg = fedavg([np.zeros(2), np.ones(2)], weights=[3.0, 1.0])
    assert np.allclose(agg, [0.25, 0.25])


def test_get_aggregator_unknown_name_lists_options():
    with pytest.raises(UnknownAggregatorError) as excinfo:
        get_aggregator("definitely_not_a_rule")
    assert "multi_krum" in str(excinfo.value)


def test_every_registered_rule_has_a_documented_tolerance_bound():
    from trustfed.aggregation.robust import AGGREGATORS

    for name in AGGREGATORS:
        bound = tolerance_bound(name)
        assert bound.condition and bound.note
    assert "breakdown fraction" in tolerance_table()


def test_documented_tolerance_matches_krum_precondition():
    # Krum needs n > 2f + 2, so at n=9 the largest tolerated f is 3.
    assert max_byzantine("krum", 9) == 3
    assert max_byzantine("fedavg", 9) == 0
    assert max_byzantine("median", 9) == 4


def test_recommended_trim_beta_covers_the_attackers():
    beta = recommended_trim_beta(10, 3)
    assert int(np.floor(beta * 10)) >= 3
    assert beta < 0.5


def test_krum_prefers_the_dense_honest_cluster_over_a_lone_outlier():
    honest = [np.full(4, 1.0) + 0.01 * i for i in range(5)]
    agg = krum(honest + [np.full(4, 900.0)], n_byzantine=1)
    assert np.linalg.norm(agg - 1.0) < 0.5
