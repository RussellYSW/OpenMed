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
