"""Data-space poisoning: corrupt a client's local training set, not its update.

Update-space attacks (:mod:`trustfed.attack.poison`) assume the attacker can
write arbitrary numbers into the parameter vector. Data poisoning is the weaker,
more realistic threat: the attacker only controls the *labels* (or features) at
one site -- a mislabelled registry export, a compromised annotation pipeline, a
disgruntled data manager -- and the client then trains honestly on bad data.

That distinction matters for defense evaluation. Data poisoning produces updates
whose norms look completely normal, so norm-based and distance-based rules see
much less signal than they do against a scaling attack.

Each attack has the signature ``attack(X, y, rng, **kwargs) -> (X, y)`` and must
not modify its inputs in place.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np

from trustfed.attack.errors import AttackError, UnknownAttackError

ArrayPair = Tuple[np.ndarray, np.ndarray]


def no_data_attack(
    X: np.ndarray, y: np.ndarray, rng: np.random.Generator, **_: object
) -> ArrayPair:
    """Return the data unchanged (control condition)."""
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def label_flip(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    *,
    fraction: float = 1.0,
    source: Optional[int] = None,
    target: Optional[int] = None,
    **_: object,
) -> ArrayPair:
    """Flip a fraction of this site's binary labels.

    Parameters
    ----------
    fraction:
        Fraction of eligible rows to flip, in ``[0, 1]``. ``1.0`` inverts every
        eligible label.
    source, target:
        If both are given, the attack is *targeted*: only rows labelled
        ``source`` are relabelled to ``target`` (e.g. hide rapid decliners by
        relabelling 1 -> 0). If either is ``None`` the attack is untargeted and
        flips ``y -> 1 - y``.

    Raises
    ------
    AttackError
        If ``fraction`` is outside ``[0, 1]`` or the labels are not binary.

    Notes
    -----
    Untargeted flipping of *all* labels is close to negating the honest gradient
    and is therefore easy to spot; partial flipping (``fraction`` around 0.3) is
    the harder and more realistic case.
    """
    if not 0.0 <= fraction <= 1.0:
        raise AttackError(f"fraction must be in [0, 1], got {fraction}")
    y_arr = np.asarray(y, dtype=float).copy()
    if y_arr.size and not np.all(np.isin(np.unique(y_arr), (0.0, 1.0))):
        raise AttackError("label_flip expects binary 0/1 labels")

    if source is not None and target is not None:
        eligible = np.flatnonzero(y_arr == float(source))
        new_value = float(target)
    else:
        eligible = np.arange(y_arr.shape[0])
        new_value = None  # signals "invert"

    if eligible.size == 0 or fraction == 0.0:
        return np.asarray(X, dtype=float), y_arr

    k = int(round(fraction * eligible.size))
    if k <= 0:
        return np.asarray(X, dtype=float), y_arr
    chosen = rng.choice(eligible, size=min(k, eligible.size), replace=False)
    if new_value is None:
        y_arr[chosen] = 1.0 - y_arr[chosen]
    else:
        y_arr[chosen] = new_value
    return np.asarray(X, dtype=float), y_arr


def feature_corruption(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    *,
    sigma: float = 3.0,
    fraction: float = 1.0,
    **_: object,
) -> ArrayPair:
    """Add Gaussian noise to a fraction of this site's feature rows.

    Models a miscalibrated instrument or a broken ETL step rather than a
    deliberate adversary. Degrades the site's own update quality without making
    it an obvious outlier.
    """
    if not 0.0 <= fraction <= 1.0:
        raise AttackError(f"fraction must be in [0, 1], got {fraction}")
    if sigma < 0.0:
        raise AttackError(f"sigma must be >= 0, got {sigma}")
    X_arr = np.asarray(X, dtype=float).copy()
    n = X_arr.shape[0]
    k = int(round(fraction * n))
    if k > 0 and sigma > 0.0:
        rows = rng.choice(n, size=k, replace=False)
        X_arr[rows] += rng.normal(0.0, sigma, size=(k, X_arr.shape[1]))
    return X_arr, np.asarray(y, dtype=float)


DATA_ATTACKS: Dict[str, Callable[..., ArrayPair]] = {
    "none": no_data_attack,
    "label_flip": label_flip,
    "feature_corruption": feature_corruption,
}


def get_data_attack(name: str) -> Callable[..., ArrayPair]:
    """Look up a data-space attack by name.

    Raises
    ------
    UnknownAttackError
        If ``name`` is not a registered data-space attack.
    """
    try:
        return DATA_ATTACKS[name]
    except KeyError:
        raise UnknownAttackError(
            f"unknown data attack '{name}'. options: {sorted(DATA_ATTACKS)}"
        ) from None


def is_data_attack(name: str) -> bool:
    """Return ``True`` if ``name`` names a data-space (not update-space) attack."""
    return name in DATA_ATTACKS and name != "none"


__all__ = [
    "DATA_ATTACKS",
    "get_data_attack",
    "is_data_attack",
    "no_data_attack",
    "label_flip",
    "feature_corruption",
]
