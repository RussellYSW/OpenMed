"""Update-poisoning attacks used to exercise the robust aggregators.

These model the *residual* threat after attestation: a client that runs approved
code but whose data is corrupted, whose host is compromised, or which suffers a
Byzantine hardware/network fault, and therefore submits a harmful parameter
vector. Each attack maps an honest parameter vector to a malicious one.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def no_attack(theta: np.ndarray, rng: np.random.Generator, **_) -> np.ndarray:
    return np.asarray(theta, dtype=float)


def gaussian_attack(
    theta: np.ndarray, rng: np.random.Generator, sigma: float = 12.0, **_
) -> np.ndarray:
    """Submit large-variance random noise (a crude but effective Byzantine
    fault that drags the mean around)."""
    return rng.normal(0.0, sigma, size=np.asarray(theta).shape)


def sign_flip_attack(
    theta: np.ndarray, rng: np.random.Generator, scale: float = 4.0, **_
) -> np.ndarray:
    """Negate and amplify the honest update, pulling the global model the wrong
    way -- a classic gradient-inversion / model-poisoning attack."""
    return -scale * np.asarray(theta, dtype=float)


def scaling_attack(
    theta: np.ndarray, rng: np.random.Generator, scale: float = 20.0, **_
) -> np.ndarray:
    """Boost the honest update's magnitude so it dominates a naive average
    (model-replacement style)."""
    return scale * np.asarray(theta, dtype=float)


ATTACKS: dict[str, Callable] = {
    "none": no_attack,
    "gaussian": gaussian_attack,
    "sign_flip": sign_flip_attack,
    "scaling": scaling_attack,
}


def get_attack(name: str) -> Callable:
    try:
        return ATTACKS[name]
    except KeyError:
        raise ValueError(f"unknown attack '{name}'. options: {sorted(ATTACKS)}")
