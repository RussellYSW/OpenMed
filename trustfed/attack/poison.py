"""Update-poisoning attacks used to exercise the robust aggregators.

These model the *residual* threat after attestation: a client that runs approved
code but whose data is corrupted, whose host is compromised, or which suffers a
Byzantine hardware/network fault, and therefore submits a harmful parameter
vector. Each attack maps an honest parameter vector to a malicious one and has
the signature ``attack(theta, rng, **kwargs) -> np.ndarray``; unused keyword
arguments are ignored so the client can pass context (such as ``global_theta``)
uniformly.

The attacks here are *non-adaptive*: they do not look at what the other clients
submitted or at which aggregation rule is in use. Adaptive, aggregator-aware
attacks live in :mod:`trustfed.attack.adaptive`; data-space attacks (label
flipping) live in :mod:`trustfed.attack.data_poison`.

These implementations exist to test defenses in-process. They are not exploit
tooling: nothing here reaches the network, and every function operates on arrays
already inside the simulation.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from trustfed.attack.errors import UnknownAttackError


def no_attack(theta: np.ndarray, rng: np.random.Generator, **_: object) -> np.ndarray:
    """Return the honest update unchanged (control condition)."""
    return np.asarray(theta, dtype=float)


def gaussian_attack(
    theta: np.ndarray,
    rng: np.random.Generator,
    sigma: float = 12.0,
    **_: object,
) -> np.ndarray:
    """Submit large-variance random noise.

    A crude but effective Byzantine fault: it drags the mean around and has an
    enormous norm, so it is also the easiest attack for any robust rule to
    reject. Useful mainly as a lower bound on defense quality.
    """
    return rng.normal(0.0, float(sigma), size=np.asarray(theta).shape)


def sign_flip_attack(
    theta: np.ndarray,
    rng: np.random.Generator,
    scale: float = 4.0,
    **_: object,
) -> np.ndarray:
    """Negate and amplify the honest update.

    Pulls the global model in exactly the wrong direction -- a classic
    gradient-inversion / model-poisoning attack. Amplifying by ``scale`` makes it
    strong against the mean but conspicuous to distance-based rules.
    """
    return -float(scale) * np.asarray(theta, dtype=float)


def scaling_attack(
    theta: np.ndarray,
    rng: np.random.Generator,
    scale: float = 20.0,
    **_: object,
) -> np.ndarray:
    """Boost the honest update's magnitude so it dominates a naive average."""
    return float(scale) * np.asarray(theta, dtype=float)


def model_replacement_attack(
    theta: np.ndarray,
    rng: np.random.Generator,
    *,
    global_theta: Optional[np.ndarray] = None,
    n_clients: int = 1,
    target: Optional[np.ndarray] = None,
    boost: Optional[float] = None,
    **_: object,
) -> np.ndarray:
    """Model replacement / "train-and-scale" (Bagdasaryan et al., AISTATS 2020).

    The attacker wants the *aggregate* to equal a chosen target model. Under an
    unweighted mean of ``n`` clients it therefore submits
    ``global + n * (target - global)``, so that averaging cancels the boost
    exactly and the global model is replaced in a single round.

    Parameters
    ----------
    theta:
        The attacker's honestly-trained update; used as the target when
        ``target`` is not given (i.e. "replace the model with my local optimum").
    global_theta:
        The global model broadcast this round. If ``None``, the attack degrades
        to a plain scaling attack, since the offset cannot be computed.
    n_clients:
        The attacker's estimate of the number of aggregated clients; this is the
        boost factor.
    target:
        Explicit target model to install.
    boost:
        Override the boost factor directly.

    Notes
    -----
    Effective against FedAvg, and against norm clipping only if the boosted
    update stays inside the clip radius -- which is precisely why bounded
    influence helps here.
    """
    t = np.asarray(theta, dtype=float)
    tgt = t if target is None else np.asarray(target, dtype=float)
    gamma = float(n_clients if boost is None else boost)
    if global_theta is None:
        return gamma * tgt
    g = np.asarray(global_theta, dtype=float)
    return g + gamma * (tgt - g)


ATTACKS: Dict[str, Callable[..., np.ndarray]] = {
    "none": no_attack,
    "gaussian": gaussian_attack,
    "sign_flip": sign_flip_attack,
    "scaling": scaling_attack,
    "model_replacement": model_replacement_attack,
}


def get_attack(name: str) -> Callable[..., np.ndarray]:
    """Look up an update-space attack by name.

    Raises
    ------
    UnknownAttackError
        If ``name`` is not an update-space attack. Data-space attacks (e.g.
        ``label_flip``) are resolved with
        :func:`trustfed.attack.data_poison.get_data_attack` instead; the error
        message says so.
    """
    try:
        return ATTACKS[name]
    except KeyError:
        from trustfed.attack.data_poison import DATA_ATTACKS

        if name in DATA_ATTACKS:
            raise UnknownAttackError(
                f"'{name}' is a data-space attack; use "
                f"trustfed.attack.data_poison.get_data_attack('{name}')"
            ) from None
        raise UnknownAttackError(
            f"unknown attack '{name}'. options: {sorted(ATTACKS)}"
        ) from None


def attack_name(fn: Callable[..., np.ndarray]) -> str:
    """Reverse lookup: registry name for an attack function."""
    for key, value in ATTACKS.items():
        if value is fn:
            return key
    return getattr(fn, "__name__", "custom")


__all__ = [
    "ATTACKS",
    "get_attack",
    "attack_name",
    "no_attack",
    "gaussian_attack",
    "sign_flip_attack",
    "scaling_attack",
    "model_replacement_attack",
]
