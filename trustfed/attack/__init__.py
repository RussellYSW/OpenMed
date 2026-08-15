"""Attacks used to evaluate TrustFed's defenses.

Three families, in increasing order of threat-model strength:

* :mod:`trustfed.attack.data_poison` -- the attacker only corrupts local data.
* :mod:`trustfed.attack.poison` -- the attacker writes arbitrary values into its
  own update, but acts alone and ignores the defense.
* :mod:`trustfed.attack.adaptive` -- the attackers collude, observe the honest
  updates, and target the specific aggregation rule in use.

Everything here runs in-process on simulated arrays. It exists so that claims
about robustness are tested rather than asserted.
"""

from __future__ import annotations

from trustfed.attack.adaptive import (
    AdaptiveAdversary,
    adaptive_updates,
    alie_z,
    strategy_map,
)
from trustfed.attack.data_poison import (
    DATA_ATTACKS,
    feature_corruption,
    get_data_attack,
    is_data_attack,
    label_flip,
    no_data_attack,
)
from trustfed.attack.errors import AttackError, UnknownAttackError
from trustfed.attack.poison import (
    ATTACKS,
    attack_name,
    gaussian_attack,
    get_attack,
    model_replacement_attack,
    no_attack,
    scaling_attack,
    sign_flip_attack,
)

__all__ = [
    "ATTACKS",
    "get_attack",
    "attack_name",
    "no_attack",
    "gaussian_attack",
    "sign_flip_attack",
    "scaling_attack",
    "model_replacement_attack",
    "DATA_ATTACKS",
    "get_data_attack",
    "is_data_attack",
    "no_data_attack",
    "label_flip",
    "feature_corruption",
    "AdaptiveAdversary",
    "adaptive_updates",
    "alie_z",
    "strategy_map",
    "AttackError",
    "UnknownAttackError",
]
