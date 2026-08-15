"""Typed exceptions raised by :mod:`trustfed.attack`."""

from __future__ import annotations


class AttackError(ValueError):
    """Raised when an attack is misconfigured or requested under a wrong name."""


class UnknownAttackError(AttackError, KeyError):
    """Raised when an attack name is not in the registry."""

    def __str__(self) -> str:  # KeyError's repr would quote the message
        return self.args[0] if self.args else ""


__all__ = ["AttackError", "UnknownAttackError"]
