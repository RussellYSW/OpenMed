"""The federated learning plane: clients, coordinator, and client selection.

``Server`` runs the round loop, ``Client`` holds private data and produces
attested updates, and :mod:`trustfed.federated.selection` decides who takes part
in each round. Behavioural evidence about clients accumulates in
:class:`trustfed.federated.history.ClientHistory`.
"""

from __future__ import annotations

from trustfed.federated.client import APPROVED_CODE_IDENTITY, Client, Update
from trustfed.federated.errors import (
    ClientConfigError,
    FederatedError,
    NoAcceptedUpdatesError,
    SelectionError,
)
from trustfed.federated.history import ClientHistory, ClientRecord, RoundObservation
from trustfed.federated.selection import (
    SELECTORS,
    AllClientsSelector,
    LossBasedSelector,
    RandomSelector,
    ReputationSelector,
    Selector,
    get_selector,
)
from trustfed.federated.server import RoundResult, Server

__all__ = [
    "Client",
    "Update",
    "APPROVED_CODE_IDENTITY",
    "Server",
    "RoundResult",
    "Selector",
    "AllClientsSelector",
    "RandomSelector",
    "LossBasedSelector",
    "ReputationSelector",
    "SELECTORS",
    "get_selector",
    "ClientHistory",
    "ClientRecord",
    "RoundObservation",
    "FederatedError",
    "NoAcceptedUpdatesError",
    "SelectionError",
    "ClientConfigError",
]
