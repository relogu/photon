"""Utility functions for strategies."""

from flwr.common.typing import NDArrays, Parameters
from flwr.server.strategy import FedAvg

from photon.strategy.fedadam import FedAdam
from photon.strategy.fedavg_eff import FedAvgEfficient
from photon.strategy.fedmom import FedMom
from photon.strategy.fednestorov import FedNesterov
from photon.strategy.fedyogi import FedYogi


def initialize_strategy(
    strategy: FedAvg,
    parameters: Parameters,
    momentum_vector: NDArrays | None = None,
    second_momentum_vector: NDArrays | None = None,
) -> None:
    """Initialize the strategy with the given parameters and momentum vectors.

    Parameters
    ----------
    strategy : FedAvg
        The federated learning strategy to initialize.
    parameters : Parameters
        The initial parameters for the strategy.
    momentum_vector : NDArrays or None, optional
        The first momentum vector, required for certain strategies.
    second_momentum_vector : NDArrays or None, optional
        The second momentum vector, required for certain strategies.

    """
    # NOTE: The strategy needs to hold a copy of the initial parameters, but we want
    # it to free the reference it holds as an attribute. Then, similarly to the
    # `initialize_parameters()` of FedAvg, we nullify such attribute
    if strategy.initial_parameters:
        strategy.initial_parameters = None
    # NOTE: Since we initialized the strategy object before creating the parameters,
    # we must assign to the strategy attributes the parameters we got from the
    # initialization
    if isinstance(strategy, FedNesterov | FedMom | FedYogi | FedAdam | FedAvgEfficient):
        strategy.parameters = parameters
    if isinstance(strategy, FedNesterov | FedMom | FedYogi | FedAdam):
        assert momentum_vector is not None, "Momentum vector must be initialized"
        strategy.momentum_vector = momentum_vector
    else:
        momentum_vector = None
    if isinstance(strategy, FedYogi | FedAdam):
        assert second_momentum_vector is not None, (
            "Second momentum vector must be initialized"
        )
        strategy.second_momentum_vector = second_momentum_vector
    else:
        second_momentum_vector = None
