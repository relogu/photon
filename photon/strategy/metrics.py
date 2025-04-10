"""The module provides utility functions and classes for computing metrics in FL.

It includes:
- Exponential Moving Average (EMA) with de-biasing.
- Server-side metric callbacks for collecting and aggregating metrics.
- Noise scale estimation in federated learning.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any

import numpy as np
from flwr.common import NDArrays, parameters_to_ndarrays

from photon.strategy.strategy_with_cfg import FedAvgWithConfig

if TYPE_CHECKING:
    from photon.conf.base_schema import BaseConfig


def ema_with_debias(
    avg: float | None,
    beta: float,
    y_i: float,
    i: int,
) -> tuple[float, float]:
    """Compute the exponential moving average (EMA) with de-biasing.

    Parameters
    ----------
    avg : float | None
        The current average value.
    beta : float
        The smoothing factor for the EMA.
    y_i : float
        The new value to include in the EMA.
    i : int
        The current step count.

    Returns
    -------
    tuple[float, float]
        The updated average and the de-biased average.

    """
    if avg is None:
        avg = 0.0
    avg = beta * avg + (1 - beta) * y_i
    return avg, avg / (1 - beta ** (i + 1))


class ServerMetricCallback(ABC):
    """Callback for collecting metrics on the server side."""

    def __init__(
        self,
        strategy: FedAvgWithConfig,
        metrics: dict[str, Any],
        server_round: int,
    ) -> None:
        """Initialize the ServerMetricCallback object.

        Parameters
        ----------
        metrics : dict[str, Any]
            Dictionary to store the collected metrics.
        strategy : FedAvgWithConfig
            The federated learning strategy.
        server_round : int
            The current server round.

        Raises
        ------
        ValueError
            If the configuration object is missing.

        """
        self.metrics = metrics
        if strategy.cfg is None:
            msg = "The configuration object is missing."
            raise ValueError(msg)
        self.cfg: BaseConfig = strategy.cfg
        self.strategy = strategy
        self.server_round = server_round

    @abstractmethod
    def process_per_client_results(
        self,
        update_iterator: Iterator[np.ndarray] | Iterable[np.ndarray],
        num_samples: int,
    ) -> Iterator[np.ndarray]:
        """Add per-client metrics to the metrics dictionary.

        Parameters
        ----------
        update_iterator : Iterator[np.ndarray]
            The client's updated model.
        num_samples : int
            The number of samples in the client's dataset.

        Yields
        ------
        Iterator[np.ndarray]
            The original model after gathering metrics.

        """

    @abstractmethod
    def round_end(self, fedavg_result: NDArrays | None) -> None:
        """Add parameters metrics to the metrics dictionary.

        Parameters
        ----------
        current_round : int
            The current round number.
        fedavg_result : NDArrays | None
            The aggregated model from the clients.

        """


class FedSimpleNoiseScale(ServerMetricCallback):
    """Callback for estimating the noise scale in federated learning."""

    def __init__(
        self,
        strategy: FedAvgWithConfig,
        metrics: dict[str, Any],
        server_round: int,
    ) -> None:
        """Initialize the FedSimpleNoiseScale object.

        Parameters
        ----------
        strategy : FedAvgWithConfig
            The federated learning strategy.
        metrics : dict[str, Any]
            Dictionary to store the collected metrics.
        server_round : int
            The current server round

        """
        super().__init__(strategy, metrics, server_round)
        self.summed_grads_squares: float = 0.0
        self.beta = self.cfg.fl.noise_scale_beta
        self.running_trace_estimate = 0.0
        self.running_squared_gradients_estimate = 0.0
        self.running_noise_scale = 0.0
        self.counter = 0
        self.n_clients = 0
        self.old_parameters = parameters_to_ndarrays(self.strategy.parameters)

    def process_per_client_results(
        self,
        update_iterator: Iterator[np.ndarray] | Iterable[np.ndarray],
        num_samples: int,  # noqa: ARG002
    ) -> Iterator[np.ndarray]:
        """Add per-client metrics to the metrics dictionary.

        Parameters
        ----------
        num_samples : int
            The number of samples in the client's dataset.
        update_iterator : Iterator[np.ndarray]
            The client's updated model.
        num_samples : int
            The number of samples in the client's dataset

        Yields
        ------
        Iterator[np.ndarray]
            The original model after gathering metrics.

        """
        for x, y in zip(self.old_parameters, update_iterator, strict=True):
            # Sum the square of the gradients for each client
            self.summed_grads_squares += ((x - y) ** 2).sum()
            yield y

        # Increase the number of clients
        self.n_clients += 1

    def round_end(
        self,
        fedavg_result: NDArrays | None,
    ) -> None:
        """Add parameters metrics to the metrics dictionary.

        Parameters
        ----------
        current_round : int
            The current round number.
        fedavg_result : NDArrays | None
            The aggregated pseudo-gradient from the clients.

        """
        if fedavg_result is None or self.summed_grads_squares == 0.0:
            return

        pseudo_gradient = [
            x - y for x, y in zip(self.old_parameters, fedavg_result, strict=True)
        ]

        # Increase the counter for the exponentially moving average
        self.counter += 1

        # Compute the L2 norm squared of the pseudo-gradients
        g_big_l2norm_squared = sum((x**2).sum() for x in pseudo_gradient)
        # Average the sum of the gradients squared
        self.summed_grads_squares /= self.n_clients
        # Set the big and small batch sizes. Specifically, the small batch size is one,
        # i.e., one client, and the big batch size is the total number of clients in the
        # current round. As such, the simple noise scale will be represented as if the
        # batch size was one.
        b_small = 1
        b_big = self.n_clients

        # Estimate the trace of the covariance matrix of the gradients
        trace_estimate = (self.summed_grads_squares - g_big_l2norm_squared) / (
            (1 / b_small) - (1 / b_big)
        )
        # Estimate the squared norm of the gradients
        squared_gradients_estimate = (
            b_big * g_big_l2norm_squared - b_small * self.summed_grads_squares
        ) / (b_big - b_small)

        # Compute exponential moving averages
        self.running_trace_estimate, scale = ema_with_debias(
            self.running_trace_estimate,
            self.beta,
            trace_estimate,
            self.counter,
        )
        self.running_squared_gradients_estimate, noise = ema_with_debias(
            self.running_squared_gradients_estimate,
            self.beta,
            squared_gradients_estimate,
            self.counter,
        )
        self.running_noise_scale, noise_scale_ema_bias = ema_with_debias(
            self.running_noise_scale,
            self.beta,
            trace_estimate / squared_gradients_estimate,
            self.counter,
        )
        # Compute the noise scale
        noise_scale_with_emas = scale / noise
        noise_scale = trace_estimate / squared_gradients_estimate

        # Log the current value of the noise scale
        self.metrics |= {
            "noise_scale/b_small": b_small,
            "noise_scale/b_big": b_big,
            "noise_scale/g_small_l2norm_squared": self.summed_grads_squares,
            "noise_scale/g_big_l2norm_squared": g_big_l2norm_squared,
            "noise_scale/trace_estimate": trace_estimate,
            "noise_scale/squared_gradients_estimate": squared_gradients_estimate,
            "noise_scale/noise_scale_with_emas": noise_scale_with_emas,
            "noise_scale/noise_scale_ema": self.running_noise_scale,
            "noise_scale/noise_scale_ema_bias": noise_scale_ema_bias,
            "noise_scale/noise_scale_raw": noise_scale,
        }

        # NOTE: Empty buffer and variables
        self.summed_grads_squares = 0
        self.n_clients = 0
