"""Federated Averaging with Nestorov Momentum strategy.

This implementation is based on FedMom from [this paper](https://arxiv.org/abs/2002.02090).
It can either partially aggregate updated model parameters as soon as they arrive and the compute
the averaged pseudo-gradient or partially aggregate pseudo-gradients while computing them.
The averaged pseudo-gradient is then used to update the global model parameters.
"""

from collections.abc import Callable, Iterable
from copy import deepcopy
from logging import INFO
from pathlib import Path

import numpy as np
from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    log,
    ndarray_to_bytes,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate

from photon.conf.base_schema import BaseConfig
from photon.strategy.aggregation import (
    aggregate_cumulative_average,
    parameters_to_ndarrays_gen,
)
from photon.strategy.constants import FIRST_MOMENTUM, MODEL_PARAMETERS
from photon.strategy.metrics import ServerMetricCallback
from photon.strategy.strategy_with_cfg import FedAvgWithConfig
from photon.utils import (
    l2_norm,
    sum_of_squares,
)


# flake8: noqa: E501
class FedMom(FedAvgWithConfig):
    """Federated Averaging with Nestorov Momentum strategy.

    Implementation based on FedMom from [this paper](https://arxiv.org/abs/2002.02090).

    Parameters
    ----------
    fraction_fit : float, optional
        Fraction of clients used during training. In case `min_fit_clients`
        is larger than `fraction_fit * available_clients`, `min_fit_clients`
        will still be sampled. Defaults to 1.0.
    fraction_evaluate : float, optional
        Fraction of clients used during validation. In case `min_evaluate_clients`
        is larger than `fraction_evaluate * available_clients`, `min_evaluate_clients`
        will still be sampled. Defaults to 1.0.
    min_fit_clients : int, optional
        Minimum number of clients used during training. Defaults to 2.
    min_evaluate_clients : int, optional
        Minimum number of clients used during validation. Defaults to 2.
    min_available_clients : int, optional
        Minimum number of total clients in the system. Defaults to 2.
    evaluate_fn : (
            Callable[
                [int, NDArrays, dict[str, Scalar]],
                tuple[float, dict[str, Scalar]] | None,
            ]
            | None
        )
        Optional function used for validation. Defaults to None.
    on_fit_config_fn : Callable[[int], dict[str, Scalar]], optional
        Function used to configure training. Defaults to None.
    on_evaluate_config_fn : Callable[[int], dict[str, Scalar]], optional
        Function used to configure validation. Defaults to None.
    accept_failures : bool, optional
        Whether or not accept rounds containing failures. Defaults to True.
    initial_parameters : Parameters, optional
        Initial global model parameters.
    fit_metrics_aggregation_fn : MetricsAggregationFn | None
        Metrics aggregation function, optional.
    evaluate_metrics_aggregation_fn : MetricsAggregationFn | None
        Metrics aggregation function, optional.
    seed : int, optional
        Seed for reproducibility. Defaults to 1337.
    server_learning_rate : float, optional
        Learning rate used by the server-side optimizer. Defaults to 0.7.
    server_momentum: float, optional
        Momentum coefficient used by the server-side optimizer. Defaults to 0.9.
    track_norms: bool, optional
        Flag for tracking the norms of the aggregated updates. Defaults to True.
    obtain_server_metrics_callback: ServerMetricCallback, optional
        Callback for collecting metrics on the server side. Defaults to None.
    track_inplace_aggregation: bool, optional
        Flag for tracking the difference between standard and in-place aggregation. Defaults to False.
    cfg : BaseConfig, optional
        Configuration object. Defaults to None.

    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes,line-too-long
    def __init__(  # noqa: PLR0913, D107
        self,
        *,
        initial_parameters: Parameters,
        saving_path: Path | None = None,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: (
            Callable[
                [int, NDArrays, dict[str, Scalar]],
                tuple[float, dict[str, Scalar]] | None,
            ]
            | None
        ) = None,
        on_fit_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        on_evaluate_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        accept_failures: bool = True,
        fit_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        evaluate_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        server_learning_rate: float = 0.7,
        server_momentum: float = 0.9,
        track_norms: bool = True,
        obtain_server_metrics_callback: type[ServerMetricCallback] | None = None,
        track_inplace_aggregation: bool = False,
        cfg: BaseConfig | None = None,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        if saving_path is None:
            saving_path = Path(Path.cwd())
        self.saving_path = saving_path

        # Default optimizer values
        self.server_learning_rate = server_learning_rate
        self.server_momentum = server_momentum

        # NOTE: This avoids translating between parameters and NDArrays every time. However, it incurs in a higher memory peak. We decided to go for the previous approach that uses a pointer to the parameters at the server.
        # ndarray_params = parameters_to_ndarrays(initial_parameters)  # noqa: ERA001
        self.parameters: Parameters = initial_parameters
        if self.parameters is not self.initial_parameters:
            msg = "The initial_parameters and parameters should be the same object"
            raise ValueError(msg)

        # Set state_keys variables for ease of uploading to S3
        self.state_keys = (
            MODEL_PARAMETERS,
            FIRST_MOMENTUM,
        )

        log(
            INFO,
            "Using Nesterov Momentum (FedMom) with server_learning_rate=%s and"
            " server_momentum=%s",
            self.server_learning_rate,
            self.server_momentum,
        )
        self.momentum_vector: NDArrays = deepcopy(
            parameters_to_ndarrays(initial_parameters),
        )

        self.track_norms = track_norms
        self.track_inplace_aggregation = track_inplace_aggregation
        self.cfg = cfg

        self.obtain_server_metrics_callback = obtain_server_metrics_callback

    def aggregate_fit(  # noqa: PLR0914, C901
        self,
        server_round: int,
        results: Iterable[tuple[ClientProxy, FitRes]],
        failures: Iterable[tuple[ClientProxy, FitRes] | BaseException],  # noqa: ARG002
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate fit results using weighted average.

        Parameters
        ----------
        server_round : int
            Current server round.
        results : Iterable[tuple[ClientProxy, FitRes]]
            Iterable of tuples containing the client and the fit result.
        failures : Iterable[tuple[ClientProxy, FitRes] | BaseException]
            Iterable of tuples containing the client and the fit result or an exception.

        Returns
        -------
        tuple[Parameters | None, dict[str, Scalar]]
            The aggregated parameters and the metrics.

        Raises
        ------
        ValueError
            When using server-side optimization, model needs to be initialized.

        """
        if self.parameters is None:  # type: ignore[reportUnnecessaryComparison]
            msg = "When using server-side optimization, model needs to be initialized."
            raise ValueError(msg)

        fit_metrics: list[tuple[int, dict[str, Scalar]]] = []

        def acc_metrics(
            result: tuple[ClientProxy, FitRes],
        ) -> tuple[ClientProxy, FitRes]:
            _, fit_res = result
            fit_metrics.append((fit_res.num_examples, fit_res.metrics))
            return result

        results = (acc_metrics(result) for result in results)

        results_cached: list[tuple[ClientProxy, FitRes]] = []

        if self.track_inplace_aggregation:
            results_cached = list(results)
            results = (val for val in results_cached)

        metrics_aggregated: dict[str, Scalar] = {}

        metrics_callback: ServerMetricCallback | None = (
            self.obtain_server_metrics_callback(
                self,
                metrics_aggregated,
                server_round,
            )
            if self.obtain_server_metrics_callback is not None
            else None
        )

        # Get the cumulative average of the results
        fedavg_result = aggregate_cumulative_average(
            results,
            metrics_callback=metrics_callback,
        )

        # Return None if no results were aggregated
        if fedavg_result is None:
            return None, {}

        # Initialize the metrics
        layerwise_l2_norms_pseudo_gradient: list[float] = []
        layerwise_l2_norms_momentum_vector: list[float] = []
        layerwise_l2_norms_fedavg_result: list[float] = []
        layerwise_l2_norms_model: list[float] = []
        # Loop over layer, apply the server optimizer and compute metrics
        for i, x in enumerate(parameters_to_ndarrays_gen(self.parameters)):
            # Layer i pseudo-gradient
            layer_pseudo_gradient = x - fedavg_result[i]

            # Using FedMom implementation
            # Layer i new momentum vector
            layer_new_momentum_vector = (
                x - self.server_learning_rate * layer_pseudo_gradient
            )
            # Layer i new values
            layer_fedavgm_result = (
                1 + self.server_momentum
            ) * layer_new_momentum_vector - self.server_momentum * self.momentum_vector[
                i
            ]
            # Assign new values to the parameters variable
            self.parameters.tensors[i] = ndarray_to_bytes(layer_fedavgm_result)
            # Assign new values to the momentum vector
            self.momentum_vector[i] = layer_new_momentum_vector
            # Metrics collection
            layerwise_l2_norms_pseudo_gradient.append(l2_norm([layer_pseudo_gradient]))
            layerwise_l2_norms_momentum_vector.append(
                l2_norm([layer_new_momentum_vector]),
            )
            layerwise_l2_norms_fedavg_result.append(l2_norm([x]))
            layerwise_l2_norms_model.append(l2_norm([layer_fedavgm_result]))

        if self.track_norms:
            metrics_aggregated |= {
                "server/l2_norm_pseudo_gradient": np.sqrt(
                    np.sum(np.square(layerwise_l2_norms_pseudo_gradient)),
                ),
                "server/l2_norm_momentum_vector": np.sqrt(
                    np.sum(np.square(layerwise_l2_norms_momentum_vector)),
                ),
                "server/l2_norm_fedavg_result": np.sqrt(
                    np.sum(np.square(layerwise_l2_norms_fedavg_result)),
                ),
                "server/l2_norm_model": np.sqrt(
                    np.sum(np.square(layerwise_l2_norms_model)),
                ),
            }
            for i, (a, b, c, d) in enumerate(
                zip(
                    layerwise_l2_norms_pseudo_gradient,
                    layerwise_l2_norms_momentum_vector,
                    layerwise_l2_norms_fedavg_result,
                    layerwise_l2_norms_model,
                    strict=True,
                ),
            ):
                metrics_aggregated |= {f"server/layer/{i}/l2_norm_pseudo_gradient": a}
                metrics_aggregated |= {f"server/layer/{i}/l2_norm_momentum_vector": b}
                metrics_aggregated |= {f"server/layer/{i}/l2_norm_fedavg_result": c}
                metrics_aggregated |= {f"server/layer/{i}/l2_norm_model": d}
            log(
                INFO,
                "Nesterov Momentum:"
                " l2_norm(pseudo_gradient)=%s,"
                " l2_norm(self.momentum_vector)=%s,"
                " l2_norm(fedavg_result)=%s"
                " l2_norm(model)=%s,",
                metrics_aggregated["server/l2_norm_pseudo_gradient"],
                metrics_aggregated["server/l2_norm_momentum_vector"],
                metrics_aggregated["server/l2_norm_fedavg_result"],
                metrics_aggregated["server/l2_norm_model"],
            )

        if self.track_inplace_aggregation:
            normal_result = aggregate(
                [
                    (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                    for _, fit_res in results_cached
                ],
            )
            layer_by_layer_diff = 0.0
            for x, y in zip(normal_result, fedavg_result, strict=False):
                layer_by_layer_diff += sum_of_squares([x - y])
            metrics_aggregated |= {
                "server/l2_norm_fedavg_gap": float(np.sqrt(layer_by_layer_diff)),
            }
            log(
                INFO,
                "Inplace aggregation gap: l2_norm(normal_result -"
                " fedavg_result)=%s, len_results: %s",
                layer_by_layer_diff,
                len(results_cached),
            )
        if metrics_callback is not None:
            metrics_callback.round_end(
                fedavg_result,
            )

        return self.parameters, metrics_aggregated
