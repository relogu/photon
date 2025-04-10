"""Strategy holding a BaseConfig."""

from collections.abc import Callable

from flwr.common import (
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
)
from flwr.server.strategy import FedAvg

from photon.conf.base_schema import BaseConfig


class FedAvgWithConfig(FedAvg):
    """A strategy guaranteed to hold a BaseConfig."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes,line-too-long
    def __init__(  # noqa: PLR0913
        self,
        *,
        initial_parameters: Parameters,
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
        cfg: BaseConfig | None = None,
    ) -> None:
        """Initialize the FedNesterov strategy.

        Parameters
        ----------
        initial_parameters : Parameters
            Initial global model parameters.
        fraction_fit : float, optional
            Fraction of clients used during training. Defaults to 1.0.
        fraction_evaluate : float, optional
            Fraction of clients used during validation. Defaults to 1.0.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_evaluate_clients : int, optional
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        evaluate_fn : Callable, optional
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable, optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable, optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not to accept rounds containing failures. Defaults to True.
        fit_metrics_aggregation_fn : MetricsAggregationFn, optional
            Metrics aggregation function for training. Defaults to None.
        evaluate_metrics_aggregation_fn : MetricsAggregationFn, optional
            Metrics aggregation function for evaluation. Defaults to None.
        cfg : BaseConfig, optional
            Configuration object. Defaults to None.

        """
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
        self.cfg = cfg
        self.parameters = initial_parameters
