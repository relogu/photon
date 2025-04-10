"""Dispatch strategy based on configuration."""

import copy
import sys
from typing import TypeAlias

from flwr.common import Parameters

from photon.conf.base_schema import BaseConfig, StrategyName
from photon.strategy.aggregation import weighted_average
from photon.strategy.fedadam import FedAdam
from photon.strategy.fedavg_eff import FedAvgEfficient as FedAvgEff
from photon.strategy.fedmom import FedMom
from photon.strategy.fednestorov import FedNesterov
from photon.strategy.fedyogi import FedYogi
from photon.strategy.metrics import FedSimpleNoiseScale

SupportedStrategies: TypeAlias = FedNesterov | FedMom | FedYogi | FedAdam | FedAvgEff


def dispatch_strategy(
    cfg: BaseConfig,
) -> SupportedStrategies:
    """Dispatch the appropriate federated learning strategy based on the configuration.

    Parameters
    ----------
    cfg : BaseConfig
        The configuration object containing the strategy name and its parameters.

    Returns
    -------
    SupportedStrategies
        The initialized federated learning strategy.

    Raises
    ------
    ValueError
        If an unknown strategy name is provided.

    """
    # NOTE: We need to lowercase the match as the Enum instantiate with auto() in Python
    # lowercases by default
    match cfg.fl.strategy_name.lower():
        case StrategyName.NESTOROV:
            assert cfg.fl.strategy_kwargs.server_learning_rate is not None, (
                "Server learning rate is required for Nestorov strategy."
            )
            assert cfg.fl.strategy_kwargs.server_momentum is not None, (
                "Server momentum is required for Nestorov strategy."
            )
            return FedNesterov(
                # NOTE: We put a fake array as it will be touched on again later
                initial_parameters=Parameters(tensors=[], tensor_type="empty"),
                fit_metrics_aggregation_fn=weighted_average,
                evaluate_metrics_aggregation_fn=weighted_average,
                server_learning_rate=cfg.fl.strategy_kwargs.server_learning_rate,
                server_momentum=cfg.fl.strategy_kwargs.server_momentum,
                # These are not really important anymore with this new server
                fraction_fit=sys.float_info.min,
                fraction_evaluate=sys.float_info.min,
                min_fit_clients=cfg.fl.n_clients_per_round,
                min_evaluate_clients=1,
                min_available_clients=cfg.fl.n_clients_per_round,
                evaluate_fn=None,
                on_fit_config_fn=None,
                on_evaluate_config_fn=None,
                accept_failures=False,
                obtain_server_metrics_callback=(
                    FedSimpleNoiseScale if cfg.fl.use_noise_scale_metric else None
                ),
                cfg=copy.deepcopy(cfg),
            )
        case StrategyName.FEDMOM:
            assert cfg.fl.strategy_kwargs.server_learning_rate is not None, (
                "Server learning rate is required for Nestorov strategy."
            )
            assert cfg.fl.strategy_kwargs.server_momentum is not None, (
                "Server momentum is required for Nestorov strategy."
            )
            return FedMom(
                # NOTE: We put a fake array as it will be touched on again later
                initial_parameters=Parameters(tensors=[], tensor_type="empty"),
                fit_metrics_aggregation_fn=weighted_average,
                evaluate_metrics_aggregation_fn=weighted_average,
                server_learning_rate=cfg.fl.strategy_kwargs.server_learning_rate,
                server_momentum=cfg.fl.strategy_kwargs.server_momentum,
                # These are not really important anymore with this new server
                fraction_fit=sys.float_info.min,
                fraction_evaluate=sys.float_info.min,
                min_fit_clients=cfg.fl.n_clients_per_round,
                min_evaluate_clients=1,
                min_available_clients=cfg.fl.n_clients_per_round,
                evaluate_fn=None,
                on_fit_config_fn=None,
                on_evaluate_config_fn=None,
                accept_failures=False,
            )
        case StrategyName.FEDAVG:
            return FedAvgEff(
                # NOTE: We put a fake array as it will be touched on again later
                initial_parameters=Parameters(tensors=[], tensor_type="empty"),
                fit_metrics_aggregation_fn=weighted_average,
                evaluate_metrics_aggregation_fn=weighted_average,
                server_learning_rate=1.0,
                # These are not really important anymore with this new server
                fraction_fit=sys.float_info.min,
                fraction_evaluate=sys.float_info.min,
                min_fit_clients=cfg.fl.n_clients_per_round,
                min_evaluate_clients=1,
                min_available_clients=cfg.fl.n_clients_per_round,
                evaluate_fn=None,
                on_fit_config_fn=None,
                on_evaluate_config_fn=None,
                accept_failures=False,
                obtain_server_metrics_callback=(
                    FedSimpleNoiseScale if cfg.fl.use_noise_scale_metric else None
                ),
                cfg=copy.deepcopy(cfg),
            )
        case StrategyName.FEDYOGI:
            return FedYogi(
                # NOTE: We put a fake array as it will be touched on again later
                initial_parameters=Parameters(tensors=[], tensor_type="empty"),
                fit_metrics_aggregation_fn=weighted_average,
                evaluate_metrics_aggregation_fn=weighted_average,
                # These are not really important anymore with this new server
                fraction_fit=sys.float_info.min,
                fraction_evaluate=sys.float_info.min,
                min_fit_clients=cfg.fl.n_clients_per_round,
                min_evaluate_clients=1,
                min_available_clients=cfg.fl.n_clients_per_round,
                evaluate_fn=None,
                on_fit_config_fn=None,
                on_evaluate_config_fn=None,
                accept_failures=False,
                eta=cfg.fl.strategy_kwargs.eta,
                beta_1=cfg.fl.strategy_kwargs.beta_1,
                beta_2=cfg.fl.strategy_kwargs.beta_2,
                tau=cfg.fl.strategy_kwargs.tau,
            )
        case StrategyName.FEDADAM:
            return FedAdam(
                # NOTE: We put a fake array as it will be touched on again later
                initial_parameters=Parameters(tensors=[], tensor_type="empty"),
                fit_metrics_aggregation_fn=weighted_average,
                evaluate_metrics_aggregation_fn=weighted_average,
                # These are not really important anymore with this new server
                fraction_fit=sys.float_info.min,
                fraction_evaluate=sys.float_info.min,
                min_fit_clients=cfg.fl.n_clients_per_round,
                min_evaluate_clients=1,
                min_available_clients=cfg.fl.n_clients_per_round,
                evaluate_fn=None,
                on_fit_config_fn=None,
                on_evaluate_config_fn=None,
                accept_failures=False,
                eta=cfg.fl.strategy_kwargs.eta,
                beta_1=cfg.fl.strategy_kwargs.beta_1,
                beta_2=cfg.fl.strategy_kwargs.beta_2,
                tau=cfg.fl.strategy_kwargs.tau,
            )
        case _:
            msg = "Unknown strategy"
            raise ValueError(msg)
