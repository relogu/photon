"""Utility functions for fit tasks on main server loop in flwr next."""

import ast
import time
from collections.abc import Callable, Generator
from logging import ERROR, WARNING
from typing import Any, cast

from composer.loggers import RemoteUploaderDownloader
from flwr.common import (
    Code,
    FitRes,
    Message,
    MessageType,
    Parameters,
    Scalar,
    Status,
    log,
)
from flwr.common.recordset_compat import (
    recordset_to_fitres,
)
from flwr.common.typing import ConfigsRecordValues
from flwr.server import Driver
from flwr.server.strategy import FedAvg

from photon.conf.base_schema import BaseConfig
from photon.server.s3_utils import (
    replace_parameters_in_recordset_with_remote,
)
from photon.server.server_util import (
    TooManyFailuresError,
    message_collaborative,
)
from photon.utils import (
    ClientState,
)
from photon.wandb_history import WandbHistory


def handle_fit_replies(  # noqa: PLR0914
    cfg: BaseConfig,
    replies: Generator[Message, None, None],
    server_state: tuple[FedAvg, int, int],
    remote_up_down: RemoteUploaderDownloader | None,
    client_state: dict[str | int, ClientState],
) -> (
    tuple[
        Parameters | None,
        dict[str, Scalar],
        tuple[list[tuple[dict[str, Scalar], Status, int]], list[FitRes | None]],
        dict[str | int, ClientState],
        int,
    ]
    | None
):
    """Handle fit replies from clients.

    Parameters
    ----------
    cfg : BaseConfig
        The configuration object.
    replies : Generator[Message, None, None]
        The generator of messages from the clients.
    server_state : Tuple[FedAvg, int, int]
        The server state containing the strategy, the current round,
        and the server steps cumulative.
    remote_up_down : RemoteUploaderDownloader | None
        The object to handle S3 communication.
    client_state : dict[str | int, ClientState]
        The dictionary of client states.


    Returns
    -------
    None | Tuple[
        Parameters | None,
        Dict[str, Scalar],
        Tuple[List[Tuple[Dict[str, Scalar], Status, int]], List[FitRes | None]],
        dict[str | int, ClientState],
        int,
    ]
        The aggregated parameters, the aggregated metrics, the metrics and failures, the
        update client states and the updated server steps cumulative.

    Raises
    ------
    TooManyFailuresError
        If the number of failures exceeds the maximum allowed.

    """
    strat, current_round, server_steps_cumulative = server_state
    # Translate message with fake parameters with parameters downloaded from the S3
    processed_msgs = (
        (
            replace_parameters_in_recordset_with_remote(
                remote_uploader_downloader=remote_up_down,
                incoming_message=msg,
                comm_stack=cfg.photon.comm_stack,
                msg_str="fitres",
            )
            if msg.has_content()
            else msg
        )
        for msg in replies
    )

    # Translate Messages to FitRes
    status = Status(code=Code.FIT_NOT_IMPLEMENTED, message="Unexpected empty content")
    error_fitres = FitRes(
        status=status,
        parameters=Parameters(tensors=[], tensor_type="empty"),
        metrics={},
        num_examples=1,
    )
    all_fitres = (
        (
            recordset_to_fitres(msg.content, keep_input=True)
            if msg.has_content()
            else error_fitres
        )
        for msg in processed_msgs
    )

    results_and_failures = (
        ((fit_res.status.code == Code.OK, fit_res)) for fit_res in all_fitres
    )

    # Using a generator limits us in failure/metrics accumulation
    # The output params are not used in the aggregation
    # They are merely populated by the processing of the generator
    failures: list[FitRes | None] = []

    metrics_accumulator: list[tuple[dict[str, Scalar], Status, int]] = []

    handle_success_and_failure = get_handle_success_and_failure_fit(
        metrics_accumulator,
        failures,
        accept_failures_cnt=cfg.fl.accept_failures_cnt,
    )
    handled_results_and_failures = (
        handle_success_and_failure(result)  # type: ignore[arg-type]
        for result in results_and_failures
    )

    results = (result for success, result in handled_results_and_failures if success)

    parameters_aggregated = None
    metrics_aggregated: dict = {}

    try:
        # Aggregate training results
        parameters_aggregated, metrics_aggregated = strat.aggregate_fit(
            current_round,
            ((None, fit_res) for fit_res in results),  # type: ignore[reportArgumentType, arg-type]
            failures,  # type: ignore[reportArgumentType, arg-type]
        )

        # Collect statistics that Photon uses from the FitRes of the NodeManagers

        fit_metrics = [
            (num_examples, metrics) for metrics, _, num_examples in metrics_accumulator
        ]
        client_state_accumulator: dict[str | int, dict[str, Any]] = {}
        for _, inner_metrics in fit_metrics:
            acc: dict[str | int, dict[str, Any]] = ast.literal_eval(
                cast("str", inner_metrics["client_state_acc"]),
            )
            client_state_accumulator |= acc
        # NOTE: When using partial participation
        # We need to accumulate the keys of the old state
        # and the new state
        client_state |= {
            k: ClientState(**v) for k, v in client_state_accumulator.items()
        }
        # NOTE: Update the server steps cumulative by adding to the previous
        # value the maximum number of local steps done across the clients sample
        # in this round
        max_steps = max(
            *[_client_state.steps_done for _client_state in client_state.values()],
            0,
        )
        server_steps_cumulative += max_steps
        # NOTE: Reset the steps_done for all clients to zero as it is just meant
        # to be an ephemeral record of the local steps done during one federated
        # round
        for c_state in client_state.values():
            c_state.steps_done = 0

        # Aggregate the metrics
        # NOTE: This bypasses any metrics aggregation in the aggregate_fit of the
        # strategy because the metrics are empty there
        if strat.fit_metrics_aggregation_fn:
            metrics_aggregated |= strat.fit_metrics_aggregation_fn(fit_metrics)

        elif current_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")
    except TooManyFailuresError as e:
        if cfg.fl.ignore_failed_rounds:
            log(
                ERROR,
                """Ignoring failed round %s: %s,
                there are %s failures: %s""",
                current_round,
                e,
                len(failures),
                failures,
            )
        else:
            raise
    return (
        parameters_aggregated,
        metrics_aggregated,
        (metrics_accumulator, failures),
        client_state,
        server_steps_cumulative,
    )


def get_handle_success_and_failure_fit(
    metrics_accumulator: list[tuple[dict[str, Scalar], Status, int]],
    fit_failures: list[FitRes | None],
    accept_failures_cnt: int | None,
) -> Callable[
    [tuple[bool, FitRes] | tuple[bool, None]],
    tuple[bool, FitRes] | tuple[bool, None],
]:
    """Closure to generate a function which handles client success and failure.

    The function distinguishes between intentional and unintentional failures.
    It enforces constraints on the number of unintentional failures.
    It stores results and failures in the respective lists.

    Parameters
    ----------
    metrics_accumulator : List[Tuple[ClientProxy, Dict[str, Scalar], Status, int]]
        The list where the metrics are accumulated.
    fit_failures : List[FitRes | None]
        The list where the failures are accumulated.
    accept_failures_cnt : int | None
        The maximum number of failures to accept.

    Returns
    -------
    handle_success_and_failure : Callable[
        [
            Tuple[bool, FitRes]
            | Tuple[bool, FitRes | BaseException]
        ],
        Tuple[bool, FitRes]
        | Tuple[bool, FitRes | BaseException],
    ]
        The function which handles client success and failure while saving the outputs.

    """

    def handle_success_and_failure_fit(
        result: tuple[bool, FitRes] | tuple[bool, None],
    ) -> tuple[bool, FitRes] | tuple[bool, None]:
        cnt_failures = 0

        match result:
            case (True, res):
                fit_res = cast("FitRes", res)
                metrics_accumulator.append(
                    (
                        fit_res.metrics,
                        fit_res.status,
                        fit_res.num_examples,
                    ),
                )
                return (True, fit_res)
            case (False, res):
                cnt_failures += 1
                if (
                    accept_failures_cnt is not None
                    and cnt_failures > accept_failures_cnt
                ):
                    msg = f"""Unintentional failures passed
                        the maximum: {accept_failures_cnt}"""
                    raise TooManyFailuresError(
                        msg,
                    )
                fit_failures.append(res)  # type: ignore[arg-type]
                return (False, res) if res is not None else (False, None)  # type: ignore[return-value]
        return result

    return handle_success_and_failure_fit


def fit_round(  # noqa: PLR0913, PLR0917
    sampled_clients: list[str] | list[int],
    all_node_ids: list[int],
    driver: Driver,
    fit_config_fn: Callable[[int, int | str], dict[str, ConfigsRecordValues]],
    current_round: int,
    client_state: dict[str | int, ClientState],
    server_steps_cumulative: int,
    cfg: BaseConfig,
    strategy: FedAvg,
    remote_up_down: RemoteUploaderDownloader | None,
    history: WandbHistory,
    parameters: Parameters,
) -> tuple[
    Parameters,
    dict[str | int, ClientState],
    int,
    WandbHistory,
]:
    """Execute a round of federated training.

    Parameters
    ----------
    sampled_clients : list[str] | list[int]
        The list of clients to sample.
    all_node_ids : list[int]
        The list of all node ids.
    driver : Driver
        The driver object.
    fit_config_fn : Callable[[int, int | str], dict[str, ConfigsRecordValues]]
        The function to generate the fit configurations.
    current_round : int
        The current round number.
    client_state : dict[str | int, ClientState]
        The dictionary of client states.
    server_steps_cumulative : int
        The cumulative number of server steps.
    cfg : BaseConfig
        The configuration object.
    strategy : FedAvg
        The strategy object.
    remote_up_down : RemoteUploaderDownloader | None
        The object to handle S3 communication.
    history : WandbHistory
        The history object.
    parameters : Parameters
        The model parameters.



    Returns
    -------
        tuple: A tuple containing updated parameters, client state, server steps
            cumulative, and history.

    """
    fit_round_time = time.time_ns()
    fit_replies = (
        # Send a single client-message to every NodeManager at a time
        message_collaborative(
            driver=driver,
            message_type=MessageType.TRAIN,
            sampled_clients=sampled_clients,
            gen_ins_function=fit_config_fn,
            all_node_ids=all_node_ids,
            current_round=current_round,
            msg_str="fitins",
            client_state=client_state,
            server_steps_cumulative=server_steps_cumulative,
        )
    )
    res_fit = handle_fit_replies(
        cfg,
        fit_replies,
        (strategy, current_round, server_steps_cumulative),
        remote_up_down,
        client_state,
    )
    if res_fit is not None:
        (
            parameters_aggregated,
            fit_metrics,
            (_raw_metrics, _failures),
            client_state,
            server_steps_cumulative,
        ) = res_fit
        parameters = (
            parameters_aggregated if parameters_aggregated is not None else parameters
        )
        history.add_metrics_distributed_fit(
            server_round=current_round,
            metrics=fit_metrics,
        )
    history.add_metrics_centralized(
        server_round=current_round,
        metrics={"server/fit_round_time": (time.time_ns() - fit_round_time) * 1e-9},
    )

    return parameters, client_state, server_steps_cumulative, history
