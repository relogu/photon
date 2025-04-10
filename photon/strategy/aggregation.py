"""Handle aggregation in-place and potentially async."""

import ast
import time
from collections import defaultdict
from collections.abc import Generator, Iterable, Iterator
from functools import partial, reduce
from logging import DEBUG
from typing import Any

import numpy as np
from flwr.common import FitRes, NDArrays, Parameters, bytes_to_ndarray
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy

from photon.strategy.metrics import ServerMetricCallback


def aggregate_parameters(
    metrics_callback: ServerMetricCallback | None,
    accumulator: tuple[NDArrays | None, int],
    current: tuple[Iterator[np.ndarray] | Iterable[np.ndarray], int],
) -> tuple[NDArrays | None, int]:
    """Aggregate parameters in-place.

    Having this as a function avoids leaking variables
    Since python for-loops are unscoped.
    The function will use the memory of the the passed `current` parameter.

    Parameters
    ----------
    metrics_callback: ServerMetricCallback | None
        Optional callback for adding per-client metrics.
    accumulator : Tuple[NDArrays | None, int]
        A tuple containing the accumulated parameters and the number of examples seen.
    current : Tuple[NDArrays, int]
        A tuple containing the current parameters and the number of examples.

    Returns
    -------
        Tuple containing the updated parameters and the new total number of samples

    """
    current_params, num_examples = current
    start_time = time.time()
    log(DEBUG, f"Started aggregating client parameters with samples: {num_examples}")

    params, prev_total_examples = accumulator

    # NOTE: to keep this efficient the function returns a generator
    # the implicit contract is that it will not alter the state of the memory
    # passed to it
    if metrics_callback is not None:
        metrics_callback.process_per_client_results(current_params, num_examples)

    # Compute the new total number of samples
    new_total_samples = prev_total_examples + num_examples

    # Compute scaling factor for the accumulator
    acc_scaling_factor = float(prev_total_examples) / new_total_samples

    # Compute scaling factor for the update
    scaling_factor = float(num_examples) / new_total_samples

    if params is None:
        params = list(current_params)
    else:
        # Avoid allocating any temporaries
        # NOTE: We want the for-loop variable to be overwritten
        for x, y in zip(params, current_params, strict=True):
            x *= acc_scaling_factor  # noqa: PLW2901
            y *= scaling_factor  # noqa: PLW2901
            x += y  # noqa: PLW2901
            # Lack of scoping requires this
            del y

    # NOTE: Maybe be useless but let's help the Python GC figure out what to do
    del current_params

    log(
        DEBUG,
        f"""Aggregated client with samples: {num_examples}
                total samples used: {new_total_samples}
                time: {time.time() - start_time} """,
    )

    return params, new_total_samples


def aggregate_inplace(
    results: Iterable[tuple[Iterator[np.ndarray] | Iterable[np.ndarray], int]],
    metrics_callback: ServerMetricCallback | None = None,
) -> NDArrays | None:
    """Compute in-place weighted average, lazily and async.

    Parameters
    ----------
    results : Iterable[Tuple[NDArrays, int]]
        The results to aggregate.
    metrics_callback : ServerMetricCallback | None
        Optional callback for adding per-client metrics.

    Returns
    -------
    NDArrays | None
        The aggregated parameters.

    """
    # Holds the parameters and the total number of samples
    accumulator: tuple[NDArrays | None, int] = (None, 0)

    # Choose the aggregation function
    aggregation_fn = partial(aggregate_parameters, metrics_callback)

    # Aggregate the parameters
    agg_results, _ = reduce(aggregation_fn, results, accumulator)  # type: ignore[call-overload]

    return agg_results


def aggregate_cumulative_average(
    results: Iterable[tuple[ClientProxy, FitRes]],
    metrics_callback: ServerMetricCallback | None,
) -> NDArrays | None:
    """Compute in-place weighted average, lazily and async.

    Parameters
    ----------
    results : Iterable[Tuple[ClientProxy, FitRes]]
        The results to aggregate.
    metrics_callback : ServerMetricCallback | None
        Optional callback for adding per-client metrics.

    Returns
    -------
    NDArrays | None
        The aggregated parameters.

    """
    # NOTE: Only one ndarray exists at a time
    return aggregate_inplace(
        results=(
            (
                parameters_to_ndarrays_gen(fit_res.parameters),
                fit_res.num_examples,
            )
            for _, fit_res in results
        ),
        metrics_callback=metrics_callback,
    )


def parameters_to_ndarrays_gen(
    parameters: Parameters,
) -> Generator[np.ndarray, None, None]:
    """Convert parameters object to NumPy ndarrays.

    Parameters
    ----------
    parameters : Parameters
        The parameters to convert.

    Returns
    -------
    Generator[np.ndarray, None, None]
        A generator yielding NumPy ndarrays

    """
    return (bytes_to_ndarray(tensor) for tensor in parameters.tensors)


def weighted_average(
    metrics: list[tuple[int, dict]],
) -> dict:
    """Compute a weighted average over pre-defined metrics.

    Parameters
    ----------
    metrics : List[Tuple[int, Dict]]
        The metrics to aggregate.

    Returns
    -------
    Dict
        The weighted average over pre-defined metrics.

    """
    client_state_accumulator: dict[int | str, dict[str, Any]] = {}
    total_num_examples = sum(num_examples for num_examples, _ in metrics)
    weighted_metrics: dict = defaultdict(float)

    for num_examples, metric in metrics:
        if metric != {}:
            cid = metric.pop("cid", None)
            client_state = metric.pop("client_state", None)
            client_state_acc = metric.pop("client_state_acc", None)
            for key, value in metric.items():
                if not isinstance(value, str):
                    weighted_metrics[key] += num_examples * value
            if cid is not None and client_state is not None:
                client_state_accumulator[cid] = ast.literal_eval(client_state)
            if client_state_acc is not None:
                client_state_accumulator |= ast.literal_eval(client_state_acc)

    ret_dict = {
        key: value / total_num_examples for key, value in weighted_metrics.items()
    }
    if client_state_accumulator:
        ret_dict |= {"client_state_acc": str(client_state_accumulator)}

    return ret_dict
