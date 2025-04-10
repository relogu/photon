"""Client Application for a Flower-based LLM training node.

This module defines a client-side application using the Flower framework. It handles
parameter broadcasting, training, evaluation, and communication with a node manager.
"""

import time
import warnings
from collections.abc import Iterator
from logging import DEBUG

from flwr.common import (
    Code,
    ConfigsRecord,
    Context,
    EvaluateRes,
    FitRes,
    Message,
    RecordSet,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import update_console_handler
from flwr.common.recordset_compat import (
    evaluateres_to_recordset,
    fitres_to_recordset,
    parametersrecord_to_parameters,
    recordset_to_evaluateins,
    recordset_to_fitins,
)

from photon.node_manager.node_manager_app import NodeManagerApp
from photon.server.broadcast_utils import BROADCAST_INS
from photon.server.s3_utils import (
    S3_COMM_CONFIG,
    replace_parameters_in_recordset_with_remote,
    replace_remote_with_parameters_in_recordset,
)
from photon.server.server_util import COMM_, COMM_STACK, PARAMETERS
from photon.shm.constants import NM_PARAMETERS_SHM
from photon.shm.utils import get_parameters_shm, is_shm_existing, set_parameters_shm
from photon.utils import custom_ray_garbage_collector

EXPECTED_LATENCY = 300  # 5 minutes of expected latency between supernode and superlink

FIT_RESULTS = "fitres"
EVALUATE_RESULTS = "evaluateres"

# Fix the logger
update_console_handler(level=DEBUG, colored=False, timestamps=True)

# Filter user warning from configuration of MPT
warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    message=("If not using a Prefix Language Model*"),
    append=True,
)
# Filter deprecation warning from pkg_resources
warnings.filterwarnings(
    action="ignore",
    category=DeprecationWarning,
    message=("Deprecated call to *"),
    append=True,
)
warnings.filterwarnings(
    action="ignore",
    category=DeprecationWarning,
    message=("pkg_resources is deprecated*"),
    append=True,
)

# Flower ClientApp
app = NodeManagerApp()


def set_parameters(msg: Message, ctx: Context) -> Message:  # noqa: ARG001
    """Set parameters for this client node.

    This function downloads parameters from S3 if necessary, places them into shared
    memory, and returns a confirmation message.

    Parameters
    ----------
    msg : Message
        Incoming message containing parameters.
    ctx : Context
        Execution context passed by Flower.

    Returns
    -------
    Message
        Reply message confirming parameters were set.

    """
    msg = replace_parameters_in_recordset_with_remote(
        remote_uploader_downloader=app.remote_up_down,
        incoming_message=msg,
        comm_stack=app.cfg.photon.comm_stack,
        msg_str=f"{BROADCAST_INS}",
    )
    # Create new shared memory
    app.round_parameters, app.round_parameters_sh = get_parameters_shm(
        parameters_metadata=app.parameters_metadata,
        create=not is_shm_existing(app.node_manager_uuid + NM_PARAMETERS_SHM),
        name=app.node_manager_uuid + NM_PARAMETERS_SHM,
    )
    parameters = parameters_to_ndarrays(
        parametersrecord_to_parameters(
            msg.content.parameters_records[f"{BROADCAST_INS}.{PARAMETERS}"],
            keep_input=False,
        ),
    )
    set_parameters_shm(app.round_parameters, parameters)
    recordset = RecordSet()
    recordset.configs_records["broadcast"] = ConfigsRecord({"status": "OK"})
    return msg.create_reply(content=recordset)


def free_resources(msg: Message, ctx: Context) -> Message:  # noqa: ARG001
    """Free resources for this client node.

    Parameters
    ----------
    msg : Message
        Incoming message requesting to free resources.
    ctx : Context
        Execution context passed by Flower.

    Returns
    -------
    Message
        Reply message confirming resource deallocation.

    """
    return msg.create_reply(content=msg.content)


@app.lifespan()
def lifespan(_: Context) -> Iterator[None]:
    """Define the lifespan of the client application."""
    with custom_ray_garbage_collector(
        garbage_queue=app.ray_garbage_queue,
        list_of_threads=app.list_of_threads,
        join_at_the_end=False,
    ):
        yield


@app.train()
def train(msg: Message, ctx: Context) -> Message:  # noqa: ARG001
    """Perform training on this client node.

    This function handles potential worker restarts based on the round, executes the
    training process, compiles the results, and returns them in a reply message.

    Parameters
    ----------
    msg : Message
        Incoming message containing FitIns and configuration.
    ctx : Context
        Execution context passed by Flower.

    Returns
    -------
    Message
        Reply message with FitRes containing the trained parameters and metrics.

    """
    start_time = time.time()
    fitins = recordset_to_fitins(msg.content, keep_input=False)
    config = fitins.config
    assert "server_round" in config, "Server round must be in the config"
    if (int(config["server_round"]) + 1) % app.refresh_period == 0:
        app.close_workers()
        app.create_and_start_workers()

    trained_parameters, num_examples, metrics = app.fit(
        configs=msg.content.configs_records,
    )
    status = Status(code=Code.OK, message="chiappe sode")
    parameters_to_return = ndarrays_to_parameters(trained_parameters)
    fitres = FitRes(
        status=status,
        parameters=parameters_to_return,
        metrics=metrics,
        num_examples=num_examples,
    )
    recordset = fitres_to_recordset(fitres, keep_input=False)
    recordset.configs_records[f"{FIT_RESULTS}.{S3_COMM_CONFIG}"] = ConfigsRecord(
        {
            "endpoint_id": COMM_ + app.node_manager_uuid + NM_PARAMETERS_SHM,
            "folder_name": COMM_STACK,
            "file_name": PARAMETERS,
        },
    )
    ttl = EXPECTED_LATENCY + int(time.time() - start_time)
    if app.list_of_ray_object_refs:
        for ray_object_to_collect in app.list_of_ray_object_refs:
            app.ray_garbage_queue.put(ray_object_to_collect)
    app.list_of_ray_object_refs = None
    new_msg, app.list_of_ray_object_refs = replace_remote_with_parameters_in_recordset(
        remote_uploader_downloader=app.remote_up_down,
        outgoing_message=msg.create_reply(
            content=recordset,
            ttl=ttl,
        ),
        comm_stack=app.cfg.photon.comm_stack,
        msg_str=FIT_RESULTS,
    )
    return new_msg


@app.evaluate()
def evaluate(msg: Message, ctx: Context) -> Message:  # noqa: ARG001
    """Perform evaluation on this client node.

    This function receives evaluation instructions, executes the evaluation process,
    and returns the results in a reply message.

    Parameters
    ----------
    msg : Message
        Incoming message containing EvaluateIns and configuration.
    ctx : Context
        Execution context passed by Flower.

    Returns
    -------
    Message
        Reply message with EvaluateRes containing the evaluation results.

    """
    evaluateins = recordset_to_evaluateins(msg.content, keep_input=False)
    config = evaluateins.config
    assert "server_round" in config, "Server round must be in the config"
    loss, num_examples, metrics = app.eval(configs=msg.content.configs_records)
    status = Status(code=Code.OK, message="chiappe sode")
    evaluateres = EvaluateRes(
        status=status,
        loss=loss,
        metrics=metrics,
        num_examples=num_examples,
    )
    recordset = evaluateres_to_recordset(evaluateres)
    recordset.configs_records[f"{EVALUATE_RESULTS}.{S3_COMM_CONFIG}"] = ConfigsRecord(
        {
            "endpoint_id": app.node_manager_uuid + NM_PARAMETERS_SHM,
            "folder_name": COMM_STACK,
            "file_name": PARAMETERS,
        },
    )
    return msg.create_reply(recordset)


@app.query()
def query(msg: Message, ctx: Context) -> Message:
    """Dispatch incoming queries to the appropriate handler.

    This function routes incoming query messages to subroutines for
    parameter broadcasting, property retrieval, or resource cleanup.

    Parameters
    ----------
    msg : Message
        Incoming message with query instructions.
    ctx : Context
        Execution context passed by Flower.

    Returns
    -------
    Message
        Reply message with the result of the query.

    Raises
    ------
    ValueError
        If the query_type is unknown.

    """
    content = msg.content
    assert "query" in content.configs_records, "Query message must contain 'query' key"
    query_type = content.configs_records["query"]["type"]
    match query_type:
        case "broadcast_parameters":
            return set_parameters(msg=msg, ctx=ctx)
        case "free_resources":
            return free_resources(msg=msg, ctx=ctx)
    msg = f"Unknown query_type: {query_type!s}."  # type: ignore[reportAssignmentType, assignment]
    raise ValueError(msg)
