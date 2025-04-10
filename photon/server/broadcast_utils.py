"""Utility functions for broadcasting to clients on main server loop in flwr next."""

from logging import DEBUG

from composer.loggers import RemoteUploaderDownloader
from flwr.common import (
    DEFAULT_TTL,
    Code,
    ConfigsRecord,
    Message,
    MessageType,
    Parameters,
    RecordSet,
    log,
)
from flwr.common.recordset_compat import parameters_to_parametersrecord
from flwr.server import Driver
from ray import ObjectRef

from photon.conf.base_schema import CommStack
from photon.server.s3_utils import replace_remote_with_parameters_in_recordset
from photon.server.server_util import COMM_STACK, PARAMETERS

BROADCAST_INS = "broadcast_ins"
BROADCAST_P = "broadcast_parameters"


def parameters_to_broadcast_recordset(
    parameters: Parameters,
    *,
    keep_input: bool,
) -> RecordSet:
    """Convert Parameters into RecordSet for broadcasting, optionally keeping the input.

    This function takes a set of parameters and converts them into a format suitable for
    broadcasting to a record set. It allows for keeping the original input parameters
    as part of the conversion process. The converted parameters and a configuration
    indicating the action type ("set_parameters") are added to the record set.

    Parameters
    ----------
    parameters : Parameters
        The parameters to be converted and broadcasted.
    keep_input : bool
        A flag indicating whether to keep the original input parameters.

    Returns
    -------
    RecordSet
        The record set containing the converted parameters and configuration record.

    """
    recordset = RecordSet()
    parametersrecord = parameters_to_parametersrecord(parameters, keep_input)
    recordset.parameters_records[f"{BROADCAST_INS}.{PARAMETERS}"] = parametersrecord
    recordset.configs_records["query"] = ConfigsRecord({"type": BROADCAST_P})
    return recordset


def broadcast_parameters_to_nodes(  # noqa: PLR0913, PLR0917
    driver: Driver,
    parameters: Parameters,
    node_ids: list[int],
    current_round: int,
    remote_uploader_downloader: RemoteUploaderDownloader | None,
    server_uuid: str,
    comm_stack: CommStack,
) -> list[ObjectRef] | None:
    """Broadcast parameters to specified nodes using either direct messaging or S3.

    This function takes a set of parameters and broadcasts them to a list of node IDs.
    It supports two modes of communication: direct messaging through the `driver` and
    indirect messaging via S3 when `comm_stack.s3` is True. The function first creates a
    recordset from the parameters, adds status and S3 configuration information to the
    recordset, and then either uploads the parameters to S3 (if `comm_stack.s3` is True)
    or prepares them for direct messaging. It then sends the messages to all specified
    nodes and waits for their acknowledgments, ensuring all nodes have successfully
    received the parameters.

    Parameters
    ----------
    driver : Driver
        The communication driver responsible for sending and receiving messages.
    parameters : Parameters
        The parameters to be broadcasted to the nodes.
    node_ids : list[int]
        A list of node IDs to which the parameters will be broadcasted.
    current_round : int
        The current round of the operation, used for tracking and logging.
    remote_uploader_downloader : RemoteUploaderDownloader | None
        The remote uploader/downloader instance for S3 communication. Required if
        `comm_stack.s3` is True.
    server_uuid : str
        The unique identifier of the server, used for S3 communication.
    comm_stack : CommStack
        The communication stack configuration indicating which communication methods
        are enabled (S3, shared memory, Ray).

    Returns
    -------
    list[ObjectRef] | None
        A list of Ray object references to the broadcasted parameters, or None if the
        broadcast failed.

    Raises
    ------
    ValueError
        If any node reports a failure in receiving or processing the broadcasted
        parameters.

    Notes
    -----
    The function assumes the existence of `parameters_to_broadcast_recordset`,
    `replace_remote_with_parameters_in_recordset`, `log`, and `time.sleep`
    functions/utilities, as well as `MessageType`, `ConfigsRecord`, `Code`, and `DEBUG`
    constants. It also relies on the `Driver` interface for message handling.

    Steps
    -----
    1. Create a recordset from the parameters.
    2. Add status and S3 configuration information to the recordset.
    3. If `comm_stack.s3` is True, upload the parameters to S3.
    4. Prepare messages for direct messaging or S3 communication.
    5. Send the messages to all specified nodes.
    6. Wait for acknowledgments from all nodes.
    7. Verify that all nodes have successfully received the parameters.

    """
    # Message name
    msg_str = f"{BROADCAST_INS}"
    # Create one message per node from one single recordset
    messages = []
    recordset = parameters_to_broadcast_recordset(
        parameters=parameters,
        keep_input=True,
    )
    # Add Status to the recordset
    recordset.configs_records[f"{msg_str}.status"] = ConfigsRecord(
        {
            "code": int(Code.OK.value),
            "message": "Broadcasting parameters",
        },
    )
    # Add S3 configuration to the recordset
    recordset.configs_records[f"{msg_str}.s3_comm_config"] = ConfigsRecord(
        {
            "endpoint_id": server_uuid,
            "folder_name": COMM_STACK,
            "file_name": PARAMETERS,
        },
    )
    # Translating the message and uploading the parameters to S3 if asked to
    fake_message, list_of_ray_object_refs = replace_remote_with_parameters_in_recordset(
        remote_uploader_downloader=remote_uploader_downloader,
        # Create a fake message to upload to S3
        outgoing_message=driver.create_message(
            content=recordset,
            message_type=MessageType.QUERY,
            dst_node_id=node_ids[0],
            group_id=str(current_round),
            ttl=DEFAULT_TTL,
        ),
        comm_stack=comm_stack,
        msg_str=msg_str,
    )
    # Replacing recordset with the empty one
    recordset = fake_message.content
    # Send the message to all nodes
    for node_id in node_ids:
        message = driver.create_message(
            content=recordset,
            message_type=MessageType.QUERY,
            dst_node_id=node_id,
            group_id=str(current_round),
            ttl=DEFAULT_TTL,
        )
        messages.append(message)
    # Push all messages to all nodes
    message_ids = driver.push_messages(messages)
    log(DEBUG, f"Pushed {len(messages)} broadcast messages to {len(node_ids)} nodes")
    # Wait for results, ignore empty message_ids
    message_ids = [message_id for message_id in message_ids if message_id]
    all_replies: list[Message] = []
    while True:
        replies = driver.pull_messages(message_ids=message_ids)
        all_replies += replies
        if len(all_replies) == len(message_ids):
            break
    # Filter correct results
    all_broadcast_res = [msg.content for msg in all_replies if msg.has_content()]
    # Elaborate results
    for message_res in all_broadcast_res:
        assert "broadcast" in message_res.configs_records, "Broadcast key not found"
        assert "status" in message_res.configs_records["broadcast"], (
            "Status key not found"
        )
        if message_res.configs_records["broadcast"]["status"] != "OK":
            msg = "Broadcast failed"
            raise ValueError(msg)
    log(DEBUG, f"Received {len(all_broadcast_res)} results")
    return list_of_ray_object_refs
