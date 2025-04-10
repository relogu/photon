"""Utility functions for running S3-related tasks on main server loop in flwr next."""

import ast
import inspect
import operator
import pickle  # noqa: S403
import re
import time
from dataclasses import asdict
from itertools import groupby
from logging import DEBUG, INFO, WARNING
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import ray
from composer import Trainer
from composer.loggers import RemoteUploaderDownloader
from composer.utils.file_helpers import (
    list_remote_objects,
    maybe_create_object_store_from_uri,
    parse_uri,
    validate_given_remote_path,
)
from composer.utils.object_store import S3ObjectStore
from flwr.common import (
    Code,
    ConfigsRecord,
    Message,
    NDArrays,
    Parameters,
    log,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.recordset_compat import (
    _extract_status_from_recordset,  # noqa: PLC2701
    parameters_to_parametersrecord,
    parametersrecord_to_parameters,
)
from omegaconf import OmegaConf

from photon.conf.base_schema import BaseConfig, CommStack, S3CommConfig
from photon.shm.utils import (
    ModelParametersMetadata,
    get_parameters_shm,
    is_shm_existing,
    set_parameters_shm,
)
from photon.strategy.constants import (
    FIRST_MOMENTUM,
    MODEL_PARAMETERS,
    SECOND_MOMENTUM,
)
from photon.strategy.fedadam import FedAdam
from photon.strategy.fedavg_eff import FedAvgEfficient
from photon.strategy.fedmom import FedMom
from photon.strategy.fednestorov import FedNesterov
from photon.strategy.fedyogi import FedYogi
from photon.utils import (
    ClientState,
    create_remote_up_down,
    download_file_from_s3,
    dump_model_parameters_to_file,
    load_model_parameters_from_file,
    set_trainer_params_from_ndarrays,
    upload_file_to_s3,
)
from photon.wandb_history import WandbHistory

S3_COMM_CONFIG = "s3_comm_config"


class NoCheckpointsFoundError(Exception):
    """Exception raised when there are no checkpoints in the path looked up."""


def delete_object(
    object_path: str,
) -> None:
    """Delete a local/remote object from a specified path.

    This function attempts to delete the specified object from the provided path. If the
    path is a remote S3 path, it uses the `delete_remote_object` function to delete the
    object. If the path is a local path, it deletes the local file using the `unlink`
    method.

    Parameters
    ----------
    object_path : str
        The path to the object to be deleted. This can be either a local path or a
        remote S3 path.

    Example
    -------
    >>> delete_object("s3://mybucket/myfolder/myfile.txt")
    >>> delete_object("/local/path/to/myfile.txt")

    Notes
    -----
    This function uses the `delete_remote_object` function to delete objects from a
    remote S3 path. For local paths, it uses the `Path.unlink` method to delete local
    files.

    """
    try:
        delete_remote_object(object_path)
    except ValueError:
        # Local path
        Path(object_path).unlink()


def list_objects(
    run_uuid_path: str,
) -> tuple[bool, list[str]]:
    """List objects in a local/(remote s3) given path.

    This function attempts to list objects in the specified path. If the path is remote
    S3 path, it uses the `list_remote_objects` function to list the objects. If the path
    is a local path, it lists all files recursively within the directory.

    Parameters
    ----------
    run_uuid_path : str
        The path to list objects from. This can be either a local path or a remote S3
        path.

    Returns
    -------
    tuple[bool, list[str]]
        A tuple where the first element is a boolean indicating whether the path is
        remote (True for remote, False for local), and the second element is a list of
        object paths.

    Example
    -------
    >>> is_remote, objects = list_objects("s3://mybucket/myfolder")
    >>> print(is_remote)
    True
    >>> print(objects)
    ['s3://mybucket/myfolder/file1.txt', 's3://mybucket/myfolder/file2.txt']

    >>> is_remote, objects = list_objects("/local/path/to/folder")
    >>> print(is_remote)
    False
    >>> print(objects)
    ['/local/path/to/folder/file1.txt', '/local/path/to/folder/file2.txt']

    """
    try:
        return True, list_remote_objects(run_uuid_path)
    except ValueError:
        # Local path
        return False, [str(p) for p in Path(run_uuid_path).rglob("*") if p.is_file()]


def extract_s3_comm_config_from_configrecord(
    s3_comm_config: ConfigsRecord,
) -> tuple[str, str, str]:
    """Extract S3 communication configuration details from a ConfigsRecord object.

    This function parses a ConfigsRecord object containing S3 communication
    configuration and extracts essential information required for S3 operations.
    Specifically, it retrieves the `endpoint_id`, `file_name`, and `current_round`
    from the ConfigsRecord. These values are crucial for identifying the correct S3
    bucket and path, and for versioning or round-specific operations.

    Parameters
    ----------
    s3_comm_config : ConfigsRecord
        A ConfigsRecord object containing the S3 communication configuration. Expected
        to have keys for `endpoint_id`, `file_name`, and `current_round`.

    Returns
    -------
    tuple[str, str, str]
        A tuple containing `endpoint_id`, `file_name`, and `current_round` as strings.

    Raises
    ------
    ValueError
        If any of the required keys (`endpoint_id`, `file_name`, or `current_round`) are
        missing from the ConfigsRecord.

    Notes
    -----
    The function ensures that all returned values are strings, even if they are provided
    as different types in the ConfigsRecord. This standardization facilitates their use
    in S3 operations without further type checking or conversion.

    """
    # Extract endpoint id from the content of the message
    endpoint_id: Any
    if "endpoint_id" in s3_comm_config:
        endpoint_id = str(s3_comm_config["endpoint_id"])
    else:
        msg = "endpoint_id is not present in the message"
        raise ValueError(msg)
    file_name: Any
    if "file_name" in s3_comm_config:
        file_name = str(s3_comm_config["file_name"])
    else:
        msg = "file_name is not present in the message"
        raise ValueError(msg)
    folder_name: Any
    if "folder_name" in s3_comm_config:
        folder_name = str(s3_comm_config["folder_name"])
    else:
        msg = "folder_name is not present in the message"
        raise ValueError(msg)
    return endpoint_id, file_name, folder_name


def interpret_resume_round(
    resume_round: int | None,
    run_uuid_path: str,
    state_keys: tuple[str, ...],
    *,
    raise_error: bool = True,
) -> int | None:
    """Interpret the resume round parameter for server checkpoint resumption.

    This function interprets the `resume_round` parameter, which specifies the round
    to resume server operations from. If `resume_round` is negative, it is treated as
    an index into the list of sorted rounds obtained from the server's path, allowing
    for reverse indexing. If `resume_round` is None, the function returns None,
    indicating no specific round to resume from. An error is raised if no checkpoints
    are found when `raise_error` is True and `resume_round` is negative but no rounds
    are available.

    Parameters
    ----------
    resume_round : int | None
        The round number to resume from. If negative, treated as a reverse index. If
        None, indicates no resumption is required.
    run_uuid_path : str
        The path to the run uuid root.
    state_keys : tuple[str, ...]
        A tuple of state keys used to identify the federated rounds.
    raise_error : bool, optional
        Whether to raise an error if no checkpoints are found and `resume_round` is
        negative. Default is True.

    Returns
    -------
    int | None
        The interpreted round number to resume from, or None if no resumption.

    Raises
    ------
    NoCheckpointsFoundError
        If `raise_error` is True, no checkpoints are found, and `resume_round` < 0.

    """
    log(
        DEBUG,
        "The parameter `resume_round=%s` will be interpret as an index "
        "for the list of rounds for the run_uuid_path=%s",
        resume_round,
        run_uuid_path,
    )
    if resume_round is None:
        return None
    if resume_round < 0:
        server_round_indices = obtain_sorted_runs(run_uuid_path, state_keys)
        log(DEBUG, "Found server round indices %s", server_round_indices)
        if not server_round_indices and raise_error:
            raise NoCheckpointsFoundError
        if server_round_indices:
            resume_round = server_round_indices[resume_round]
    return resume_round


def import_checkpoints(
    remote_up_down: RemoteUploaderDownloader,
    cfg: BaseConfig,
    strategy: FedNesterov | FedMom | FedYogi | FedAdam | FedAvgEfficient,
) -> None:
    """Import checkpoints from a previous run for restoration based on configuration.

    This function validates the configuration parameters for checkpoints import,
    constructs the server path for the checkpoints, and calculates the resume round.
    It then calls `copy_old_checkpoints_to_new_run` to copy the checkpoints from
    the previous run to the current run's server path. The function ensures that all
    configuration parameters are not None and logs the restoration process.

    Parameters
    ----------
    remote_up_down : RemoteUploaderDownloader
        The uploader and downloader instance for interacting with remote storage.
    cfg : BaseConfig
        The configuration object containing run UUIDs, checkpoint information, and S3
        communication configuration.
    strategy : FedNesterov | FedMom | FedYogi | FedAdam | FedAvgEfficient
        The federated learning strategy being used.

    """
    assert cfg.run_uuid is not None, (
        "Cannot import checkpoint for restoration if `cfg.run_uuid` is None"
    )
    assert cfg.photon.restore_run_uuid is not None, (
        "Cannot import checkpoint for restoration if"
        " `cfg.photon.restore_run_uuid` is None"
    )
    assert cfg.photon.checkpoint is not None, (
        "Cannot import checkpoint for restoration if `cfg.photon.checkpoint` is None"
    )
    assert cfg.photon.comm_stack.s3 is not None, (
        "Cannot import checkpoint for restoration if `cfg.photon.comm_stack.s3` is None"
    )
    assert (
        cfg.s3_comm_config.bucket_name is not None
    ), """Cannot import checkpoint for restoration if `cfg.s3_comm_config.
        bucket_name` is None"""

    log(DEBUG, "Importing checkpoints from a previous run for restoration.")
    server_path = (
        f"s3://{cfg.s3_comm_config.bucket_name}/{cfg.photon.restore_run_uuid}/server/"
    )
    # Get the state keys from server and strategy
    state_keys = ("state.bin", *strategy.state_keys)
    # Interpret the resume round
    cfg.photon.resume_round = interpret_resume_round(
        cfg.photon.resume_round,
        server_path,
        state_keys=state_keys,
    )
    assert cfg.photon.resume_round is not None, (
        "Cannot import checkpoint for restoration if `cfg.photon.resume_round` is None"
    )
    log(DEBUG, "Restore round %s", cfg.photon.resume_round)

    copy_old_checkpoints_to_new_run(
        remote_up_down=remote_up_down,
        bucket_uri=f"s3://{cfg.s3_comm_config.bucket_name}",
        restore_run_data=(
            cfg.run_uuid,
            cfg.photon.restore_run_uuid,
            cfg.photon.resume_round,
            cfg.photon.resume_round * int(cfg.llm_config.local_steps.replace("ba", "")),
        ),
        n_total_clients=cfg.fl.n_total_clients,
        copy_client_checkpoints=cfg.photon.copy_client_checkpoints,
    )


def _upload_server_state(
    server_state: tuple[WandbHistory, int, float, int],
    client_state: dict[str | int, ClientState],
    remote_up_down: RemoteUploaderDownloader,
) -> None:
    """Upload the current server state to the S3 Object Store.

    This function serializes and uploads the current server state, including the
    federated learning round number, history, time elapsed, cumulative server steps,
    and client states, to the S3 Object Store. It creates a temporary directory to
    store the serialized state before uploading.

    Parameters
    ----------
    server_state : tuple[WandbHistory, int, float, int]
        A tuple containing the server state components: history, current round, time
    client_state : dict[str | int, ClientState]
        A dictionary mapping client identifiers to their states.
    remote_up_down : RemoteUploaderDownloader
        The uploader/downloader object for interacting with the S3 Object Store.

    """
    history, current_round, current_time_elapsed, server_steps_cumulative = server_state
    # Create a temporary directory
    temp_dir = TemporaryDirectory()
    # Create the server state dictionary containing light stuff
    current_server_state = {
        "server_round": current_round,
        "history": history,
        "time_offset": current_time_elapsed,
        "client_state": str({k: asdict(v) for k, v in client_state.items()}),
        "server_steps_cumulative": server_steps_cumulative,
    }
    # Dump the server state to disk and upload to the S3 Object Store
    with (Path(temp_dir.name) / "current_server_state.bin").open("wb") as f:
        pickle.dump(current_server_state, f)
    log(DEBUG, "Push server state to S3")
    upload_file_to_s3(
        remote_up_down,
        f"{current_round}/state.bin",
        Path(temp_dir.name) / "current_server_state.bin",
    )


def _upload_momentum_vector(
    current_round: int,
    momentum_vector: NDArrays,
    remote_up_down: RemoteUploaderDownloader,
    *,
    is_second_momentum: bool = False,
) -> None:
    """Upload the current momentum vector to the S3 Object Store.

    This function serializes and uploads the current momentum vector, used in
    optimization algorithms, to the S3 Object Store for the given federated learning
    round. It creates a temporary directory to store the serialized momentum vector
    before uploading and logs the time taken to dump the momentum
    vector to disk.

    Parameters
    ----------
    current_round : int
        The current federated learning round.
    momentum_vector : NDArrays
        The momentum vector to be uploaded.
    remote_up_down : RemoteUploaderDownloader
        The uploader/downloader object for interacting with the S3 Object Store.
    is_second_momentum : bool, optional
        Whether the momentum vector is the second momentum vector, by default False.

    """
    # Create a temporary directory
    temp_dir = TemporaryDirectory()
    log(DEBUG, "Dump momentum vector to disk")
    filename_no_ext = FIRST_MOMENTUM if not is_second_momentum else SECOND_MOMENTUM
    dump_mom_vec_time = time.time()
    dump_model_parameters_to_file(
        Path(temp_dir.name) / f"{filename_no_ext}.npz",
        momentum_vector,
    )
    log(
        DEBUG,
        "Push momentum vector to S3 Object Store. Time to dump to disk: %s",
        time.time() - dump_mom_vec_time,
    )
    upload_file_to_s3(
        remote_up_down,
        f"{current_round}/{filename_no_ext}.npz",
        Path(temp_dir.name) / f"{filename_no_ext}.npz",
    )


def _upload_model_parameters(
    parameters: Parameters,
    current_round: int,
    remote_up_down: RemoteUploaderDownloader,
) -> None:
    """Upload the model parameters to the S3 Object Store for the current round.

    This function serializes and uploads the model parameters to the S3 Object Store,
    organizing them by the current federated learning round. It first dumps them
    to a temporary file and then uploads this file.

    Parameters
    ----------
    parameters : Parameters
        The model parameters to be uploaded.
    current_round : int
        The current federated learning round.
    remote_up_down : RemoteUploaderDownloader
        The uploader/downloader object for interacting with the S3 Object Store.

    """
    # Create a temporary directory
    temp_dir = TemporaryDirectory()
    dump_model_time = time.time()
    dump_model_parameters_to_file(
        Path(temp_dir.name) / f"{MODEL_PARAMETERS}.npz",
        parameters_to_ndarrays(parameters),
    )
    log(
        DEBUG,
        "Push parameters to S3 Object Store. Time to dump to disk: %s",
        time.time() - dump_model_time,
    )
    upload_file_to_s3(
        remote_up_down,
        f"{current_round}/{MODEL_PARAMETERS}.npz",
        Path(temp_dir.name) / f"{MODEL_PARAMETERS}.npz",
    )


def upload_server_checkpoint(
    parameters: Parameters | None,
    server_state: tuple[WandbHistory, int, float | None, int | None],
    momenta: tuple[NDArrays | None, NDArrays | None],
    client_state: dict[str | int, ClientState] | None,
    remote_up_down: RemoteUploaderDownloader,
) -> None:
    """Upload the server checkpoint to the S3 Object Store.

    This function uploads various components of the server's state as part of the
    checkpointing process. It includes the model parameters, training history, current
    round, time elapsed, cumulative server steps, momentum vectors, and client states.
    Each component is uploaded separately, and only if it is not None.

    Parameters
    ----------
    parameters : Parameters | None
        The model parameters to be uploaded, if any.
    server_state : tuple[WandbHistory, int, float | None, int | None]
        The server state components to be uploaded. The tuple contains the training
    momenta : tuple[NDArrays | None, NDArrays | None]
        The momentum vectors to be uploaded, if any.
    client_state : dict[str | int, ClientState] | None
        The client states to be uploaded, if any.
    remote_up_down : RemoteUploaderDownloader
        The uploader/downloader object for interacting with the S3 Object Store.

    """
    # Uploading the server state
    history, current_round, current_time_elapsed, server_steps_cumulative = server_state
    momentum_vector, second_momentum_vector = momenta
    if (
        history
        and current_time_elapsed is not None
        and server_steps_cumulative is not None
        and client_state
    ):
        _upload_server_state(
            server_state=(
                history,
                current_round,
                current_time_elapsed,
                server_steps_cumulative,
            ),
            client_state=client_state,
            remote_up_down=remote_up_down,
        )
    # Dump and upload momentum vector if present
    if momentum_vector is not None:
        _upload_momentum_vector(
            current_round=current_round,
            momentum_vector=momentum_vector,
            remote_up_down=remote_up_down,
        )
    # Dump and upload second momentum vector if present
    if second_momentum_vector is not None:
        _upload_momentum_vector(
            current_round=current_round,
            momentum_vector=second_momentum_vector,
            remote_up_down=remote_up_down,
            is_second_momentum=True,
        )
    # Dump and upload model parameters
    if parameters is not None:
        _upload_model_parameters(
            parameters=parameters,
            current_round=current_round,
            remote_up_down=remote_up_down,
        )


def download_server_checkpoint(  # noqa: PLR0915, PLR0914
    cfg: BaseConfig,
    remote_up_down: RemoteUploaderDownloader,
    timeout: float = 0.5,
) -> tuple[
    Parameters,
    WandbHistory,
    int,
    float,
    int,
    dict[str | int, ClientState],
    NDArrays | None,
    NDArrays | None,
]:
    """Download the server checkpoint from the S3 Object Store.

    This function downloads the server checkpoint, including model parameters and
    potentially other state information, from the S3 Object Store. It checks for the
    existence of the server parameters file, waits until it is found (with a delay
    between checks defined by `timeout`), and then downloads the file. The downloaded
    parameters are then loaded from the file into memory. It also downloads the server
    state file, which contains the history, time offset, client states, and cumulative
    server steps. Finally, if a momentum vector is present, it is downloaded as well.

    Parameters
    ----------
    cfg : BaseConfig
        The configuration object containing S3 configurations and run UUID.
    remote_up_down : RemoteUploaderDownloader
        The uploader/downloader object for interacting with the S3 Object Store.
    timeout : float, optional
        The timeout in seconds to wait between checks for the server parameters file,
        by default 0.5.

    Returns
    -------
    tuple
        A tuple containing the loaded server checkpoint components. The exact components
        include model parameters, training history, current round, time elapsed,
        cumulative server steps, client states, and (potentially) the momentum vectors.

    """
    # Set the path to server checkpoints
    server_path = f"s3://{cfg.s3_comm_config.bucket_name}/{cfg.run_uuid}/server/"
    # Create a temporary directory
    temp_dir = TemporaryDirectory()

    # Model parameters
    # Check whether the server parameters exist
    file_found = False
    remote_file_name_no_ext = (
        server_path + f"{cfg.photon.resume_round}/{MODEL_PARAMETERS}"
    )
    while not file_found:
        file_found = validate_given_remote_path(
            remote_file_name_no_ext + ".bin",
        ) or validate_given_remote_path(remote_file_name_no_ext + ".npz")
        time.sleep(timeout)
    # Set the server parameters file names depending on the extension found
    remote_file_name = (
        f"{cfg.photon.resume_round}/{MODEL_PARAMETERS}.bin"
        if validate_given_remote_path(remote_file_name_no_ext + ".bin")
        else f"{cfg.photon.resume_round}/{MODEL_PARAMETERS}.npz"
    )
    local_file_name = (
        Path(temp_dir.name) / f"{MODEL_PARAMETERS}.bin"
        if validate_given_remote_path(remote_file_name_no_ext + ".bin")
        else Path(temp_dir.name) / f"{MODEL_PARAMETERS}.npz"
    )
    log(DEBUG, "Pull server parameters from S3 Object Store")
    # Download the parameters
    download_file_from_s3(remote_up_down, remote_file_name, local_file_name)
    log(DEBUG, "Read server parameters from disk")
    checkpoint_parameters = load_model_parameters_from_file(local_file_name)
    parameters = ndarrays_to_parameters(checkpoint_parameters)

    # Server state (history, time_offset, client_state, server_steps_cumulative)
    log(DEBUG, "Pull server state from S3 Object Store")
    # Download the server state from S3 Object Store
    download_file_from_s3(
        remote_up_down,
        f"{cfg.photon.resume_round}/state.bin",
        str(Path(temp_dir.name) / "current_server_state.bin"),
    )
    log(DEBUG, "Read server state from disk")
    with (Path(temp_dir.name) / "current_server_state.bin").open("rb") as f:
        server_state = pickle.load(f)  # noqa: S301
    start_round = server_state["server_round"]
    assert start_round == cfg.photon.resume_round, (
        "Server round mismatch with checkpoint"
    )
    history: WandbHistory = server_state["history"]
    if "client_state" in server_state:
        saved_client_state: dict[str | int, dict[str, Any]] = ast.literal_eval(
            server_state["client_state"],
        )
    else:
        # NOTE: This `local_steps_cumulative` is just used for backlogging and not by
        # any logic during training so we put a zero for now.
        log(
            WARNING,
            "No client state found in the checkpoint.We will put dummy values of zero.",
        )
        saved_client_state = {
            cid: {"local_steps_cumulative": 0} for cid in range(cfg.fl.n_total_clients)
        }
    # NOTE: We maintain partial compatibility across ClientState implementations
    # by only loading the attributes we actually need
    # this decision should be revisited at a late time

    # Automatically deduce the existing args from the ClientState class
    client_state_args = inspect.signature(ClientState.__init__).parameters.keys()

    # Filter and create client_state dictionary in one line
    client_state = {
        k: ClientState(**{attr: v[attr] for attr in v if attr in client_state_args})
        for k, v in saved_client_state.items()
    }
    time_offset = 0.0
    if "time_offset" in server_state:
        time_offset = server_state["time_offset"]
    if "server_steps_cumulative" in server_state:
        server_steps_cumulative = server_state["server_steps_cumulative"]
    else:
        # Make it back compatible with the previous versions
        server_steps_cumulative = max(
            *[
                _client_state.local_steps_cumulative
                for _client_state in client_state.values()
            ],
            0,
        )

    # Momentum vector
    momentum_vector: NDArrays | None = None
    remote_file_name_momentum = (
        server_path + f"{cfg.photon.resume_round}/{FIRST_MOMENTUM}.npz"
    )
    if "momentum" in server_state:
        log(DEBUG, "Get momentum vector from server state")
        momentum_vector = server_state["momentum"]
    elif validate_given_remote_path(remote_file_name_momentum):
        log(DEBUG, "Pull momentum from S3 Object Store")
        # Set the file names depending on the extension found
        remote_file_name = f"{cfg.photon.resume_round}/{FIRST_MOMENTUM}.npz"
        local_file_name = Path(temp_dir.name) / f"{FIRST_MOMENTUM}.npz"
        # Download the parameters
        download_file_from_s3(remote_up_down, remote_file_name, local_file_name)
        momentum_vector = load_model_parameters_from_file(local_file_name)

    # Momentum vector
    second_momentum_vector: NDArrays | None = None
    remote_file_name_momentum = (
        server_path + f"{cfg.photon.resume_round}/{SECOND_MOMENTUM}.npz"
    )
    if "momentum" in server_state:
        log(DEBUG, "Get momentum vector from server state")
        second_momentum_vector = server_state["momentum"]
    elif validate_given_remote_path(remote_file_name_momentum):
        log(DEBUG, "Pull momentum from S3 Object Store")
        # Set the file names depending on the extension found
        remote_file_name = f"{cfg.photon.resume_round}/{SECOND_MOMENTUM}.npz"
        local_file_name = Path(temp_dir.name) / f"{SECOND_MOMENTUM}.npz"
        # Download the parameters
        download_file_from_s3(remote_up_down, remote_file_name, local_file_name)
        second_momentum_vector = load_model_parameters_from_file(local_file_name)
    log(INFO, "Server checkpoint loaded")
    return (
        parameters,
        history,
        start_round,
        time_offset,
        server_steps_cumulative,
        client_state,
        momentum_vector,
        second_momentum_vector,
    )


def replace_remote_with_parameters_in_recordset(
    remote_uploader_downloader: RemoteUploaderDownloader | None,
    outgoing_message: Message,
    msg_str: str,
    comm_stack: CommStack,
) -> tuple[Message, list[ray.ObjectRef] | None]:
    """Replace parameters in the recordset with those from S3, shared memory, or Ray.

    This function checks the status of the task associated with the incoming message.
    If the task was successful and S3 communication is enabled, it downloads the
    parameters from S3 and updates the incoming message's recordset with these
    parameters. The function supports downloading parameters in either binary or NumPy
    compressed formats. If shared memory communication is enabled, it retrieves the
    parameters from shared memory and updates the message's recordset. If Ray
    communication is enabled, it retrieves the parameters from Ray object store and
    updates the message's recordset. If the task failed or none of the communication
    methods are enabled, the original message is returned without modification.

    Parameters
    ----------
    remote_uploader_downloader : RemoteUploaderDownloader | None
        The uploader/downloader instance for interacting with S3. Required if
        `comm_stack.s3` is True.
    outgoing_message : Message
        The message whose parameters are to be replaced with those downloaded from S3,
        retrieved from shared memory, or retrieved from Ray object store.
    msg_str : str
        A string identifier used to prefix keys in the message's content and to locate
        the specific parameters within the recordset.
    comm_stack : CommStack
        The communication stack configuration indicating which communication methods
        are enabled (S3, shared memory, Ray).

    Returns
    -------
    Message
        The modified message with parameters replaced by those downloaded from S3,
        retrieved from shared memory, or retrieved from Ray object store, or the
        original message if none of the communication methods are enabled or the task
        associated with the message failed.
    list[ray.ObjectRef] | None
        A list of Ray object references if Ray communication is enabled, or None if Ray
        communication is not enabled.

    Notes
    -----
    The function assumes the existence of `extract_s3_comm_config_from_configrecord`,
    `validate_given_remote_path`, `download_file_from_s3`, `ndarrays_to_parameters`,
    `load_model_parameters_from_file`, `parameters_to_parametersrecord`, `log`,
    `get_parameters_shm`, `is_shm_existing`, `set_parameters_shm`,
    `ModelParametersMetadata`, `ray`, and `ray.cloudpickle` functions/utilities, as well
    as the `DEBUG` constant for logging purposes. It also relies on the structure of the
    `Message` object and the `RemoteUploaderDownloader` interface for S3 interactions.

    Steps
    -----
    1. Extract the content of the incoming message.
    2. Check the status of the task associated with the message.
    3. If the task was successful and `comm_stack.s3` is True:
       a. Create a temporary directory for storing the downloaded parameters.
       b. Extract S3 communication configuration from the message.
       c. Check whether the server has uploaded the parameters.
       d. Download the parameters from S3.
       e. Update the message's recordset with the downloaded parameters.
    4. If `comm_stack.shm` is True:
       a. Get parameters metadata from the recordset.
       b. Create or get the shared memory for the parameters.
       c. Retrieve the parameters from shared memory.
       d. Update the message's recordset with the retrieved parameters.
    5. If `comm_stack.ray` is True:
       a. Get serialized ObjectRef from recordset.
       b. Unpickle and get object from Ray object store.
       c. Retrieve the parameters from Ray object store.
       d. Update the message's recordset with the retrieved parameters.
    6. If none of the communication methods are enabled, return the original message.

    """
    # Define the list of Ray object references
    ray_object_refs: list[ray.ObjectRef] | None = None
    # Extract the content of the incoming message
    recordset = outgoing_message.content
    # Check if it's necessary to download from S3
    if comm_stack.s3 and remote_uploader_downloader is not None:
        # Create a temporary directory for storing the downloaded parameters
        temp_dir: TemporaryDirectory = TemporaryDirectory()
        # Extract endpoint id from the content of the message
        endpoint_id, file_name, folder_name = extract_s3_comm_config_from_configrecord(
            recordset.configs_records[f"{msg_str}.s3_comm_config"],
        )
        # Set the file names
        local_file_name = Path(temp_dir.name) / f"{endpoint_id}_{file_name}.npz"
        dump_model_parameters_to_file(
            local_file_name,
            parameters_to_ndarrays(
                parametersrecord_to_parameters(
                    record=recordset.parameters_records[f"{msg_str}.parameters"],
                    keep_input=False,
                ),
            ),
        )
        # Upload the parameters to S3 Object Store
        upload_file_to_s3(
            remote_up_down=remote_uploader_downloader,
            remote_file_name=f"{folder_name}/{endpoint_id}/{file_name}.npz",
            local_file_name=local_file_name,
        )
        # Empty the recordset parameters
        recordset.parameters_records[f"{msg_str}.parameters"] = (
            parameters_to_parametersrecord(
                Parameters(tensors=[], tensor_type="empty"),
                keep_input=False,
            )
        )
        # Update the content of the message
        outgoing_message.content = recordset
        # NOTE: See if this is still necessary!
        # Check whether the server has uploaded the parameters
        remote_file_name_no_ext = (
            f"s3://{remote_uploader_downloader.remote_bucket_name}/"
            f"{remote_uploader_downloader.backend_kwargs['prefix']}/"
            f"{folder_name}/{endpoint_id}/{file_name}"
        )
        while not (
            validate_given_remote_path(
                remote_file_name_no_ext + ".bin",
            )
            or validate_given_remote_path(remote_file_name_no_ext + ".npz")
        ):
            time.sleep(0.5)
        log(
            DEBUG,
            "Node %s parameters have been pushed to the S3",
            endpoint_id,
        )
        return outgoing_message, ray_object_refs
    if comm_stack.shm:  # Using SharedMemory to communicate parameters
        # Get NDArrays from the parameters
        ndarrays_parameters = parameters_to_ndarrays(
            parametersrecord_to_parameters(
                record=recordset.parameters_records[f"{msg_str}.parameters"],
                keep_input=False,
            ),
        )
        # Get parameters metadata
        parameters_metadata = ModelParametersMetadata.from_ndarrays(ndarrays_parameters)
        # Create the parameters shared memory
        shm_name = str(
            recordset.configs_records[f"{msg_str}.s3_comm_config"]["endpoint_id"],
        )
        shm_parameters, _shm_parameters_sh = get_parameters_shm(
            parameters_metadata=parameters_metadata,
            create=not is_shm_existing(shm_name),
            name=shm_name,
        )
        # Set the parameters in the shared memory
        set_parameters_shm(shm_parameters, ndarrays_parameters)
        # Empty the recordset parameters
        recordset.parameters_records[f"{msg_str}.parameters"] = (
            parameters_to_parametersrecord(
                Parameters(tensors=[], tensor_type="empty"),
                keep_input=False,
            )
        )
        # Serialize ModelParametersMetadata and set it in the recordset
        parameters_metadata_dict = parameters_metadata.__dict__
        parameters_metadata_dict["dtypes"] = [
            str(v) for v in parameters_metadata_dict["dtypes"]
        ]
        recordset.configs_records[f"{msg_str}.parameters_metadata"] = ConfigsRecord(
            {"metadata": str(parameters_metadata_dict)},
        )
        # Update the content of the message
        outgoing_message.content = recordset
        return outgoing_message, ray_object_refs
    if comm_stack.ray:  # Using Ray to communicate parameters
        # Put object in Ray object store
        ray_object_refs = [
            ray.put(arr)
            for arr in parameters_to_ndarrays(
                parametersrecord_to_parameters(
                    record=recordset.parameters_records[f"{msg_str}.parameters"],
                    keep_input=False,
                ),
            )
        ]
        # Put serialized ObjectRef in recordset
        recordset.configs_records[f"{msg_str}.ray"] = ConfigsRecord(
            {
                "object_refs": [
                    ray.cloudpickle.dumps(_obj_ref) for _obj_ref in ray_object_refs
                ],
            },
        )
        # Empty the recordset parameters
        recordset.parameters_records[f"{msg_str}.parameters"] = (
            parameters_to_parametersrecord(
                Parameters(tensors=[], tensor_type="empty"),
                keep_input=False,
            )
        )
        # Update the content of the message
        outgoing_message.content = recordset
    # No translation performed as we assume the task failed
    return outgoing_message, ray_object_refs


def replace_parameters_in_recordset_with_remote(
    remote_uploader_downloader: RemoteUploaderDownloader | None,
    incoming_message: Message,
    msg_str: str,
    comm_stack: CommStack,
) -> Message:
    """Replace parameters in the recordset with those from S3, shared memory, or Ray.

    This function checks the status of the task associated with the incoming message.
    If the task was successful and S3 communication is enabled, it downloads the
    parameters from S3 and updates the incoming message's recordset with these
    parameters. The function supports downloading parameters in either binary or NumPy
    compressed formats. If shared memory communication is enabled, it retrieves the
    parameters from shared memory and updates the message's recordset. If Ray
    communication is enabled, it retrieves the parameters from Ray object store and
    updates the message's recordset. If the task failed or none of the communication
    methods are enabled, the original message is returned without modification.

    Parameters
    ----------
    remote_uploader_downloader : RemoteUploaderDownloader | None
        The uploader/downloader instance for interacting with S3. Required if
        `comm_stack.s3` is True.
    incoming_message : Message
        The message whose parameters are to be replaced with those downloaded from S3,
        retrieved from shared memory, or retrieved from Ray object store.
    msg_str : str
        A string identifier used to prefix keys in the message's content and to locate
        the specific parameters within the recordset.
    comm_stack : CommStack
        The communication stack configuration indicating which communication methods
        are enabled (S3, shared memory, Ray).

    Returns
    -------
    Message
        The modified message with parameters replaced by those downloaded from S3,
        retrieved from shared memory, or retrieved from Ray object store, or the
        original message if none of the communication methods are enabled or the task
        associated with the message failed.

    Notes
    -----
    The function assumes the existence of `extract_s3_comm_config_from_configrecord`,
    `validate_given_remote_path`, `download_file_from_s3`, `ndarrays_to_parameters`,
    `load_model_parameters_from_file`, `parameters_to_parametersrecord`, `log`,
    `get_parameters_shm`, `is_shm_existing`, `set_parameters_shm`,
    `ModelParametersMetadata`, `ray`, and `ray.cloudpickle` functions/utilities, as well
    as the `DEBUG` constant for logging purposes. It also relies on the structure of the
    `Message` object and the `RemoteUploaderDownloader` interface for S3 interactions.

    Steps
    -----
    1. Extract the content of the incoming message.
    2. Check the status of the task associated with the message.
    3. If the task was successful and `comm_stack.s3` is True:
       a. Create a temporary directory for storing the downloaded parameters.
       b. Extract S3 communication configuration from the message.
       c. Check whether the server has uploaded the parameters.
       d. Download the parameters from S3.
       e. Update the message's recordset with the downloaded parameters.
    4. If `comm_stack.shm` is True:
       a. Get parameters metadata from the recordset.
       b. Create or get the shared memory for the parameters.
       c. Retrieve the parameters from shared memory.
       d. Update the message's recordset with the retrieved parameters.
    5. If `comm_stack.ray` is True:
       a. Get serialized ObjectRef from recordset.
       b. Unpickle and get object from Ray object store.
       c. Retrieve the parameters from Ray object store.
       d. Update the message's recordset with the retrieved parameters.
    6. If none of the communication methods are enabled, return the original message.

    """
    # Extract the content of the incoming message
    recordset = incoming_message.content
    if _extract_status_from_recordset(msg_str, recordset).code != Code.OK:
        # No translation performed as we assume the task failed
        return incoming_message
    # Check if it's necessary to download from S3
    if comm_stack.s3 and remote_uploader_downloader is not None:
        # Create a temporary directory for storing the downloaded parameters
        temp_dir: TemporaryDirectory = TemporaryDirectory()
        # Extract endpoint id from the content of the message
        endpoint_id, file_name, folder_name = extract_s3_comm_config_from_configrecord(
            recordset.configs_records[f"{msg_str}.s3_comm_config"],
        )
        # Check whether the server has uploaded the parameters
        remote_file_name_no_ext = (
            f"s3://{remote_uploader_downloader.remote_bucket_name}/"
            f"{remote_uploader_downloader.backend_kwargs['prefix']}/"
            f"{folder_name}/{endpoint_id}/{file_name}"
        )
        while not (
            validate_given_remote_path(
                remote_file_name_no_ext + ".bin",
            )
            or validate_given_remote_path(remote_file_name_no_ext + ".npz")
        ):
            time.sleep(0.5)
        # Set the file names depending on the extension found
        local_file_name = (
            Path(temp_dir.name) / f"tmp-{endpoint_id}.bin"
            if validate_given_remote_path(remote_file_name_no_ext + ".bin")
            else Path(temp_dir.name) / f"tmp-{endpoint_id}.npz"
        )
        download_file_from_s3(
            remote_up_down=remote_uploader_downloader,
            remote_file_name=(
                f"{folder_name}/{endpoint_id}/{file_name}.bin"
                if validate_given_remote_path(remote_file_name_no_ext + ".bin")
                else f"{folder_name}/{endpoint_id}/{file_name}.npz"
            ),
            local_file_name=local_file_name,
        )
        recordset.parameters_records[f"{msg_str}.parameters"] = (
            parameters_to_parametersrecord(
                ndarrays_to_parameters(
                    load_model_parameters_from_file(local_file_name),
                ),
                keep_input=False,
            )
        )
        incoming_message.content = recordset
        log(
            DEBUG,
            "Node %s parameters have been read from disk and assigned to the Message",
            endpoint_id,
        )
        return incoming_message
    if comm_stack.shm:  # Using SharedMemory to communicate parameters
        # Get parameters metadata
        parameters_metadata_dict = ast.literal_eval(
            str(
                recordset.configs_records[f"{msg_str}.parameters_metadata"]["metadata"],
            ),
        )
        parameters_metadata_dict["dtypes"] = [
            np.dtype(v) for v in parameters_metadata_dict["dtypes"]
        ]
        # Create the parameters shared memory
        shm_name = str(
            recordset.configs_records[f"{msg_str}.s3_comm_config"]["endpoint_id"],
        )
        shm_parameters, _shm_parameters_sh = get_parameters_shm(
            parameters_metadata=ModelParametersMetadata(**parameters_metadata_dict),
            create=False,
            name=shm_name,
        )
        recordset.parameters_records[f"{msg_str}.parameters"] = (
            parameters_to_parametersrecord(
                ndarrays_to_parameters(
                    [arr.astype(np.float32) for arr in shm_parameters],
                ),
                keep_input=False,
            )
        )
        incoming_message.content = recordset
        return incoming_message
    if comm_stack.ray:  # Using Ray to communicate parameters
        # Get serialized ObjectRef from recordset
        serialized_obj_refs = recordset.configs_records[f"{msg_str}.ray"]["object_refs"]
        assert isinstance(
            serialized_obj_refs,
            list,
        ), "Serialized object references must be a list"
        # Unpickle and get object from Ray object store
        obj_refs = [ray.cloudpickle.loads(_obj_ref) for _obj_ref in serialized_obj_refs]
        _obj_refs_ready, _obj_refs_not = ray.wait(obj_refs, num_returns=len(obj_refs))
        recordset.parameters_records[f"{msg_str}.parameters"] = (
            parameters_to_parametersrecord(
                ndarrays_to_parameters(
                    [np.array(ray.get(_obj_ref)) for _obj_ref in obj_refs],
                ),
                keep_input=False,
            )
        )
        incoming_message.content = recordset
    # No translation performed as we assume the task failed
    return incoming_message


def get_file_from_path(
    input_file_path: str,
    run_uuid: str,
    s3_comm_config: S3CommConfig,
    tmp_dir: TemporaryDirectory,
) -> Path:
    """Retrieve a file from a given path, which can be either a local path or an S3 URI.

    This function interprets the input file path to determine whether it's a local path
    or an S3 URI. If it is an S3 URI, the function downloads the file from the specified
    S3 bucket to a temporary directory. If it is a local path, the function verifies the
    existence of the file. The function returns the local path to the file.

    Parameters
    ----------
    input_file_path : str
        The path to the file to be retrieved. Can be a local file path or an S3 URI.
    run_uuid : str
        The unique identifier for the run, used for S3 operations.
    s3_comm_config : S3CommConfig
        The S3 communication configuration.
    tmp_dir : TemporaryDirectory
        The temporary directory for storing the downloaded file.

    Returns
    -------
    Path
        The local file path to the retrieved file.

    Raises
    ------
    ValueError
        If the backend specified in the URI is unknown.

    """
    # Interpret the URI
    backend, bucket_name, remote_file_name = parse_uri(input_file_path)
    local_file_path: Path | None = None
    if backend == "s3":
        log(
            INFO,
            "Downloading model %s from S3 bucket %s",
            remote_file_name,
            bucket_name,
        )
        # Create RemoteUploaderDownloader object
        remote_up_down = create_remote_up_down(
            bucket_name=bucket_name,
            prefix="",
            run_uuid=run_uuid,
            num_attempts=5,
            client_config=OmegaConf.to_container(
                s3_comm_config.backend_kwargs.client_config,
            ),  # type: ignore[reportArgumentType, arg-type]
        )
        local_file_path = Path(tmp_dir.name) / (
            "checkpoint" + Path(remote_file_name).suffix
        )
        download_file_from_s3(remote_up_down, remote_file_name, local_file_path)
    elif not backend:
        log(
            INFO,
            "File path %s is local.",
            Path(input_file_path),
        )
        local_file_path = Path(input_file_path)
    else:
        msg = f"Unknown backend: {backend}"
        raise ValueError(msg)
    assert local_file_path is not None, "Local file path is None"
    assert local_file_path.exists(), f"Local file path {local_file_path} does not exist"
    return local_file_path


def load_pretrained_model_from_path(
    pretrained_model_path: str,
    run_uuid: str,
    s3_comm_config: S3CommConfig,
    trainer: Trainer,
) -> None:
    """Load a pretrained model from a specified path and set it to the trainer.

    This function supports loading models from both local file paths and S3 URIs.
    It downloads the model if the path points to an S3 bucket, and then sets the
    parameters in the provided trainer object.

    Parameters
    ----------
    pretrained_model_path : str
        Path to pre-trained model
    run_uuid : str
        The unique identifier for the run, used for S3 operations.
    s3_comm_config : S3CommConfig
        The S3 communication configuration.
    trainer : Trainer
        The trainer object to which the pretrained model parameters are set.

    """
    log(
        INFO,
        "Loading pretrained model from %s",
        pretrained_model_path,
    )
    # Create a temporary directory for storing the downloaded parameters
    tmp_dir = TemporaryDirectory()
    # Load the local file path
    local_file_path = get_file_from_path(
        tmp_dir=tmp_dir,
        input_file_path=pretrained_model_path,
        run_uuid=run_uuid,
        s3_comm_config=s3_comm_config,
    )
    initial_parameters = load_model_parameters_from_file(local_file_path)
    set_trainer_params_from_ndarrays(initial_parameters, trainer)


def get_num_batches_from_checkpoint_name(checkpoint_name: str) -> int:
    """Extract the number of batches from the checkpoint name.

    The checkpoint name is expected to be in the format:
    ep{n_epochs}-ba{n_batches}-rank{rank}.pt

    Parameters
    ----------
    checkpoint_name : str
        The name of the checkpoint file.

    Returns
    -------
        int: The number of batches extracted from the checkpoint name.

    Raises
    ------
        ValueError: If the checkpoint name does not match the expected format.

    """
    match = re.search(r"-ba(\d+)-", checkpoint_name)
    if match:
        return int(match.group(1))
    msg = f"Invalid checkpoint name format: {checkpoint_name}"
    raise ValueError(msg)


def obtain_sorted_runs(run_uuid_path: str, state_keys: tuple[str, ...]) -> list[int]:
    """Obtain the sorted runs from the server path.

    This function lists the objects in the specified run UUID path, extracts unique run
    numbers from the paths under `{run_uuid_path}/server/`, and filters them based on
    the provided state keys. It returns the sorted list of valid run numbers.

    Parameters
    ----------
    run_uuid_path : str
        The path to the run UUID root.
    state_keys : tuple[str, ...]
        The state keys to check in the paths. Keys are intended to be the prefixes of
        any file name. For example, if the keys are ("state", "model"), then the
        function will return any path that starts with "state" and "model".

    Returns
    -------
    list[int]
        The sorted list of valid run numbers.

    Example
    -------
    >>> run_uuid_path = "s3://mybucket/myfolder"
    >>> state_keys = ("state", "model")
    >>> sorted_runs = obtain_sorted_runs(run_uuid_path, state_keys)
    >>> print(sorted_runs)
    [1, 2, 3]

    Notes
    -----
    This function uses the `list_objects` function to list objects in the given path.
    It extracts unique run numbers from the paths under `{run_uuid_path}/server/` and
    filters them based on the provided state keys.

    """
    _is_remote, remote_objects = list_objects(run_uuid_path)

    # Extract unique run numbers
    run_numbers = {
        int(reg.group(1))
        for path in remote_objects
        if (reg := re.search(r"server/(\d+)/.*$", path)) is not None
    }

    valid_runs = set()

    for run in run_numbers:
        # Filter paths for the current run
        run_paths = [path for path in remote_objects if f"server/{run}/" in path]

        # Check if all state_keys are present in the paths for this run
        if all(
            any(state_key in path for path in run_paths) for state_key in state_keys
        ):
            valid_runs.add(run)

    return sorted(valid_runs)


def delete_clients_checkpoints(run_uuid_path: str, end_idx: int | None = -1) -> None:
    """Delete client checkpoints from a specified path. Can be either a local or remote.

    This function deletes the specified client checkpoints from the provided run UUID
    path. It lists all the objects in the path, extracts unique client IDs, and removes
    the corresponding checkpoints for each client based on the `end_idx` parameter. The
    function supports both local and remote S3 paths.

    Parameters
    ----------
    run_uuid_path : str
        The path to the run UUID, which can be either a local path or a remote S3 path.
    end_idx : int, optional
        The index up to which checkpoints should be deleted. Defaults to -1, which means
        all checkpoints except for the last.

    Example
    -------
    >>> delete_clients_checkpoints("s3://mybucket/myfolder", end_idx=5)
    >>> delete_clients_checkpoints("/local/path/to/folder", end_idx=5)

    Notes
    -----
    This function uses the `list_objects` function to list objects in the given path.
    For remote S3 paths, it uses the `delete_object` function to delete objects from the
    S3 bucket. For local paths, it uses the `delete_object` function to delete local
    files.

    """
    # List all the remote objects in the run UUID path
    _is_remote, remote_objects = list_objects(run_uuid_path)
    # Extract unique client IDs from the remote objects
    unique_client_ids = {
        int(reg.group(1))
        for path in remote_objects
        if (reg := re.search(r"client_(\d+)/.*$", path)) is not None
    }
    for client_id in unique_client_ids:
        # List all the remote objects for the client
        is_remote, client_remote_objects = list_objects(
            f"{run_uuid_path}/client_{client_id}/",
        )
        # Remove symlinks from the list of files
        client_remote_objects = [
            cro for cro in client_remote_objects if not cro.endswith(".symlink")
        ]
        # Sort by number of batches the client trained on
        sorted_client_objects = sorted(
            client_remote_objects,
            key=get_num_batches_from_checkpoint_name,
        )
        # Delete only the last `end_idx` checkpoints
        objects_to_remove = sorted_client_objects[:end_idx]
        for object_to_remove in objects_to_remove:
            if is_remote:
                # Parse the URI to extract the backend and bucket name
                backend, bucket_name, _prefix = parse_uri(run_uuid_path)
                delete_object(f"{backend}://{bucket_name}/{object_to_remove}")
            else:
                delete_object(object_to_remove)


def delete_rounds(
    run_uuid_path: str,
    state_keys: tuple[str, ...],
    end_idx: int | None = -1,
) -> None:
    """List objects in a local/(remote s3) given path.

    This function attempts to list objects in the specified path. If the path is remote
    S3 path, it uses the `list_remote_objects` function to list the objects. If the path
    is a local path, it lists all files recursively within the directory.

    Parameters
    ----------
    run_uuid_path : str
        The path to list objects from. This can be either a local path or a remote S3
        path.
    state_keys : tuple[str, ...]
        The state keys to check in the paths. Keys are intended to be the prefixes of
        any file name. For example, if the keys are ("state", "model"), then the
        function will return any path that starts with "state" and "model".
    end_idx : int, optional
        The index up to which rounds should be deleted. Defaults to -1, which means all

    Example
    -------
    >>> is_remote, objects = list_objects("s3://mybucket/myfolder")
    >>> print(is_remote)
    True
    >>> print(objects)
    ['s3://mybucket/myfolder/file1.txt', 's3://mybucket/myfolder/file2.txt']

    >>> is_remote, objects = list_objects("/local/path/to/folder")
    >>> print(is_remote)
    False
    >>> print(objects)
    ['/local/path/to/folder/file1.txt', '/local/path/to/folder/file2.txt']

    Notes
    -----
    This function uses the `list_remote_objects` function to list objects in a remote
    S3 path. For local paths, it uses the `Path.rglob` method to recursively list all
    files in the directory.

    """
    # List all the federated rounds in the run UUID path
    sorted_rounds = obtain_sorted_runs(run_uuid_path, state_keys)
    # Delete only the last `end_idx` rounds
    rounds_to_delete = sorted_rounds[:end_idx]
    for round_to_delete in rounds_to_delete:
        # List all the remote objects for the server at the round specified
        is_remote, objects_to_remove = list_objects(
            f"{run_uuid_path}/server/{round_to_delete}/",
        )
        # Remove the objects found
        for object_to_remove in objects_to_remove:
            if is_remote:
                # Parse the URI to extract the backend and bucket name
                backend, bucket_name, _prefix = parse_uri(run_uuid_path)
                delete_object(f"{backend}://{bucket_name}/{object_to_remove}")
            else:
                delete_object(object_to_remove)


def delete_remote_object(object_name: str) -> None:
    """Delete an object from an S3 bucket.

    This function deletes an object from an S3 bucket using the provided object name.
    It creates an S3 object store from the object name, parses the URI to extract the
    prefix, and then deletes the object from the object store.

    Parameters
    ----------
    object_name : str
        The name of the object to delete.

    Raises
    ------
        ValueError: If the object name is not a valid URI or if the object cannot be
            deleted.

    """
    # Create an object store from the object name
    object_store: S3ObjectStore | None = maybe_create_object_store_from_uri(object_name)  # type: ignore[assignment,reportAssignmentType]
    if object_store is None:
        msg = f"Invalid object name: {object_name}"
        raise ValueError(msg)
    # Parse the URI to extract the prefix to use as the key to delete the file
    _backend, _bucket_name, prefix = parse_uri(object_name)
    # Delete the object from the object store
    object_store.client.delete_object(
        Bucket=object_store.bucket,
        Key=object_store.get_key(prefix),
    )


def copy_old_checkpoints_to_new_run(
    remote_up_down: RemoteUploaderDownloader,
    bucket_uri: str,
    restore_run_data: tuple[str, str, int, int],
    n_total_clients: int | None,
    *,
    copy_client_checkpoints: bool = True,
) -> None:
    """Copy old checkpoints to the new run folder.

    Parameters
    ----------
    remote_up_down : RemoteUploaderDownloader
        The remote uploader/downloader object.
    bucket_uri : str
        The URI of the bucket.
    restore_run_data : tuple[str, str, int, int]
        A tuple containing the run UUID, restore run UUID, restore run round, and
        restore run step.
    n_total_clients : int | None
        The total number of clients expected in the new run.
    copy_client_checkpoints : bool, optional
        A flag indicating whether to copy the client checkpoints. Defaults to True.

    Raises
    ------
        NotImplementedError: If the backend is not an S3ObjectStore.
        ValueError: If the old run folder or the new run folder is not found.

    """
    run_uuid, restore_run_uuid, restore_run_round, restore_run_step = restore_run_data
    if not isinstance(remote_up_down.remote_backend, S3ObjectStore):
        msg = "Support for resuming from non-S3 backends is not yet implemented."
        raise NotImplementedError(
            msg,
        )

    new_run_folder = bucket_uri + f"/{run_uuid}"
    old_run_folder = bucket_uri + f"/{restore_run_uuid}"
    if (old_run_val := validate_given_remote_path(old_run_folder)) and (
        _new_run_val := validate_given_remote_path(new_run_folder)
    ):
        state_bin = restore_run_uuid + f"/server/{restore_run_round}/state.bin"

        momentum_vec = (
            old_run_folder + f"/server/{restore_run_round}/{FIRST_MOMENTUM}.npz"
        )

        parameters_no_ext = (
            old_run_folder + f"/server/{restore_run_round}/{MODEL_PARAMETERS}"
        )
        parameters = (
            parameters_no_ext.replace(bucket_uri + "/", "") + ".bin"
            if validate_given_remote_path(parameters_no_ext + ".bin")
            else (parameters_no_ext.replace(bucket_uri + "/", "") + ".npz")
        )

        # Extract the client and the batches
        # NOTE: (?:\d+) means a do-not-capture group
        # As such we allow any number of epochs without extracting
        # The number of epochs
        client_path_batches = sorted(
            [
                (
                    path,
                    int(reg.group(1)),
                    int(reg.group(2)),
                )
                for path in list_remote_objects(old_run_folder)
                if (reg := re.search(r"client_(\d+)/ep(?:\d+)-ba(\d+)", path))
                is not None
            ],
            key=operator.itemgetter(1, 2),
        )

        # For each client, choose the latest checkpoint
        # That is consistent with the step of the resume round
        # groupby acts like an sql groupby
        client_paths = [
            list(filter(lambda x: x[2] <= restore_run_step, group))[-1][0]
            for _, group in groupby(client_path_batches, key=operator.itemgetter(1))
        ]

        if (
            copy_client_checkpoints
            and n_total_clients is not None
            and (found_clients := len(client_paths)) != n_total_clients
        ):
            msg = (
                f"Found {found_clients} clients in the old run folder {old_run_folder},"
                f" but expected {n_total_clients}."
            )
            raise ValueError(
                msg,
            )

        paths_to_copy = [state_bin, parameters]

        if validate_given_remote_path(momentum_vec):
            paths_to_copy.append(momentum_vec.replace(bucket_uri + "/", ""))
        else:
            log(
                DEBUG,
                f"Could not find momentum vector to copy from {momentum_vec}",
            )

        if copy_client_checkpoints:
            paths_to_copy.extend(client_paths)

        for path in paths_to_copy:
            target_key = path.replace(restore_run_uuid, run_uuid)
            log(DEBUG, "Copying %s to %s", path, target_key)
            remote_up_down.remote_backend.client.copy(
                {"Bucket": remote_up_down.remote_backend.bucket, "Key": path},
                remote_up_down.remote_backend.bucket,
                target_key,
            )

    else:
        if not old_run_val:
            msg = (
                f"Could not find the old run folder {old_run_folder} to copy"
                " checkpoints."
            )
            raise ValueError(
                msg,
            )
        msg = f"Could not find the new run folder {new_run_folder} to copy checkpoints."
        raise ValueError(
            msg,
        )


def cleanup_checkpoints(
    run_uuid: str,
    strategy_state_keys: tuple[str, ...],
    end_idx: int | None = -1,
) -> None:
    """Cleanup checkpoints.

    Parameters
    ----------
    run_uuid : str
        The UUID of the run.
    strategy_state_keys : str
        The keys of the strategy state.
    end_idx : int, optional
        The index up to which checkpoints should be deleted. Defaults to -1, which means
        all checkpoints except for the last.

    """
    # Clean up checkpoints if asked to

    # Remove old clients checkpoints from the S3 Object Store
    delete_clients_checkpoints(
        run_uuid_path=f"s3://checkpoints/{run_uuid}",
        end_idx=end_idx,
    )
    # Remove old server checkpoints from the S3 Object Store
    delete_rounds(
        run_uuid_path=f"s3://checkpoints/{run_uuid}",
        state_keys=("state.bin", *strategy_state_keys),
        end_idx=end_idx,
    )
