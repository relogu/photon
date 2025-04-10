"""Provides functionality for manipulating MosaicML configs."""

import ast
import copy
import json
import operator
import os
import re
import tempfile
import warnings
from dataclasses import asdict, dataclass
from logging import DEBUG, INFO, WARNING
from pathlib import Path
from typing import Any

from composer.devices import Device, DeviceCPU, DeviceGPU
from flwr.common.logger import log
from llmfoundry.command_utils.train import validate_config
from llmfoundry.utils.config_utils import (
    TRAIN_CONFIG_KEYS,
    TrainConfig,
    make_dataclass_and_log_config,
)
from llmfoundry.utils.registry_utils import import_file
from omegaconf import DictConfig, OmegaConf

from photon.conf.base_schema import S3CommConfig
from photon.server.s3_utils import list_objects
from photon.utils import (
    create_remote_up_down,
    download_file_from_s3,
    get_n_cpu_cores,
    get_n_cuda_devices,
    merge_freq_dicts,
)

# Constant for the frequency dictionary name
FREQ_DICT_NAME = "1_gram.json"
FREQ_DICT_CACHE_NAME = "_freq_dict.json"


@dataclass
class StreamDict:
    """Dataclass for stream dictionary."""

    remote: str | None = None
    local: str | None = None
    split: str | None = None
    proportion: float | None = None
    repeat: float | None = None
    choose: int | None = None
    download_retry: int | None = None
    download_timeout: float | None = None
    validate_hash: str | None = None
    keep_zip: bool | None = None


def get_train_config(  # noqa: PLR0913
    cfg: DictConfig,
    cid: int | str | None,
    log_name: str | None = None,
    *,
    force_cpu: bool = False,
    no_data_loading: bool = False,
    split_eval: bool = False,
) -> tuple[
    TrainConfig,
    DeviceGPU | DeviceCPU | None,
    dict[str, Any],
    dict[str, Any] | None,
]:
    """Generate the training configuration for a client.

    This function creates a deep copy of the provided configuration, imports any user-
    provided code, and sets up the training configuration for a client. It handles
    device settings, data loading configurations, logging configurations, and validation
    of the final configuration.

    Parameters
    ----------
    cfg : DictConfig
        The base configuration dictionary.
    cid : int | str | None
        The client ID, which can be an integer, string, or None.
    log_name : str | None, optional
        The name for logging. Default is None.
    force_cpu : bool, optional
        If True, forces the use of CPU even if GPUs are available. Default is False.
    no_data_loading : bool, optional
        If True, skips data loading configuration. Default is False.
    split_eval : bool, optional
        If True, splits evaluation data. Default is False.

    Returns
    -------
        tuple[
            TrainConfig,
            DeviceGPU | DeviceCPU | None,
            dict[str, Any], dict[str, Any] | None
        ]
            The training configuration object, the device, the configuration to log, and
            the ICL tasks config dict.

    Raises
    ------
        AssertionError
            If there are issues with the provided configuration or environment settings.

    """
    internal_cfg = copy.deepcopy(cfg)
    # NOTE: I got this from the original script in the llm-foundry repository. Assess
    # whether we need to keep it or not
    code_paths = internal_cfg.get("code_paths", [])
    # Import any user provided code
    for code_path in code_paths:
        import_file(code_path)

    # NOTE: We need to extract a set of global variables from the configuration object
    # to prevent the dataclass creator to crash
    internal_cfg.pop("data_local", None)
    internal_cfg.pop("data_remote", None)
    internal_cfg.pop("global_seed", None)
    internal_cfg.pop("local_steps", None)
    internal_cfg.pop("name", None)
    # NOTE: This contains OUR global parameters for the ICL tasks that the
    # `make_dataclass_and_log_config` cannot interpret, so we need to pop it
    icl_tasks_config_dict: dict[str, Any] | None = internal_cfg.pop(
        "icl_tasks_config",
        None,
    )
    # NOTE: The list of ICL tasks has been already resolved in the appropriate entry of
    # the `internal_cfg` object to be interpretable by the
    # `make_dataclass_and_log_config` function. We will later modify in-place the
    # relevant parameters to include what is contained in the root of the
    # `icl_tasks_config_dict` object
    if icl_tasks_config_dict is not None:
        icl_tasks_config_dict.pop("icl_tasks", None)

    adapt_train_batch_size_to_num_devices(internal_cfg)
    logged_cfg, train_cfg = make_dataclass_and_log_config(
        internal_cfg,
        TrainConfig,
        TRAIN_CONFIG_KEYS,
        transforms="all",
        icl_tasks_required=internal_cfg.get("icl_tasks", None) is not None,
    )

    # Set the device in case multiple GPUs are requested to be
    # independent and not collaborative. If `device == None` the
    # Trainer will automatically initialize PyTorch Distributed
    # with the parameters from the environmental variables.
    visible_devices = ast.literal_eval(str(os.getenv("APPOINTED_CUDA_DEVICE", "null")))
    # The worker has been appointed a single GPU
    if type(visible_devices) is int and not force_cpu:
        device: DeviceGPU | DeviceCPU | None = DeviceGPU(device_id=int(visible_devices))
    # The worker has been appointed all GPUs available
    elif type(visible_devices) is tuple and not force_cpu:
        assert len(visible_devices) > 1
        device = None
    # The worker is in a CPU-only environment
    else:
        if not force_cpu:
            assert visible_devices is None
        device = DeviceCPU()
    # Set the `num_workers` for data loaders
    set_n_workers_dataloaders(train_cfg=train_cfg, device=device)
    # Set the data configuration for the client
    if not no_data_loading:
        client_set_data_config(train_cfg=train_cfg, cid=cid, split_eval=split_eval)
    # Apply dataset defaults
    set_dataset_default_params(train_cfg)
    # Set (optional) WandB and/or Tensorboard logging parameters for accommodating
    # federated learning Ops
    log_name = f"_client_{cid}" if log_name is None else log_name
    set_client_wandb_logger(train_cfg, log_name)
    set_client_tensorboard_logger(train_cfg, log_name)

    # Check for incompatibilities between the model and data loaders
    validate_config(train_cfg)

    if not train_cfg.callbacks:
        train_cfg.callbacks = {}
    use_async_eval = any("async_eval" in name for name in train_cfg.callbacks)

    # Dataloaders
    if not no_data_loading:
        assert train_cfg.train_loader["dataset"] is not None, (
            "Dataset for train loader is not set."
        )

    # Evaluation
    if use_async_eval and train_cfg.eval_first:
        warnings.warn(
            "AsyncEval callback does not support eval_first=True. Ignoring.",
            stacklevel=2,
        )
        train_cfg.eval_first = False

    return train_cfg, device, logged_cfg, icl_tasks_config_dict


def set_icl_tasks_root_dir(
    icl_tasks_listconfig: list[dict[str, Any]],
    root_dir: str,
) -> None:
    """Update the dataset URIs in the ICL tasks configuration with a new root directory.

    This function iterates over a list of ICL (In-Context Learning) task configurations
    and updates the `dataset_uri` for each task by prepending the specified root dir.

    Parameters
    ----------
    icl_tasks_listconfig : list[dict[str, Any]]
        A list of dictionaries, each representing an ICL task configuration. Each dict
        must contain a `dataset_uri` key.
    root_dir : str
        The new root directory to prepend to the existing `dataset_uri` values.

    Example
    -------
    >>> icl_tasks_listconfig = [
    ...     {"dataset_uri": "path/to/dataset1"},
    ...     {"dataset_uri": "path/to/dataset2"},
    ... ]
    >>> root_dir = "/new/root/dir"
    >>> set_icl_tasks_root_dir(icl_tasks_listconfig, root_dir)
    >>> print(icl_tasks_listconfig)
    [
        {'dataset_uri': '/new/root/dir/path/to/dataset1'},
        {'dataset_uri': '/new/root/dir/path/to/dataset2'},
    ]

    """
    for icl_task in icl_tasks_listconfig:
        old_dataset_uri = icl_task["dataset_uri"]
        icl_task["dataset_uri"] = root_dir + "/" + old_dataset_uri


def preprocess_stream_paths(dataset_config: DictConfig) -> tuple[str, str, str]:
    """Preprocess the stream paths from the dataset configuration.

    This function extracts and processes the root remote path, root local path, and
    split from the dataset configuration. It ensures that the paths have trailing
    slashes if they are not empty.

    Parameters
    ----------
    dataset_config : DictConfig
        The dataset configuration object containing the paths and split information.

    Returns
    -------
    tuple[str, str, str]
        A tuple containing the processed root remote path, root local path, and split.

    Example
    -------
    >>> from omegaconf import OmegaConf
    >>> dataset_config = OmegaConf.create({
    ...     "root_remote": "s3://bucket/path",
    ...     "root_local": "/local/path",
    ...     "split": "train"
    ... })
    >>> root_remote, root_local, split = preprocess_stream_paths(dataset_config)
    >>> print(root_remote, root_local, split)
    s3://bucket/path/ /local/path/ train

    """
    root_remote = dataset_config.pop("root_remote", "")
    root_remote = root_remote + "/" if root_remote else root_remote
    root_local = dataset_config.pop("root_local", "")
    root_local = root_local + "/" if root_local else root_local
    split = dataset_config.pop("split", "")
    return root_remote, root_local, split


def concatenate_streams(clients_streams: list[dict[str, Any]]) -> dict[str, Any]:
    """Concatenate client streams into a single dictionary.

    This function concatenates the streams from multiple clients into a single dict,
    assigning unique keys to each stream.

    Parameters
    ----------
    clients_streams : list[dict[str, Any]]
        A list of dictionaries, each containing client stream configurations.

    Returns
    -------
    dict[str, Any]
        A dictionary containing all concatenated client streams with unique keys.

    Example
    -------
    >>> clients_streams = [
    ...     {"client_streams": {"stream1": {"local": "/path1"}}},
    ...     {"client_streams": {"stream2": {"local": "/path2"}}}
    ... ]
    >>> concatenated_streams = concatenate_streams(clients_streams)
    >>> print(concatenated_streams)
    {'stream_0': {'local': '/path1'}, 'stream_1': {'local': '/path2'}}

    """
    counter = 0
    current_client_stream: dict[str, Any] = {}
    for client_stream in clients_streams:
        assert "client_streams" in client_stream, "Client streams not found."
        client_streams = client_stream["client_streams"]
        assert isinstance(
            client_streams,
            dict,
        ), f"Client streams is not a dict but a {type(client_streams)}."
        for stream in client_streams.values():
            current_client_stream |= {f"stream_{counter}": stream}
            counter += 1

    return current_client_stream


def get_actual_stream(
    root_local: str,
    root_remote: str,
    split: str,
    current_client_stream: dict[str, Any],
) -> dict[str, StreamDict]:
    """Get the actual stream configuration for the train loader.

    This function sets up the streams dictionary for the train loader by propagating the
    split, remote, and local paths to each stream. It ensures that the paths do not have
    trailing slashes.

    Parameters
    ----------
    root_local : str
        The root local path for the streams.
    root_remote : str
        The root remote path for the streams.
    split : str
        The split (e.g., train, test) for the streams.
    current_client_stream : dict[str, Any]
        The current client stream configuration.

    Returns
    -------
    dict[str, StreamDict]
        A dictionary containing the actual stream configuration for the train loader.

    Example
    -------
    >>> current_client_stream = {
    ...     "stream_0": {"local": "pth1", "remote": "s3://bkt/pth1", "split": "train"},
    ...     "stream_1": {"local": "pth2", "remote": "s3://bkt/pth2", "split": "test"}
    ... }
    >>> actual_streams = get_actual_stream(
    >>> "/local/root", "s3://bucket/root", "train", current_client_stream
    >>> )
    >>> print(actual_streams)
    {'stream_0': StreamDict(
            local='/local/root/path1', remote='s3://bucket/root/path1', split='train'
        ),
     'stream_1': StreamDict(
            local='/local/root/path2', remote='s3://bucket/root/path2', split='train'
        ),
    }

    """
    # Set streams dictionary for the train loader
    actual_streams = {
        key: StreamDict(**value) for key, value in current_client_stream.items()
    }
    # Propagate the split and the remote and local paths to each stream
    for stream in actual_streams.values():
        # Set the split, remote, and local paths
        stream.split = split or stream.split
        if root_local:
            stream.local = root_local + stream.local if stream.local else root_local
        if root_remote:
            stream.remote = (
                root_remote + stream.remote if stream.remote else root_remote
            )
        # Remove potential trailing slashes
        stream.local = stream.local.rstrip("/") if stream.local else stream.local
        stream.remote = stream.remote.rstrip("/") if stream.remote else stream.remote

    return actual_streams


def set_stream(
    cid: int | str | None,
    loader: dict[str, Any] | None,
) -> None:
    """Set the stream configuration for the loader.

    Parameters
    ----------
    cid : int | str | None
        The client ID.
    dataset_config : DictConfig
        The dataset configuration object containing the streams.
    loader : dict[str, Any] | None
        The loader configuration object.

    """
    assert loader is not None, "Loader not None"
    assert loader["dataset"] is not None, "Dataset for loader is not set."

    dataset_config = loader["dataset"]

    # Get the root path for remote and local data
    root_remote, root_local, split = preprocess_stream_paths(dataset_config)
    # Get the clients streams available
    clients_streams = dataset_config["streams"]
    # Extract the current client train stream -- it contains a dict of buckets
    # NOTE: Here, we circumvent the possible limited size of the number of client
    # streams since it should have been handled elsewhere
    current_client_stream: dict[str, Any] = {}
    if cid is not None:
        current_client_stream |= clients_streams[int(cid) % len(clients_streams)][
            "client_streams"
        ]
    else:
        # Concatenate all the streams
        current_client_stream |= concatenate_streams(clients_streams)

    actual_streams = get_actual_stream(
        root_local,
        root_remote,
        split,
        current_client_stream,
    )

    # Convert the streams to dictionaries
    streams_dict = {name: asdict(stream) for name, stream in actual_streams.items()}
    # Assign the streams to the appropriate loaders
    # NOTE: break into function, put assert over variable
    loader["dataset"]["streams"] = streams_dict


def get_split_streams(
    loader: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Get the split streams for the loaders.

    Used to obtain multiple evaluation loaders,
    one for each client.

    Parameters
    ----------
    loader : dict[str, Any] | None
        The loader configuration object.

    Returns
    -------
    list[dict[str, Any]]
        A list of dictionaries containing the split streams loaders.

    """
    assert loader is not None, "Loader not None"
    assert loader["dataset"] is not None, "Dataset for loader is not set."

    dataset_config = loader["dataset"]

    clients_streams = dataset_config["streams"]
    loaders: list[dict[str, Any]] = []
    root_remote, root_local, split = preprocess_stream_paths(dataset_config)
    for inner_cid in range(len(clients_streams)):
        current_client_stream: dict[str, Any] = {}
        current_client_stream |= clients_streams[int(inner_cid) % len(clients_streams)][
            "client_streams"
        ]

        actual_streams = get_actual_stream(
            root_local,
            root_remote,
            split,
            current_client_stream,
        )

        streams_dict = {name: asdict(stream) for name, stream in actual_streams.items()}

        client_eval_loader = copy.deepcopy(loader)
        client_eval_loader["dataset"]["streams"] = streams_dict
        client_eval_loader["label"] = f"client_{inner_cid}"
        loaders.append(client_eval_loader)
    return loaders


def client_set_data_config(
    cid: int | str | None,
    train_cfg: TrainConfig,
    *,
    split_eval: bool = False,
) -> None:
    """Set the data configuration for the client.

    This function configures the dataset for the training and evaluation data loaders
    based on the client ID and the provided training configuration. It processes the
    stream paths and assigns the appropriate streams to the data loaders.

    Parameters
    ----------
    cid : int | str | None
        The client ID.
    train_cfg : TrainConfig
        The training configuration object containing data loader configurations.
    split_eval : bool, optional
        If True, sets the evaluation split to be the same as the training split
        (default is False).

    Example
    -------
    >>> from omegaconf import OmegaConf
    >>> train_cfg = OmegaConf.create({
    ...     "train_loader": {"dataset": {"streams": [...]}, "num_workers": "auto"},
    ...     "eval_loader": {"dataset": {"streams": [...]}, "num_workers": "auto"}
    ... })
    >>> client_set_data_config(cid=1, train_cfg=train_cfg, split_eval=False)

    """
    # Set the train streams
    set_stream(cid, train_cfg.train_loader)

    if train_cfg.eval_loader is not None:
        if not split_eval:
            # Set the eval streams
            set_stream(cid, train_cfg.eval_loader)
        else:
            train_cfg.eval_loader = None
            train_cfg.eval_loaders = get_split_streams(train_cfg.train_loader)


def set_dataset_default_params(train_cfg: TrainConfig) -> None:  # noqa: C901
    """Set default parameters for the dataset configuration in the configuration.

    This function ensures that certain default parameters are set for the dataset
    configuration in the training and evaluation data loaders. It sets default values
    for `predownload`, `num_canonical_nodes`, and `shuffle_block_size` if they are not
    already specified in the configuration.

    Parameters
    ----------
    train_cfg : TrainConfig
        The training configuration object containing data loader configurations.

    Example
    -------
    >>> from omegaconf import OmegaConf
    >>> train_cfg = OmegaConf.create({
    ...     "train_loader": {"dataset": {}, "device_train_batch_size": 32},
    ...     "eval_loader": {"dataset": {}, "device_eval_batch_size": 32},
    ...     "eval_loaders": []
    ... })
    >>> set_dataset_default_params(train_cfg)
    >>> print(train_cfg)

    """
    assert train_cfg.train_loader["dataset"] is not None, (
        "Dataset for train loader is not set."
    )
    # Set the `pre-download` value as 8*batch_size
    if train_cfg.train_loader["dataset"].get("predownload", None) is None:
        train_cfg.train_loader["dataset"]["predownload"] = (
            8 * train_cfg.device_train_batch_size
        )
    if (
        train_cfg.eval_loader is not None
        and train_cfg.eval_loader["dataset"].get("pre_download", None) is None
    ):
        train_cfg.eval_loader["dataset"]["predownload"] = (
            8 * train_cfg.device_eval_batch_size
        )
    elif train_cfg.eval_loaders:
        for loader in train_cfg.eval_loaders:
            loader["dataset"]["predownload"] = 8 * train_cfg.device_eval_batch_size
    # NOTE: Set the `num_canonical_nodes` value as 64*`num_physical_nodes`, assuming
    # that we will always have just 1 real node (server)
    if train_cfg.train_loader["dataset"].get("num_canonical_nodes", None) is None:
        train_cfg.train_loader["dataset"]["num_canonical_nodes"] = 64 * 1
    if (
        train_cfg.eval_loader is not None
        and train_cfg.eval_loader["dataset"].get("num_canonical_nodes", None) is None
    ):
        train_cfg.eval_loader["dataset"]["num_canonical_nodes"] = 64 * 1
    elif train_cfg.eval_loaders:
        for loader in train_cfg.eval_loaders:
            loader["dataset"]["num_canonical_nodes"] = 64 * 1
    # Set the `shuffle_block_size` value as 8*batch_size
    if train_cfg.train_loader["dataset"].get("shuffle_block_size", None) is None:
        train_cfg.train_loader["dataset"]["shuffle_block_size"] = max(
            4_000_000 // train_cfg.train_loader["dataset"]["num_canonical_nodes"],
            1 << 18,
        )
    if (
        train_cfg.eval_loader is not None
        and train_cfg.eval_loader["dataset"].get("shuffle_block_size", None) is None
    ):
        train_cfg.eval_loader["dataset"]["shuffle_block_size"] = max(
            4_000_000 // train_cfg.eval_loader["dataset"]["num_canonical_nodes"],
            1 << 18,
        )
    elif train_cfg.eval_loaders:
        for loader in train_cfg.eval_loaders:
            loader["dataset"]["shuffle_block_size"] = max(
                4_000_000 // loader["dataset"]["num_canonical_nodes"],
                1 << 18,
            )


def set_client_save_and_load_path(cfg: DictConfig, cid: int | str) -> None:
    """Set the save folder path specifically for this client and this run.

    This function updates the `save_folder` in the configuration to include the client
    ID, ensuring that each client has a unique save path for its checkpoints and other
    saved data.

    Parameters
    ----------
    cfg : DictConfig
        The configuration object containing the save folder path.
    cid : int | str
        The client ID.

    Example
    -------
    >>> from omegaconf import OmegaConf
    >>> cfg = OmegaConf.create({"save_folder": "/path/to/save"})
    >>> set_client_save_and_load_path(cfg, cid=1)
    >>> print(cfg.save_folder)
    /path/to/save/client_1

    """
    # Set the save folder specifically for this client and this run
    if cfg.save_folder is not None:  # type: ignore[union-attr]
        cfg.save_folder = (  # type: ignore[union-attr]
            cfg.save_folder
            + "/client_"  # type: ignore[union-attr]
            + str(cid)  # type: ignore[union-attr]
        )
        log(DEBUG, "Set save folder: %s", cfg.save_folder)


def set_client_load_path(
    cfg: DictConfig,
    cid: int | str,
    n_steps: int,
) -> tuple[bool, bool]:
    """Set the load path for the client based on the number of steps and checkpoints.

    This function updates the `load_path` in the configuration to point to the right
    checkpoint for the client. It checks for existing checkpoints and sets the load path
    accordingly. If a checkpoint with the matching number of steps is found, it skips
    the current iteration.

    Parameters
    ----------
    cfg : DictConfig
        The configuration object containing the save and load paths.
    cid : int | str
        The client ID.
    n_steps : int
        The number of steps to match with the checkpoint.

    Returns
    -------
    tuple[bool, bool]
        A tuple containing two boolean values:
        - The first boolean indicates whether to skip the current iteration.
        - The second boolean indicates whether the load path was successfully set.

    Example
    -------
    >>> from omegaconf import OmegaConf
    >>> cfg = OmegaConf.create({"save_folder": "/path/to/save", "load_path": None})
    >>> skip_iteration, load_path_set = set_client_load_path(cfg, cid=1, n_steps=100)
    >>> print(skip_iteration, load_path_set)
    False True

    """
    # Set client load path
    set_client_save_and_load_path(cfg, cid)
    # Flag to notify whether to skip this iteration or not
    skip_iteration = False
    # Set the save folder specifically for this client and this run
    if cfg.save_folder is not None:  # type: ignore[union-attr]
        try:
            # Are there any checkpoints?
            _is_remote, remote_objects = list_objects(cfg.save_folder)
            if not remote_objects:
                log(
                    INFO,
                    "No checkpoints found in %s. Starting training from scratch.",
                    cfg.save_folder,
                )
                assert cfg.load_path is None
                return skip_iteration, False
            # NOTE: We always need to check all of the checkpoints
            # Given the epoch change
            # As such we extract the epoch number and number of batches
            # The number of epochs
            sorted_pairs = sorted(
                [
                    (
                        int(reg.group(1)),  # epoch number
                        int(reg.group(2)),  # number of batches
                    )
                    for path in remote_objects
                    if (
                        reg := re.search(
                            r"client_" + str(cid) + r"/ep(\d+)-ba(\d+)",
                            path,
                        )
                    )
                    is not None
                ],
                key=operator.itemgetter(1),
            )

            log(
                INFO,
                "Found the following sorted checkpoint epochs and batches: %s",
                sorted_pairs,
            )

            # Is there the next checkpoint?
            log(INFO, "Looking for the next checkpoint in %s", cfg.save_folder)
            # See if we have a checkpoint with a matching number of steps
            path_to_check = next(
                (
                    (epoch, batches)
                    for epoch, batches in sorted_pairs
                    if batches == n_steps
                ),
                None,
            )
            # NOTE: ruff is not bright and cannot see through the condition
            skip_iteration = path_to_check is not None
            if skip_iteration and path_to_check is not None:
                epoch, batches = path_to_check
                cfg.load_path = (
                    cfg.save_folder + f"/ep{epoch}-ba{batches}-" + "rank{rank}.pt"
                )
                log(
                    INFO,
                    "Skipping training iteration as checkpoint %s already exists.",
                    cfg.load_path,
                )
                # NOTE: Don't re-save the checkpoint when resuming mid-round
                cfg.save_folder = None
                return skip_iteration, True
            # Load the latest checkpoint
            log(
                INFO,
                "Looking for the latest checkpoint to load in %s",
                cfg.save_folder,
            )
            epoch, batches = sorted_pairs[-1]
            if batches < n_steps:
                cfg.load_path = (
                    cfg.save_folder + f"/ep{epoch}-ba{batches}-" + "rank{rank}.pt"
                )
                log(INFO, "Set checkpoint to load: %s", cfg.load_path)
        except Exception as e:  # noqa: BLE001
            log(WARNING, "The `load_path` wasn't set.", exc_info=e, stack_info=True)
    return skip_iteration, True


def set_client_wandb_logger(train_cfg: TrainConfig, log_name: str) -> None:
    """Set the Weights & Biases (wandb) logger configuration for a specific client.

    This function updates the wandb logger configuration in the training configuration
    to include the client-specific run name and ID. It also sets the configuration file
    path for wandb logging.

    Parameters
    ----------
    train_cfg : TrainConfig
        The training configuration object containing logger configurations.
    log_name : str
        The log name to append to the wandb run name and ID.

    Example
    -------
    >>> from omegaconf import OmegaConf
    >>> train_cfg = OmegaConf.create({
    ...     "loggers": {
    ...         "wandb": {
    ...             "init_kwargs": {"name": "server_run", "id": "server_id"}
    ...         }
    ...     }
    ... })
    >>> set_client_wandb_logger(train_cfg, log_name="_client_1")
    >>> print(train_cfg.loggers["wandb"]["init_kwargs"]["name"])
    server_run_client_1
    >>> print(train_cfg.loggers["wandb"]["init_kwargs"]["id"])
    server_id_client_1
    >>> print(train_cfg.loggers["wandb"]["config_file"])
    /path/to/save/config.yaml

    Raises
    ------
    ValueError
        If the environmental variable `PHOTON_SAVE_PATH` is not set.

    """
    # Set the wandb run name
    if train_cfg.loggers is not None and "wandb" in train_cfg.loggers:
        # Get the server run name
        run_name = train_cfg.loggers["wandb"]["init_kwargs"]["name"]
        # Add the client id to the run name
        new_run_name = run_name + f"{log_name}"
        server_id = train_cfg.loggers["wandb"]["init_kwargs"]["id"]
        train_cfg.loggers["wandb"]["init_kwargs"]["id"] = server_id + f"{log_name}"
        # Set the new run name
        train_cfg.loggers["wandb"]["init_kwargs"]["name"] = new_run_name

        # NOTE: This part won't catch any client-level modification to the config and
        # use directly the one taken form the whole run
        # Get the environmental variable for the dump folder
        save_path = os.environ.get("PHOTON_SAVE_PATH", "")
        # Raise an error if the environmental variable is not set
        if not save_path:
            msg = "The environmental variable PHOTON_SAVE_PATH is not set."
            raise ValueError(msg)
        # Add configuration to the wandb config parameter
        train_cfg.loggers["wandb"]["config_file"] = save_path + "/config.yaml"


def set_client_tensorboard_logger(train_cfg: TrainConfig, log_name: str) -> None:
    """Set the TensorBoard logger configuration for a specific client.

    This function updates the TensorBoard logger configuration in the training
    configuration to include the client-specific log name.

    Parameters
    ----------
    train_cfg : TrainConfig
        The training configuration object containing logger configurations.
    log_name : str
        The log name to append to the TensorBoard log name.

    Example
    -------
    >>> from omegaconf import OmegaConf
    >>> train_cfg = OmegaConf.create({
    ...     "loggers": {
    ...         "tensorboard": {
    ...             "log_name": "server_run"
    ...         }
    ...     }
    ... })
    >>> set_client_tensorboard_logger(train_cfg, log_name="_client_1")
    >>> print(train_cfg.loggers["tensorboard"]["log_name"])
    server_run_client_1

    """
    # Set the tensorboard run name
    if train_cfg.loggers is not None and "tensorboard" in train_cfg.loggers:
        assert train_cfg.loggers["tensorboard"] is not None, (
            "Tensorboard logger is not set."
        )
        # Add the client id to the parameters
        train_cfg.loggers["tensorboard"]["log_name"] = log_name


def adapt_train_batch_size_to_num_devices(cfg: DictConfig) -> None:
    """Adapt the training batch size to the number of visible CUDA devices.

    This function adjusts the global training batch size in the configuration
    to be appropriate for the number of visible CUDA devices. If the estimated
    batch size based on the number of devices differs from the original batch size,
    the configuration is updated, and a warning is logged.

    Parameters
    ----------
    cfg : DictConfig
        The configuration object containing the global training batch size.

    Example
    -------
    >>> from omegaconf import OmegaConf
    >>> cfg = OmegaConf.create({"global_train_batch_size": 64})
    >>> adapt_train_batch_size_to_num_devices(cfg)

    """
    visible_devices = ast.literal_eval(str(os.getenv("APPOINTED_CUDA_DEVICE", "null")))
    if type(visible_devices) is tuple:
        assert len(visible_devices) > 1
        original_batch_size = cfg.global_train_batch_size
        ratio = cfg.global_train_batch_size // len(visible_devices)
        estimated_batch_size = int(ratio * len(visible_devices))
        if estimated_batch_size != cfg.global_train_batch_size:
            cfg.global_train_batch_size = estimated_batch_size
            log(
                WARNING,
                "Train batch size (%s) was not appropriate for %s GPUs available. "
                "New train batch size: %s",
                original_batch_size,
                len(visible_devices),
                cfg.global_train_batch_size,
            )


def set_n_workers_dataloaders(
    train_cfg: TrainConfig,
    device: Device | DeviceGPU | DeviceCPU | None,
    cap: int = 32,
) -> None:
    """Set the number of workers for data loaders based on the available device type.

    This function configures the number of workers for the training and evaluation data
    based on the number of available CPU cores and the type of device (CPU or GPU). It
    ensures that the number of workers does not exceed a specified cap.

    Parameters
    ----------
    train_cfg : TrainConfig
        The training configuration object containing data loader configurations.
    device : Device | DeviceGPU | DeviceCPU | None
        The device type used for training (CPU, GPU, or None).
    cap : int, optional
        The maximum number of workers to set for the data loaders (default is 32).

    Example
    -------
    >>> from omegaconf import OmegaConf
    >>> train_cfg = OmegaConf.create({
    ...     "train_loader": {"num_workers": "auto"},
    ...     "eval_loader": {"num_workers": "auto"}
    ... })
    >>> device = DeviceGPU()
    >>> set_n_workers_dataloaders(train_cfg, device, cap=16)
    >>> print(train_cfg.train_loader["num_workers"])
    16
    >>> print(train_cfg.eval_loader["num_workers"])
    16

    Raises
    ------
    TypeError
        If the device type is not supported.

    """
    # Retrieve system information
    n_cpu_cores_available = get_n_cpu_cores()
    assert n_cpu_cores_available is not None, "Number of CPU cores is not available."
    n_workers: int
    if isinstance(device, DeviceCPU):
        # CPU-only environment that cannot be collaborative
        cpu_concurrency = int(os.getenv("CPU_CONCURRENCY", "1"))
        n_workers = n_cpu_cores_available // cpu_concurrency
    elif isinstance(device, DeviceGPU) or device is None:
        # Collaborative or not environment: multiple GPUs are concurrently used for
        # training each having its own dataloader process
        n_cuda_device = get_n_cuda_devices()
        n_workers = n_cpu_cores_available // n_cuda_device
    else:
        msg = f"Device type {type(device)} is not supported."
        raise TypeError(msg)
    if train_cfg.train_loader["num_workers"] == "auto":
        train_cfg.train_loader["num_workers"] = min(n_workers, cap)
    # NOTE: The latest version of `llmfoundry` allows for a list of evaluation
    # dataloaders. Here we assume the legacy setting where the evaluation dataloader is
    # one and not a list.
    if (
        train_cfg.eval_loader is not None
        and train_cfg.eval_loader["num_workers"] == "auto"
    ):
        train_cfg.eval_loader["num_workers"] = min(n_workers, cap)


def get_stream_freq_dict_for_client(
    client_streams: dict[str, dict[str, Any]],
    s3_comm_config: S3CommConfig | None,
    run_uuid: str | None,
    cid: int | str | None,
    *,
    allow_failures: bool = False,
) -> dict[int, int]:
    """Retrieve the frequency dictionary for a client from local or remote sources.

    This function retrieves the frequency dict for a client by checking local paths
    first and downloading from remote sources if necessary. It merges the frequency
    dictionaries across multiple streams and caches the result for future use.

    Parameters
    ----------
    client_streams : dict[str, dict[str, Any]]
        A dictionary containing the client stream configurations.
    s3_comm_config : S3CommConfig | None
        The S3 communication configuration for downloading remote files.
    run_uuid : str | None
        The unique identifier for the run.
    cid : int | str | None
        The client ID.
    allow_failures : bool, optional
        If True, allows failures in downloading remote files without raising an
        exception (default is False).

    Returns
    -------
    dict[int, tuple[int, str]]
        A dictionary where the keys are integers representing the frequency and the
        values are tuples containing the frequency count and the corresponding string.

    Example
    -------
    >>> client_streams = {
    ...     "stream1": {"local": "/path/to/local", "split": "train", "remote": "s3://bucket/path"},
    ...     "stream2": {"local": "/path/to/local", "split": "test", "remote": "s3://bucket/path"}
    ... }
    >>> s3_comm_config = S3CommConfig(...)
    >>> run_uuid = "unique_run_id"
    >>> cid = 1
    >>> freq_dict = get_stream_freq_dict_for_client(
    >>>     client_streams, s3_comm_config, run_uuid, cid
    >>> )
    >>> print(freq_dict)

    Raises
    ------
    FileNotFoundError
        If a required file is not found and `allow_failures` is False.

    """
    actual_streams = {key: StreamDict(**value) for key, value in client_streams.items()}
    # Stores the merged frequency dictionary across streams
    stream_freq_dict: dict[int, int] = {}
    tmp_dir = tempfile.gettempdir()

    cached_file_name = os.path.join(  # noqa: PTH118
        tmp_dir,
        str(cid) + FREQ_DICT_CACHE_NAME,
    )

    failed_cnt = 0

    if not os.path.exists(cached_file_name):  # noqa: PTH110
        for stream in actual_streams.values():
            assert stream.local is not None, "Local path is not set."
            assert stream.split is not None, "Split is not set."
            local_file_name = os.path.join(  # noqa: PTH118
                stream.local,
                stream.split,
                FREQ_DICT_NAME,
            )
            if not os.path.exists(local_file_name):  # noqa: PTH110
                assert stream.remote is not None, "Remote path is not set."
                assert s3_comm_config is not None, "S3 communication config is not set."
                assert run_uuid is not None, "Run UUID is not set."
                stream_remote_post_processed = stream.remote.replace("s3://", "")

                root_remote, *rest = stream_remote_post_processed.split("/")
                # NOTE: Add threadpool executor
                remote_up_down = create_remote_up_down(
                    bucket_name=root_remote,
                    prefix="",
                    run_uuid=run_uuid,
                    num_attempts=s3_comm_config.num_attempts,
                    client_config=OmegaConf.to_container(
                        s3_comm_config.backend_kwargs.client_config,
                    ),  # type: ignore[reportArgumentType, arg-type]
                )
                remote_path = os.path.join(  # noqa: PTH118
                    os.path.join(*rest),  # noqa: PTH118
                    stream.split,
                    FREQ_DICT_NAME,
                )
                try:
                    download_file_from_s3(remote_up_down, remote_path, local_file_name)
                except FileNotFoundError as _:
                    if not allow_failures:
                        raise
            try:
                with Path(local_file_name).open(encoding="utf-8") as f:
                    loaded_map: dict = json.load(f).items()

                freq_map: dict[int, int]

                freq_map = {
                    int(ast.literal_eval(k)): (
                        v[0] if isinstance(v, list | tuple) else v
                    )
                    for k, v in loaded_map
                }

                stream_freq_dict = merge_freq_dicts(stream_freq_dict, freq_map)
            except FileNotFoundError as _:
                if not allow_failures:
                    raise
                failed_cnt += 1
        log(
            DEBUG,
            "Loaded stream_freq_dict, len: %s, failures %s",
            len(stream_freq_dict),
            failed_cnt,
        )
        with Path(cached_file_name).open("w", encoding="utf-8") as f:
            json.dump(stream_freq_dict, f, indent=4)
    else:
        with Path(cached_file_name).open(encoding="utf-8") as f:
            stream_freq_dict = {int(k): v for k, v in json.load(f).items()}
        log(
            DEBUG,
            "Loaded stream_freq_dict from cache %s, len: %s",
            cached_file_name,
            len(stream_freq_dict),
        )

    return stream_freq_dict
