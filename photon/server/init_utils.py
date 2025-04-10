"""Utility functions for initialization task on main server loop in flwr next."""

import copy
import operator
import os
import re
from logging import DEBUG, INFO

import numpy as np
from composer.loggers import RemoteUploaderDownloader
from composer.utils.file_helpers import list_remote_objects
from flwr.common import (
    NDArrays,
    Parameters,
    log,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from photon.clients.configs import CentralizedConfig
from photon.clients.llm_client_functions import (
    get_trainer_object,
)
from photon.clients.utils import get_initial_parameters
from photon.conf.base_schema import BaseConfig, StrategyName
from photon.server.s3_utils import (
    download_server_checkpoint,
    interpret_resume_round,
    upload_server_checkpoint,
)
from photon.strategy.fedadam import FedAdam
from photon.strategy.fedavg_eff import FedAvgEfficient
from photon.strategy.fedmom import FedMom
from photon.strategy.fednestorov import FedNesterov
from photon.strategy.fedyogi import FedYogi
from photon.utils import (
    ClientState,
    get_parameters_from_state,
)
from photon.wandb_history import WandbHistory


def get_centralized_run_parameters(dummy_config: BaseConfig) -> Parameters:
    """Retrieve the parameters from a centralized run.

    Args:
    ----
    dummy_config : BaseConfig
        The configuration object containing the settings for the federated learning
        server.
    remote_up_down : RemoteUploaderDownloader
        The object to upload/download files from/to the S3 Object Store.

    Returns:
    -------
    Params
        The parameters from the centralized run.

    Raises:
    ------
    ValueError
        If the desired number of batches is not found in the remote objects.

    """
    dummy_config = copy.deepcopy(dummy_config)
    desired_steps = dummy_config.photon.restore_cent_run_batches
    folder = f"s3://checkpoints/{dummy_config.photon.restore_cent_run_uuid}"
    remote_objects = list_remote_objects(folder)
    log(INFO, f"Restoring from centralized run, found {remote_objects}")
    sorted_pairs = sorted(
        [
            (
                int(reg.group(1)),  # epoch number
                int(reg.group(2)),  # number of batches
            )
            for path in remote_objects
            if (reg := re.search(r"/ep(\d+)-ba(\d+)", path)) is not None
        ],
        key=operator.itemgetter(1),
    )
    path_to_check = next(
        (
            (epoch, batches)
            for epoch, batches in sorted_pairs
            if batches == desired_steps
        ),
        None,
    )
    if path_to_check is None:
        msg = f"Could not find a checkpoint with {desired_steps} batches"
        raise ValueError(msg)
    epoch, batches = path_to_check

    dummy_config_llm = dummy_config.llm_config
    dummy_config_llm.load_path = folder + f"/ep{epoch}-ba{batches}-" + "rank{rank}.pt"
    dummy_config_llm.load_ignore_keys = [
        "*scheduler*",
        "*optim*",
        "*dataset_state*",
    ]
    os.environ["APPOINTED_CUDA_DEVICE"] = str(None)
    dummy_config_llm.save_folder = None
    dummy_config_llm.device_train_microbatch_size = 1
    # Creating ClientConfig object
    client_config = CentralizedConfig(
        allow_unigram_metrics_failures=dummy_config_llm.fl.allow_unigram_metrics_failures,
        resize_vocab=dummy_config_llm.fl.resize_vocab,
        split_eval=dummy_config_llm.centralized.split_eval,
        set_trainer_params_filter_keys=dummy_config_llm.fl.set_trainer_params_filter_keys,
        set_trainer_key_to_filter=dummy_config_llm.fl.set_trainer_key_to_filter,
        use_unigram_metrics=dummy_config_llm.fl.use_unigram_metrics,
        s3_comm_config=dummy_config_llm.s3_comm_config,
    )
    trainer, *_ = get_trainer_object(
        dummy_config_llm,
        client_config=client_config,
        cid=None,
        no_data_loading=True,
    )
    return ndarrays_to_parameters(
        get_parameters_from_state(
            {},
            trainer,
        ),
    )


def initialize_round(
    cfg: BaseConfig,
    remote_up_down: RemoteUploaderDownloader | None,
    parameters: Parameters | None = None,
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
    """Initialize the state for a new round of federated learning.

    This function sets up the initial state for a new round of federated learning,
    including initializing bookkeeping variables, client states, global parameters, and
    optionally saving the initial checkpoint to an S3 Object Store if configured. It
    prepares the server for the federated learning process by setting the starting
    round, time offset, cumulative server steps, and initializing the momentum vector
    for optimization algorithms.

    Args:
    ----
    cfg : BaseConfig
        The configuration object containing settings for federated learning and system
        behavior.
    remote_up_down : RemoteUploaderDownloader | None
        An optional uploader/downloader object for interacting with remote storage,
        required if checkpointing or S3 communication is enabled.
    parameters: Parameters | None
        Optional initial parameters to use, e.g from a centralized run.

    Returns:
    -------
    tuple
        A tuple containing the initialized global parameters, history object, starting
        round number, time offset, cumulative server steps, client state dictionary,
        the initial momentum vector (or None if not applicable), the initial second
        momentum vector (or None if not applicable).

    """
    # Initialize the bookkeeping variables
    start_round: int = 0
    time_offset: float = 0.0
    server_steps_cumulative: int = 0
    history = WandbHistory(use_wandb=cfg.use_wandb)
    # Initialize client_state_dict
    client_state: dict[str | int, ClientState] = {
        cid: ClientState(0) for cid in range(cfg.fl.n_total_clients)
    }
    if parameters is None:
        # Initialize parameters only if not provided
        log(INFO, "Initializing global parameters")
        parameters = get_initial_parameters(cfg)

    momentum_vector: NDArrays = []
    second_momentum_vector: NDArrays = []
    # NOTE: We should unify the state with the strategy
    # Only focusing on saving space right now
    match cfg.fl.strategy_name.lower():
        case StrategyName.FEDMOM | StrategyName.NESTOROV:
            momentum_vector = [
                np.zeros_like(x) for x in parameters_to_ndarrays(parameters)
            ]
        case StrategyName.FEDADAM | StrategyName.FEDYOGI:
            momentum_vector = [
                np.zeros_like(x) for x in parameters_to_ndarrays(parameters)
            ]
            second_momentum_vector = [
                np.zeros_like(x) for x in parameters_to_ndarrays(parameters)
            ]

    # Save the checkpoint to S3 Object Store (w/ model parameters)
    if cfg.photon.checkpoint or cfg.photon.comm_stack.s3:
        assert remote_up_down is not None, (
            "Cannot checkpoint without a RemoteUploaderDownloader object"
        )
        upload_server_checkpoint(
            parameters=parameters,
            server_state=(history, start_round, time_offset, server_steps_cumulative),
            momenta=(momentum_vector, second_momentum_vector),
            client_state=client_state,
            remote_up_down=remote_up_down,
        )
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


def resume_from_round(
    cfg: BaseConfig,
    remote_up_down: RemoteUploaderDownloader,
    strategy: FedNesterov | FedMom | FedYogi | FedAdam | FedAvgEfficient,
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
    """Resume from a previous round.

    Parameters
    ----------
    cfg : BaseConfig
        The configuration object.
    remote_up_down : RemoteUploaderDownloader
        The object to upload/download files from/to the S3 Object Store.
    strategy : FedNesterov | FedMom | FedYogi | FedAdam | FedAvgEfficient
        The federated learning strategy to resume.

    Returns
    -------
    tuple[
        Parameters,
        WandbHistory,
        int,
        float,
        int,
        dict[str | int, ClientState],
        NDArrays | None,
        NDArrays | None
    ]
        The model parameters, the history, the round number, the time offset, the
        cumulative number of steps, the client state, and the momentum vectors.

    """
    # Obtain server and strategy state keys
    state_keys = ("state.bin", *strategy.state_keys)
    # Interpret the resume round
    cfg.photon.resume_round = interpret_resume_round(
        resume_round=cfg.photon.resume_round,
        run_uuid_path=(f"s3://{cfg.s3_comm_config.bucket_name}/{cfg.run_uuid}/"),
        # NOTE: Check whether we can relax this condition
        raise_error=cfg.photon.resume_round != -1,
        state_keys=state_keys,
    )
    assert cfg.photon.resume_round is not None, (
        "Cannot resume run if `cfg.photon.resume_round` is None"
    )
    # NOTE: Check whether we can relax this condition
    if cfg.photon.resume_round == -1:
        log(INFO, "No checkpoint found for resuming. Starting from scratch.")
        return initialize_round(cfg, remote_up_down)
    log(DEBUG, "Resume round %s", cfg.photon.resume_round)

    log(INFO, "Resuming from checkpoint")
    return download_server_checkpoint(cfg, remote_up_down)
