"""Utility module that provides functions and classes for Composer Trainer instances.

It includes functions for:
- Cleaning up Trainer instances
- Setting model parameters
- Loading checkpoints
- Configuring mutable attributes such as callbacks, loggers, and dataloaders
- Building and configuring Trainer objects from configuration files

Classes:
--------
- TrainerMutableAttributes: Dataclass containing mutable attributes of the Trainer
object.

Functions:
----------
- trainer_clean_up: Clean up the trainer instance, closing it and releasing resources.
- set_parameters_to_state: Load the given parameters into the trainer's model state.
- load_trainer_checkpoint: Load a checkpoint into the trainer from a specified path.
- get_trainer_mutables_from_config: Construct the mutable attributes of the Trainer
    object from the configuration.
- set_mutables_trainer_callbacks_and_loggers: Set the callbacks and loggers to a loaded
    Trainer instance.
- set_mutables_trainer_train_dataloader: Configure the training dataloader for the
    Trainer instance.
- set_mutables_trainer_eval_dataloader: Configure the evaluation dataloader for the
    Trainer instance.
- set_mutables_trainer: Configure the mutable attributes of the Trainer instance.
- get_trainer_object: Create and configure a Composer Trainer object based on the
    provided configuration.
"""

import ast
import copy
import gc
import logging
import os
import time
import warnings
import weakref
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from logging import DEBUG, ERROR, INFO
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO, cast

import torch

# NOTE: We need this if we want to compile the model because the attention
# implementation in the MPT code is dispatched using a dictionary that raises:
# `AssertionError: Dict types must use ConstDictVariable.`
import torch._dynamo
import wandb
from composer import Callback, DataSpec, Engine, Evaluator, Event, Timestamp, Trainer
from composer.callbacks import (
    CheckpointSaver,
    MemorySnapshot,
    OOMObserver,
)
from composer.core import ensure_data_spec, ensure_evaluator
from composer.devices import DeviceCPU, DeviceGPU
from composer.distributed import DDPSyncStrategy
from composer.loggers import (
    ConsoleLogger,
    Logger,
    LoggerDestination,
    MosaicMLLogger,
    ProgressBarLogger,
    RemoteUploaderDownloader,
    WandBLogger,
)
from composer.loggers.mosaicml_logger import (
    MOSAICML_ACCESS_TOKEN_ENV_VAR,
    MOSAICML_PLATFORM_ENV_VAR,
)
from composer.profiler import JSONTraceHandler, Profiler, TraceHandler, cyclic_schedule
from composer.trainer.trainer import (
    _filter_metrics,  # noqa: PLC2701
    _get_initial_device_train_microbatch_size,  # noqa: PLC2701
    _raise_missing_argument_exception,  # noqa: PLC2701
    _set_evaluator_interval_and_subset_num_batches,  # noqa: PLC2701
    _validate_evaluator,  # noqa: PLC2701
)
from composer.utils import (
    TPConfig,
    checkpoint,
    dist,
    ensure_tuple,
    get_device,
    maybe_create_object_store_from_uri,
    maybe_create_remote_uploader_downloader_from_uri,
    parse_uri,
    reproducibility,
)
from flwr.common.logger import log
from llmfoundry.callbacks import AsyncEval, EvalGauntlet
from llmfoundry.command_utils.train import (
    _log_num_params,  # noqa: PLC2701
    _sort_callbacks,  # noqa: PLC2701
    validate_config,
)
from llmfoundry.data.dataloader import build_dataloader
from llmfoundry.eval.metrics import InContextLearningMetric
from llmfoundry.registry import (
    metrics,
)
from llmfoundry.utils.builders import (
    add_metrics_to_eval_loaders,
    build_algorithm,
    build_callback,
    build_composer_model,
    build_evaluators,
    build_load_planner,
    build_logger,
    build_metric,
    build_optimizer,
    build_save_planner,
    build_scheduler,
    build_tokenizer,
    build_tp_strategies,
)
from llmfoundry.utils.config_utils import (
    TRAIN_CONFIG_KEYS,
    TrainConfig,
    make_dataclass_and_log_config,
    pop_config,
    process_init_device,
)
from llmfoundry.utils.exceptions import (
    BaseContextualError,
    EvalDataLoaderLocation,
    TrainDataLoaderLocation,
)
from llmfoundry.utils.mosaicml_logger_utils import (
    find_mosaicml_logger,
    log_train_analytics,
    maybe_create_mosaicml_logger,
)
from llmfoundry.utils.registry_utils import import_file
from omegaconf import DictConfig

from photon.clients.configs import CentralizedConfig, EvaluateConfig, FitConfig
from photon.clients.llm_config_functions import (
    adapt_train_batch_size_to_num_devices,
    client_set_data_config,
    get_stream_freq_dict_for_client,
    set_client_tensorboard_logger,
    set_client_wandb_logger,
    set_dataset_default_params,
    set_icl_tasks_root_dir,
    set_n_workers_dataloaders,
)
from photon.metrics.unigram_normalized_metrics import (
    UNIGRAM_METRIC_NAMES_AND_CLASSES,
    create_wrapped_subclass,
)
from photon.utils import (
    freeze_blocks,
    get_list_of_parameters_names,
    get_unigram_probabilities_tensor,
)

if TYPE_CHECKING:
    from photon.conf.base_schema import S3CommConfig

# Track which callbacks are already open, so it is possible to error and instruct the
# user to call previous_trainer.close() if necessary before attempting to reuse a
# callback
_OPEN_CALLBACKS: weakref.WeakSet = weakref.WeakSet()


@dataclass
class TrainerMutableAttributes:
    """Dataclass containing mutable attributes of the Trainer object.

    Attributes
    ----------
    train_loader : DataSpec or None
        The training data specification.
    evaluators : list[Evaluator] or None
        The evaluator objects for validation/testing.
    callbacks : Callback or Sequence[Callback] or None
        A list or single callback to use during training.
    loggers : LoggerDestination or Sequence[LoggerDestination] or None
        Logger destinations for training output.
    train_cfg : TrainConfig
        Configuration object for training.
    save_latest_filename : str or None
        Name for the latest checkpoint file.
    save_filename : str
        Template for naming checkpoints.

    """

    train_loader: DataSpec | None
    evaluators: list[Evaluator] | None
    callbacks: Callback | Sequence[Callback] | None
    loggers: LoggerDestination | Sequence[LoggerDestination] | None
    train_cfg: TrainConfig

    save_latest_filename: str | None = None
    save_filename: str = "ep{epoch}-ba{batch}-rank{rank}.pt"


def trainer_clean_up(
    trainer: Trainer,
) -> None:
    """Clean up the trainer instance, closing it and releasing resources.

    Parameters
    ----------
    trainer : Trainer
        The Composer Trainer instance to close and delete.

    """
    # Close the trainer, wait for all the collaborators first
    dist.barrier()
    trainer.close()
    # Delete the trainer
    try:
        del trainer
    except Exception as e:  # noqa: BLE001
        log(ERROR, "Error deleting trainer", exc_info=e, stack_info=True)
    # Clean-up garbage collector and cuda cache
    gc.collect()
    torch.cuda.empty_cache()


def load_trainer_checkpoint(
    trainer: Trainer,
    train_cfg: TrainConfig,
) -> None:
    """Load a checkpoint into the trainer from a specified path.

    Parameters
    ----------
    trainer : Trainer
        The Composer Trainer to which the checkpoint is loaded.
    train_cfg : TrainConfig
        Configuration object containing the checkpoint load path.

    """
    trainer.engine.run_event(Event.BEFORE_LOAD)
    assert train_cfg.load_path is not None, (
        "Load path cannot be None if checkpoint exists."
    )
    load_object_store = maybe_create_object_store_from_uri(train_cfg.load_path)
    if isinstance(load_object_store, WandBLogger) and wandb.run is None:
        load_object_store.init(trainer.state, trainer.logger)
    _, _, parsed_load_path = parse_uri(train_cfg.load_path)
    trainer._rng_state = checkpoint.load_checkpoint(  # noqa: SLF001
        state=trainer.state,
        logger=trainer.logger,
        path=parsed_load_path,
        object_store=load_object_store,
        load_weights_only=False,
        strict_model_weights=False,
        progress_bar=True,
        ignore_keys=None,
        exclude_algorithms=None,
        algorithm_passes=trainer.engine.algorithm_passes,
    )
    default_run_name: str = os.environ.get("RUN_NAME", "llm")
    trainer.state.run_name = train_cfg.run_name or default_run_name
    trainer.state.load_path = train_cfg.load_path
    if (
        trainer.state.timestamp.iteration == 0
        and trainer.state.timestamp.token_in_iteration == 0
        and trainer.state.timestamp.epoch_in_iteration == 0
    ):
        trainer.state.timestamp = trainer.state.timestamp.copy(
            epoch_in_iteration=trainer.state.timestamp.epoch,
            token_in_iteration=trainer.state.timestamp.token,
        )
    trainer.engine.run_event(Event.AFTER_LOAD)


def add_unigram_metrics(
    client_config: FitConfig | EvaluateConfig | CentralizedConfig,
    train_cfg: TrainConfig,
    streams: dict[str, dict[str, Any]],
) -> None:
    """Add unigram metrics to the metrics registry.

    Parameters
    ----------
    client_config : FitConfig or EvaluateConfig
        Configuration object for federated client training or evaluation
    train_cfg : TrainConfig
        Configuration object for training.
    streams : dict[str, dict[str, Any]]
        Dictionary containing the train streams.

    """
    s3_comm_config = cast("S3CommConfig", client_config.s3_comm_config)
    # Optional parameters will be set to default values if not specified.
    default_run_name: str = os.environ.get("RUN_NAME", "llm")
    run_name: str = train_cfg.run_name or default_run_name
    cid = (
        client_config.cid if not isinstance(client_config, CentralizedConfig) else None
    )
    train_stream_freq_dict = get_stream_freq_dict_for_client(
        streams,
        s3_comm_config,
        run_name,
        cid,
        allow_failures=client_config.allow_unigram_metrics_failures,
    )

    unigram_probabilities = get_unigram_probabilities_tensor(
        train_stream_freq_dict,
    )

    log(
        DEBUG,
        f"""Unigram probabilities for client {cid}:
        {unigram_probabilities}""",
    )

    for name, metric in UNIGRAM_METRIC_NAMES_AND_CLASSES.items():
        metrics.register(
            name,
            func=create_wrapped_subclass(
                base_class=metric,
                unigram_probabilities=unigram_probabilities,
            ),
        )


def get_trainer_mutables_from_config(  # noqa: PLR0914, C901, PLR0913, PLR0917, PLR0915, PLR0912
    trainer: Trainer,
    train_cfg: TrainConfig,
    client_config: FitConfig | EvaluateConfig,
    icl_tasks_config_dict: dict[str, Any] | None,
    device: DeviceGPU | DeviceCPU | None,
    logged_cfg: dict[str, Any],
    *,
    no_data_loading: bool = False,
) -> TrainerMutableAttributes:
    """Construct the mutable attributes of the Trainer object from the configuration.

    The function re-executes the initialization steps of the Trainer object, such as
    building the train and evaluator dataloaders, callbacks, and loggers, based on the
    configuration objects provided.

    Parameters
    ----------
    trainer : Trainer
        Composer Trainer object.
    train_cfg : TrainConfig
        Configuration object for training.
    client_config : FitConfig or EvaluateConfig
        Configuration object for federated client training or evaluation.
    icl_tasks_config_dict : dict[str, Any] or None
        ICL tasks configuration dictionary.
    device : DeviceGPU or DeviceCPU or None
        Reference to the training device.
    logged_cfg : dict[str, Any]
        Dictionary to store logs or metadata.
    no_data_loading : bool, optional
        Whether to skip data loading steps, by default False.

    Returns
    -------
    TrainerMutableAttributes
        An object containing trainer mutables such as callbacks, loggers,
        and loaders.

    Raises
    ------
    BaseContextualError
        If an error occurs while building train or evaluator dataloaders.
    ValueError
        If both frozen and unfrozen layers are specified in the configuration.

    """
    # Interpret and check consistency the configuration for (un)frozen layers
    if isinstance(client_config, FitConfig):
        assert not (
            client_config.frozen_layers is not None
            and client_config.unfrozen_layers is not None
        ), "Cannot specify both frozen and unfrozen layers"
    # Set logging level
    if train_cfg.python_log_level is not None:
        logging.basicConfig(
            # Example of format string
            # 2022-06-29 11:22:26,152: rank0[822018][MainThread]: INFO: Message here
            format=f"%(asctime)s: rank{dist.get_global_rank()}[%(process)d]"
            f"[%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
            force=True,
        )
        logging.getLogger("llmfoundry").setLevel(
            train_cfg.python_log_level.upper(),
        )  # Foundry module
        logging.getLogger(__name__).setLevel(
            train_cfg.python_log_level.upper(),
        )  # Train script
        logging.getLogger("streaming").setLevel(
            train_cfg.python_log_level.upper(),
        )  # Streaming module

    log(DEBUG, "Initializing dist with device...")
    dist.initialize_dist(get_device(device), timeout=train_cfg.dist_timeout)
    log(DEBUG, "Testing barrier with device...")
    dist.barrier()
    log(DEBUG, "Barrier test passed with device.")

    # Set seed first
    seed: int = train_cfg.seed
    reproducibility.seed_all(seed)

    # Mandatory model training configs
    model_config = train_cfg.model
    train_loader_config = train_cfg.train_loader

    # Optional fsdp data, fine-tuning, and eval configs
    fsdp_config: dict[str, Any] | None = train_cfg.fsdp_config

    eval_loader_config = (
        train_cfg.eval_loader
        if train_cfg.eval_loader is not None
        else train_cfg.eval_loaders
    )
    icl_tasks_config = train_cfg.icl_tasks or train_cfg.icl_tasks_str
    eval_gauntlet_config = train_cfg.eval_gauntlet or train_cfg.eval_gauntlet_str

    is_state_dict_sharded: bool = (
        (fsdp_config.get("state_dict_type", "full") == "sharded")
        if fsdp_config
        else False
    )
    save_latest_filename: str = train_cfg.save_latest_filename or (
        "latest-sharded-rank{rank}" if is_state_dict_sharded else "latest-rank{rank}.pt"
    )
    save_filename: str = train_cfg.save_filename or "ep{epoch}-ba{batch}-rank{rank}.pt"

    # Enable autoresume from model checkpoints if possible
    autoresume_default: bool = False
    if (
        train_cfg.save_folder is not None
        and not train_cfg.save_overwrite
        and not train_cfg.save_weights_only
    ):
        autoresume_default = True

    if not train_cfg.autoresume and autoresume_default:
        log(
            INFO,
            "As run_name, save_folder, and save_latest_filename are set, \
                changing autoresume default to True...",
        )

    # NOTE: This needs to be built as it's contained in the dataset object underlying
    # the dataloaders
    # Build tokenizer
    log(INFO, "Building tokenizer...")
    tokenizer_name = train_cfg.tokenizer["name"]
    tokenizer_kwargs = train_cfg.tokenizer.get("kwargs", {})
    tokenizer = build_tokenizer(tokenizer_name, tokenizer_kwargs)

    # Loggers
    loggers = (
        [
            build_logger(str(name), logger_cfg)
            for name, logger_cfg in train_cfg.loggers.items()
        ]
        if train_cfg.loggers
        else []
    )

    mosaicml_logger = find_mosaicml_logger(loggers)
    if mosaicml_logger is None:
        mosaicml_logger = maybe_create_mosaicml_logger()
        if mosaicml_logger is not None:
            # mosaicml_logger will be None if run isn't on MosaicML platform
            loggers.append(mosaicml_logger)

    if train_cfg.metadata is not None:
        # Optionally flatten the metadata for logging
        if train_cfg.flatten_metadata:
            logged_cfg.pop("metadata", None)
            common_keys = set(
                logged_cfg.keys(),
            ) & set(train_cfg.metadata.keys())
            if len(common_keys) > 0:
                msg = (
                    f"Keys {common_keys} are already present in the config."
                    " Please rename them in metadata or set flatten_metadata=False"
                    " to avoid flattening the metadata in the logged config."
                )
                raise ValueError(
                    msg,
                )

            logged_cfg.update(train_cfg.metadata, merge=True)

        if mosaicml_logger is not None:
            mosaicml_logger.log_metrics(train_cfg.metadata)
            mosaicml_logger._flush_metadata(force_flush=True)  # noqa: SLF001

    # Callbacks
    callback_configs = train_cfg.callbacks or {}
    callbacks: list[Callback] = [
        build_callback(
            name=str(name),
            kwargs=callback_cfg,
            train_config=logged_cfg,
        )
        for name, callback_cfg in callback_configs.items()
    ]

    if not train_cfg.callbacks:
        train_cfg.callbacks = {}
    use_async_eval = any(
        "async_eval" in callback_name for callback_name in train_cfg.callbacks
    )

    # Dataloaders
    log(INFO, "Building train loader...")
    try:
        train_loader = None
        train_streams: dict[str, dict[str, Any]] | None = None
        if not no_data_loading:
            assert train_cfg.train_loader["dataset"] is not None, (
                "Dataset for train loader is not set."
            )
            train_streams = train_loader_config["dataset"]["streams"]

            if client_config.use_unigram_metrics:
                assert train_streams is not None, "Train streams must be provided."
                add_unigram_metrics(
                    client_config,
                    train_cfg,
                    train_streams,
                )
            assert isinstance(train_loader_config, dict), (
                "Expected train_loader_config to be a dict,"
                f" got {type(train_loader_config)}"
            )
            train_loader = build_dataloader(
                train_loader_config,
                tokenizer,
                train_cfg.device_train_batch_size,
            )
    except BaseContextualError as e:
        e.location = TrainDataLoaderLocation
        raise

    if mosaicml_logger is not None:
        mosaicml_logger.log_metrics({"data_validated": time.time()})

    # Evaluation
    evaluators: list[Evaluator] = []
    _eval_gauntlet_callback: EvalGauntlet | None = None
    if use_async_eval:
        if train_cfg.eval_first:
            warnings.warn(
                "AsyncEval callback does not support eval_first=True. Ignoring.",
                stacklevel=2,
            )
            train_cfg.eval_first = False
    else:
        try:
            log(INFO, "Building eval loader...")
            # Extracting the destination directory for the eval gauntlet
            destination_dir: str | None = None
            if eval_gauntlet_config is not None and not isinstance(
                eval_gauntlet_config,
                str,
            ):
                destination_dir = eval_gauntlet_config.pop("destination_dir", None)
            if (
                icl_tasks_config_dict is not None
                and icl_tasks_config is not None
                and not isinstance(icl_tasks_config, str)
            ):
                set_icl_tasks_root_dir(
                    icl_tasks_config,
                    icl_tasks_config_dict["root_dir"],
                )
            eval_icl_seq_len: int = train_cfg.icl_seq_len or train_cfg.max_seq_len
            evaluators, _, _eval_gauntlet_callback = build_evaluators(
                eval_loader_config=eval_loader_config,
                icl_tasks_config=icl_tasks_config,
                eval_gauntlet_config=eval_gauntlet_config,
                tokenizer=tokenizer,
                device_eval_batch_size=train_cfg.device_eval_batch_size,
                icl_seq_len=eval_icl_seq_len,
                icl_subset_num_batches=train_cfg.icl_subset_num_batches,
                device_eval_microbatch_size=train_cfg.device_eval_microbatch_size,
                destination_dir=destination_dir,
            )
            # if eval_gauntlet_callback is not None:
            #     callbacks.append(eval_gauntlet_callback)  # noqa: ERA001
        except BaseContextualError as e:
            e.location = EvalDataLoaderLocation
            raise

    if mosaicml_logger is not None:
        log_train_analytics(
            mosaicml_logger,
            model_config,
            train_loader_config,
            eval_loader_config,
            train_cfg.callbacks,
            tokenizer_name,
            train_cfg.load_path,
            icl_tasks_config,
            eval_gauntlet_config,
        )

    # Now add the eval metrics
    try:
        if eval_loader_config is not None and not use_async_eval:
            # NOTE: If DDP wrapped, `trainer.state.model` doesn't possess the
            # `get_metrics` functions, so we must use the `trainer._original_model`
            # instead, as also recommended in the `Trainer.init` function
            if isinstance(
                trainer.state.model,
                torch.nn.parallel.DistributedDataParallel,
            ):
                eval_metrics = trainer._original_model.get_metrics(  # noqa: SLF001
                    is_train=False,
                )
            else:
                eval_metrics = trainer.state.model.get_metrics(is_train=False)
            non_icl_metrics = [
                metric_name
                for metric_name, metric in eval_metrics.items()
                if not isinstance(metric, InContextLearningMetric)
            ]
            evaluators = add_metrics_to_eval_loaders(
                evaluators,
                non_icl_metrics,
            )
    except BaseContextualError as e:
        e.location = EvalDataLoaderLocation
        raise

    if isinstance(client_config, FitConfig):
        client_config.unfrozen_layers = get_list_of_parameters_names(
            trainer.state.model,
        )

    return TrainerMutableAttributes(
        train_loader=train_loader,
        evaluators=evaluators,
        callbacks=callbacks,
        save_latest_filename=save_latest_filename,
        save_filename=save_filename,
        train_cfg=train_cfg,
        loggers=loggers,
    )


def set_mutables_trainer_callbacks_and_loggers(  # noqa: C901, PLR0913, PLR0917, PLR0912, PLR0915
    trainer: Trainer,
    callbacks: Callback | Sequence[Callback] | None,
    loggers: LoggerDestination | Sequence[LoggerDestination] | None,
    train_cfg: TrainConfig,
    profiler: Profiler | None = None,
    console_stream: str | TextIO = "stderr",
    save_latest_filename: str | None = None,
    save_filename: str = "ep{epoch}-ba{batch}-rank{rank}.pt",
    *,
    log_traces: bool = False,
) -> None:
    """Set the callbacks and loggers to a loaded Trainer instance.

    Parameters
    ----------
    trainer : Trainer
        The Composer Trainer instance to configure.
    callbacks : Callback or Sequence[Callback] or None
        Callbacks to use during training.
    loggers : LoggerDestination or Sequence[LoggerDestination] or None
        Logger destinations for training output.
    train_cfg : TrainConfig
        Configuration object for training.
    profiler : Profiler or None, optional
        Profiler to use during training, by default None.
    console_stream : str or TextIO, optional
        Stream for console logging, by default "stderr".
    log_traces : bool, optional
        Whether to log traces, by default False.
    save_latest_filename : str or None, optional
        Filename for the latest checkpoint, by default None.
    save_filename : str, optional
        Template for naming checkpoints, by default "ep{epoch}-ba{batch}-rank{rank}.pt".

    Raises
    ------
    ValueError
        If a `RemoteUploaderDownloader` with a `file_path_format_string` is used with
        `save_latest_filename`.

    """
    # Close callbacks and dataloaders using the Engine
    trainer.engine.close()
    old_engine = trainer.engine
    new_loggers = list(ensure_tuple(loggers))
    trainer.state._callbacks = list(ensure_tuple(callbacks))  # noqa: SLF001

    # Profiler
    if profiler is not None:
        warnings.warn(
            "The profiler is enabled. Using the profiler adds additional overhead"
            " when training.",
            stacklevel=2,
        )
        trainer.state.profiler = profiler
        for remote_uri in profiler.remote_filenames:
            remote_ud = maybe_create_remote_uploader_downloader_from_uri(
                uri=remote_uri,
                loggers=new_loggers,
            )
            if remote_ud is not None:
                new_loggers.append(remote_ud)
        trainer.state.profiler.bind_to_state(trainer.state)

    # MemorySnapshot, OOMObserver
    for cb in trainer.state.callbacks:
        if isinstance(cb, MemorySnapshot | OOMObserver) and cb.remote_file_name:
            remote_ud = maybe_create_remote_uploader_downloader_from_uri(
                uri=cb.remote_file_name,
                loggers=new_loggers,
            )
            if remote_ud is not None:
                trainer.logger.destinations = ensure_tuple([*new_loggers, remote_ud])

    if train_cfg.progress_bar and train_cfg.log_to_console:
        warnings.warn(
            "Setting both `progress_bar` and `log_to_console` both to True is not"
            " recommended and will lead to duplicate logs and weird formatting issues."
            " Please set one of them to False for a better logging experience.",
            stacklevel=2,
        )

    if any(isinstance(x, ProgressBarLogger) for x in new_loggers):
        warnings.warn(
            Warning(
                f"Specifying the {ProgressBarLogger.__name__} via `loggers` is not"
                " recommended as any values set for the following Trainer arguments"
                " will be ignored: `progress_bar`, `console_stream`, or `log_traces`."
                " The recommended way of enabling a progress bar is to set"
                " `progress_bar` to True instead of constructing a"
                f" {ProgressBarLogger.__name__} instance.",
            ),
            stacklevel=2,
        )
    elif train_cfg.progress_bar:
        new_loggers.append(
            ProgressBarLogger(stream=console_stream, log_traces=log_traces),
        )

    # Console Logging
    if any(isinstance(x, ConsoleLogger) for x in new_loggers):
        warnings.warn(
            Warning(
                f"Specifying the {ConsoleLogger.__name__} via `loggers` is not"
                " recommended as any values set for the following Trainer arguments"
                " will be ignored: `log_to_console`, `console_stream`, `log_traces`,"
                " and `console_log_interval`. The recommended way of enabling a console"
                " logging is to set `log_to_console` to True instead of constructing a"
                f" {ConsoleLogger.__name__} instance.",
            ),
            stacklevel=2,
        )
    elif train_cfg.log_to_console:
        new_loggers.append(
            ConsoleLogger(
                stream=console_stream,
                log_interval=train_cfg.console_log_interval,
                log_traces=log_traces,
            ),
        )

    # MosaicML Logger
    # Keep MosaicML logger above the RemoteUploaderDownloader so that fit end is
    # reported before the final checkpoint begins uploading
    if (
        os.environ.get(MOSAICML_PLATFORM_ENV_VAR, "false").lower() == "true"
        and os.environ.get(
            MOSAICML_ACCESS_TOKEN_ENV_VAR,
        )
        is not None
        and not any(isinstance(x, MosaicMLLogger) for x in new_loggers)
    ):
        log(
            INFO,
            "Detected run on MosaicML platform. Adding MosaicMLLogger to loggers.",
        )
        mosaicml_logger = MosaicMLLogger()
        new_loggers.append(mosaicml_logger)

    # Logger
    trainer.logger = Logger(state=trainer.state, destinations=new_loggers)

    if save_latest_filename is not None:
        remote_ud_has_format_string = [
            isinstance(logger_destination, RemoteUploaderDownloader)
            and logger_destination.file_path_format_string != "{remote_file_name}"
            for logger_destination in trainer.logger.destinations
        ]
        if any(remote_ud_has_format_string):
            msg = (
                "Specifying a `file_path_format_string` to a `RemoteUploaderDownloader`"
                " is not currently supported while using `save_latest_filename`. "
                "Please specify the path formatting via `save_folder`, `save_filename`,"
                " and `save_latest_filename`"
            )
            raise ValueError(
                msg,
            )

    trainer.state.callbacks[:] = (
        list(cast("list[Callback]", new_loggers)) + trainer.state.callbacks
    )

    latest_remote_file_name: str | None = None
    dummy_checkpoint_savers = [
        cb for cb in trainer.state.callbacks if isinstance(cb, CheckpointSaver)
    ]
    if len(dummy_checkpoint_savers) >= 1:
        if len(dummy_checkpoint_savers) > 1:
            log(
                INFO,
                "Multiple CheckpointSaver provided as callbacks."
                " Using the first one as reference.",
            )
        trainer._checkpoint_saver = dummy_checkpoint_savers[0]  # noqa: SLF001

        if trainer._checkpoint_saver.folder != train_cfg.save_folder:  # noqa: SLF001
            log(
                INFO,
                f"Using {trainer._checkpoint_saver.folder} as save_folder.",  # noqa: SLF001
            )
            train_cfg.save_folder = trainer._checkpoint_saver.folder  # noqa: SLF001

        if trainer._checkpoint_saver.latest_filename is None:  # noqa: SLF001
            save_latest_filename = None
            log(INFO, f"Using {save_latest_filename} as latest_filename.")
        elif (
            trainer._checkpoint_saver.latest_filename.filename  # noqa: SLF001
            != save_latest_filename
        ):
            save_latest_filename = str(
                trainer._checkpoint_saver.latest_filename.filename,  # noqa: SLF001
            )
            log(INFO, f"Using {save_latest_filename} as latest_filename.")

        if (
            trainer._checkpoint_saver.latest_remote_file_name  # noqa: SLF001
            is not None
        ):
            latest_remote_file_name = str(
                trainer._checkpoint_saver.latest_remote_file_name.filename,  # noqa: SLF001
            )

    # if trainer._checkpoint_saver is None and train_cfg.save_folder is not None:
    if train_cfg.save_folder is not None:
        if train_cfg.save_weights_only:
            log(
                INFO,
                "save_weights_only=True now also saves metadata and integrations!"
                " Please adjust your workflow accordingly.",
            )

        _, _, parsed_save_folder = parse_uri(train_cfg.save_folder)

        # If user passes a URI with s3:// and a bucket_name, but no other
        # path then we assume they just want their checkpoints saved directly in their
        # bucket.
        if not parsed_save_folder:
            remote_file_name = save_filename
            latest_remote_file_name = save_latest_filename

        # If they actually specify a path, then we use that for their local save path
        # and we prefix save_filename with that path for remote_file_name.
        else:
            remote_file_name = str(Path(parsed_save_folder) / Path(save_filename))
            if save_latest_filename is not None:
                latest_remote_file_name = str(
                    Path(parsed_save_folder) / Path(save_latest_filename),
                )
            else:
                latest_remote_file_name = None

        trainer._checkpoint_saver = CheckpointSaver(  # noqa: SLF001
            folder=train_cfg.save_folder,
            filename=save_filename,
            remote_file_name=remote_file_name,
            latest_filename=save_latest_filename,
            latest_remote_file_name=latest_remote_file_name,
            overwrite=train_cfg.save_overwrite,
            weights_only=train_cfg.save_weights_only,
            ignore_keys=train_cfg.save_ignore_keys,
            save_interval=train_cfg.save_interval,
            num_checkpoints_to_keep=train_cfg.save_num_checkpoints_to_keep,
        )
        trainer.state.callbacks.append(trainer._checkpoint_saver)  # noqa: SLF001
    trainer.engine = Engine(state=trainer.state, logger=trainer.logger)
    del old_engine
    gc.collect()
    # Set the logger
    trainer.state.model.logger = trainer.logger  # pyright: ignore[reportArgumentType]
    # Run Event.INIT
    trainer.engine.run_event(Event.INIT)


def set_mutables_trainer_train_dataloader(  # noqa: PLR0913
    trainer: Trainer,
    train_dataloader: DataSpec | None,
    client_config: FitConfig | EvaluateConfig | CentralizedConfig,
    train_dataloader_label: str = "train",
    train_subset_num_batches: int = -1,
    *,
    spin_dataloaders: bool = True,
) -> None:
    """Configure the training dataloader for the Trainer instance.

    Parameters
    ----------
    trainer : Trainer
        The Composer Trainer instance to configure.
    train_dataloader : DataSpec or None
        The training dataloader specification.
    client_config : FitConfig or EvaluateConfig or CentralizedConfig
        Configuration object for federated client training or evaluation.
    train_cfg : TrainConfig
        Configuration object for training.
    train_dataloader_label : str, optional
        The label for the training dataloader, by default "train".
    train_subset_num_batches : int, optional
        Number of batches to use from the training dataloader, by default -1.
    spin_dataloaders : bool, optional
        Whether to spin the dataloaders to the current epoch, by default True.

    """
    # NOTE: What follows has been taken from the incipit of the `fit` method of the
    # Trainer class
    if train_dataloader is not None:
        trainer._train_data_spec = ensure_data_spec(train_dataloader)  # noqa: SLF001
        trainer.state.set_dataloader(
            trainer._train_data_spec.dataloader,  # noqa: SLF001
            train_dataloader_label,
        )

        trainer.state.train_dataloader = trainer.state.dataloader

        if client_config.use_unigram_metrics:
            train_metrics = trainer._original_model.get_metrics(  # noqa: SLF001
                is_train=True,
            )

            non_unigram_metric_names = [
                metric_name
                for k in train_metrics
                if "unigram" not in (metric_name := str(k)).lower()
            ]

            trainer.state.train_metrics = _filter_metrics(
                train_metrics,
                non_unigram_metric_names,
            )

            trainer.state.train_metrics |= {
                metric_name: build_metric(metric_name)
                for metric_name in UNIGRAM_METRIC_NAMES_AND_CLASSES
            }

        trainer.state.device_train_microbatch_size = (
            _get_initial_device_train_microbatch_size(
                trainer.state.device_train_microbatch_size,
                trainer.state.auto_microbatching,
                trainer.state.train_dataloader,
            )
        )
    if trainer._train_data_spec is None:  # noqa: SLF001
        _raise_missing_argument_exception("train_dataloader")
    trainer.state.dataloader_len = train_subset_num_batches
    trainer.spin_dataloaders = spin_dataloaders


def set_mutables_trainer_eval_dataloader(
    trainer: Trainer,
    eval_dataloader: Iterable | DataSpec | Evaluator | Sequence[Evaluator] | None,
    train_cfg: TrainConfig,
) -> None:
    """Configure the evaluation dataloader for the Trainer instance.

    Parameters
    ----------
    trainer : Trainer
        The Composer Trainer instance to configure.
    eval_dataloader : Iterable, DataSpec, Evaluator, or Sequence[Evaluator] or None
        The evaluation dataloader specification.
    train_cfg : TrainConfig
        Configuration object for training.

    Raises
    ------
    ValueError
        If mixing Evaluator with other classes.

    """
    # NOTE: What follows has been taken from the incipit of the `fit` method of the
    # Trainer class
    if eval_dataloader is not None:
        # Need to use the `original_model` rather than `state.model`, as `state.model`
        # could be DDP wrapped.
        eval_metrics = trainer._original_model.get_metrics(  # noqa: SLF001
            is_train=False,
        )
        metric_names = [str(k) for k in eval_metrics]
        eval_dataloader = ensure_tuple(eval_dataloader)

        evaluator_types = [
            isinstance(evaluator, Evaluator) for evaluator in eval_dataloader
        ]
        if any(evaluator_types) and not all(evaluator_types):
            raise ValueError(
                "Mixing Evaluator with other classes is not allowed, please wrap"
                "all other classes with the Evaluator class. These are the classes"
                "that were detected:"
                + str([type(evaluator) for evaluator in eval_dataloader]),
            )

        evaluators = [
            ensure_evaluator(evaluator, default_metric_names=metric_names)
            for evaluator in eval_dataloader
        ]

        # match metric names to model metrics
        trainer.state.eval_metrics = {
            evaluator.label: _filter_metrics(eval_metrics, evaluator.metric_names)
            for evaluator in evaluators
        }

        _set_evaluator_interval_and_subset_num_batches(
            evaluators=evaluators,
            eval_interval=train_cfg.eval_interval,
            subset_num_batches=train_cfg.eval_subset_num_batches,
        )

        for evaluator in evaluators:
            _validate_evaluator(evaluator, trainer.state.device)

        if len(evaluators) == 0:
            if train_cfg.eval_subset_num_batches != -1:
                warnings.warn(
                    "Specifying `eval_subset_num_batches="
                    f"{train_cfg.eval_subset_num_batches}`"
                    " without an `eval_dataloader` has no effect. If trying to run an"
                    " evaluator, make sure `eval_dataloader` is specified. Otherwise,"
                    " set `eval_subset_num_batches` to default value -1.",
                    stacklevel=2,
                )
            if train_cfg.eval_interval not in {0, 1}:
                warnings.warn(
                    f"Specifying `eval_interval={train_cfg.eval_interval}` without an"
                    " `eval_dataloader` has no effect. If trying to run an evaluator,"
                    " make sure `eval_dataloader` is specified. Otherwise, set"
                    " `eval_interval` to 0 or default value 1.",
                    stacklevel=2,
                )

        trainer.state.evaluators = evaluators


def set_mutables_trainer(
    trainer: Trainer,
    trainer_mutable_attributes: TrainerMutableAttributes,
    client_config: FitConfig | EvaluateConfig | CentralizedConfig,
) -> None:
    """Configure the mutable attributes of the Trainer instance.

    Parameters
    ----------
    trainer : Trainer
        The Composer Trainer instance to configure.
    trainer_mutable_attributes : TrainerMutableAttributes
        The mutable attributes to set in the Trainer instance.
    client_config : FitConfig or EvaluateConfig or CentralizedConfig
        Configuration object for federated client training or evaluation.
    train_cfg : TrainConfig
        Configuration object for training.

    """
    # Set callbacks
    set_mutables_trainer_callbacks_and_loggers(
        trainer=trainer,
        callbacks=trainer_mutable_attributes.callbacks,
        save_latest_filename=trainer_mutable_attributes.save_latest_filename,
        save_filename=trainer_mutable_attributes.save_filename,
        train_cfg=trainer_mutable_attributes.train_cfg,
        loggers=trainer_mutable_attributes.loggers,
    )
    # Train Dataloader
    set_mutables_trainer_train_dataloader(
        trainer=trainer,
        train_dataloader=trainer_mutable_attributes.train_loader,
        client_config=client_config,
    )
    # Metrics and Evaluators
    set_mutables_trainer_eval_dataloader(
        trainer=trainer,
        eval_dataloader=trainer_mutable_attributes.evaluators,
        train_cfg=trainer_mutable_attributes.train_cfg,
    )
    # Reset Timestamp at State
    trainer.state.timestamp = Timestamp()
    trainer.state.eval_timestamp = Timestamp()
    trainer.state.predict_timestamp = Timestamp()


def get_trainer_object(  # noqa: PLR0914, C901, PLR0913, PLR0915, PLR0912
    cfg: DictConfig,
    cid: int | str | None,
    client_config: FitConfig | EvaluateConfig | CentralizedConfig,
    log_name: str | None = None,
    *,
    force_cpu: bool = False,
    no_data_loading: bool = False,
    split_eval: bool = False,
) -> tuple[
    Trainer,
    TrainConfig,
    dict[str, Any],
]:
    """Create and configure a Composer Trainer object based on the configuration.

    Parameters
    ----------
    cfg : DictConfig
        The configuration dictionary.
    cid : int or str or None
        Client ID.
    client_config : FitConfig or EvaluateConfig or CentralizedConfig
        Configuration object for federated client training or evaluation.
    log_name : str or None, optional
        Name for logging, by default None.
    force_cpu : bool, optional
        Whether to force the use of CPU, by default False.
    no_data_loading : bool, optional
        Whether to skip data loading steps, by default False.
    split_eval : bool, optional
        Whether to split evaluation, by default False.

    Returns
    -------
    tuple
        A tuple containing the Trainer object, TrainConfig, and a dictionary of logged
        configurations.

    Raises
    ------
    ValueError
        If there are inconsistencies in the configuration or if required parameters are
        missing.
    BaseContextualError
        If an error occurs while building train or evaluator dataloaders.

    """
    internal_cfg = copy.deepcopy(cfg)
    # NOTE: I got this from the original script in the llm-foundry repository. Assess
    # whether we need to keep it or not
    code_paths = internal_cfg.get("code_paths", [])
    # Import any user provided code
    for code_path in code_paths:
        import_file(code_path)
    # Interpret and check consistency the configuration for (un)frozen layers
    if isinstance(client_config, FitConfig):
        assert not (
            client_config.frozen_layers is not None
            and client_config.unfrozen_layers is not None
        ), "Cannot specify both frozen and unfrozen layers"

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

    # Set logging level
    if train_cfg.python_log_level is not None:
        logging.basicConfig(
            # Example of format string
            # 2022-06-29 11:22:26,152: rank0[822018][MainThread]: INFO: Message here
            format=f"%(asctime)s: rank{dist.get_global_rank()}[%(process)d]"
            f"[%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
            force=True,
        )
        logging.getLogger("llmfoundry").setLevel(
            train_cfg.python_log_level.upper(),
        )  # Foundry module
        logging.getLogger(__name__).setLevel(
            train_cfg.python_log_level.upper(),
        )  # Train script
        logging.getLogger("streaming").setLevel(
            train_cfg.python_log_level.upper(),
        )  # Streaming module

    # Set the device in case multiple GPUs are requested to be
    # independent and not collaborative. If `device == None` the
    # Trainer will automatically initialize PyTorch Distributed
    # with the parameters from the environmental variables.
    visible_devices = ast.literal_eval(str(os.getenv("APPOINTED_CUDA_DEVICE", "null")))
    log(DEBUG, f"Visible devices: {visible_devices}")
    # The worker has been appointed a single GPU
    if type(visible_devices) is int and not force_cpu:
        device: DeviceGPU | DeviceCPU | None = DeviceGPU(device_id=int(visible_devices))
        log(DEBUG, f"Selecting device {visible_devices}, {device}")
    # The worker has been appointed all GPUs available
    elif type(visible_devices) is tuple and not force_cpu:
        assert len(visible_devices) > 1
        device = None
    # The worker is in a CPU-only environment
    else:
        if not force_cpu:
            assert visible_devices is None
        device = DeviceCPU()
        log(DEBUG, f"Selecting device CPU, {device}")
    log(DEBUG, "Initializing dist with device...")
    dist.initialize_dist(get_device(device), timeout=train_cfg.dist_timeout)
    log(DEBUG, "Testing barrier with device...")
    dist.barrier()
    log(DEBUG, "Barrier test passed with device.")
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

    # Filter deprecation warning from torch internal usage
    warnings.filterwarnings(
        action="ignore",
        category=UserWarning,
        message=(
            "torch.distributed.*_base is a private functionand will be deprecated.*"
        ),
    )

    # Check for incompatibilities between the model and data loaders
    validate_config(train_cfg)

    # Define the CUDA allocation configuration that will be set with the environment
    # variable `PYTORCH_CUDA_ALLOC_CONF`
    cuda_alloc_conf = []
    # Get max split size mb
    max_split_size_mb: int | None = train_cfg.max_split_size_mb
    if max_split_size_mb is not None:
        cuda_alloc_conf.append(f"max_split_size_mb:{max_split_size_mb}")

    # Expandable segments
    if train_cfg.expandable_segments:
        cuda_alloc_conf.append("expandable_segments:True")

    # Set the CUDA allocation configuration
    if len(cuda_alloc_conf) > 0:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(cuda_alloc_conf)

    # Set CUDA lazy loading
    # This can save a bit of memory if not all modules are needed
    cuda_load_lazy: bool = train_cfg.cuda_load_lazy
    if cuda_load_lazy:
        os.environ["CUDA_MODULE_LOADING"] = "LAZY"

    # Set seed first
    seed: int = train_cfg.seed
    reproducibility.seed_all(seed)

    # Mandatory model training configs
    model_config = train_cfg.model
    train_loader_config = train_cfg.train_loader

    # Optional fsdp data, fine-tuning, and eval configs
    fsdp_config: dict[str, Any] | None = train_cfg.fsdp_config

    if fsdp_config is not None:
        if "load_planner" in fsdp_config:
            load_planners = list(fsdp_config["load_planner"].items())
            if len(load_planners) > 1:
                msg = "Only one load planner can be specified in the config."
                raise ValueError(
                    msg,
                )
            load_planner_name, load_planner_config = load_planners[0]
            fsdp_config["load_planner"] = build_load_planner(
                load_planner_name,
                **load_planner_config,
            )

        if "save_planner" in fsdp_config:
            save_planners = list(fsdp_config["save_planner"].items())
            if len(save_planners) > 1:
                msg = "Only one save planner can be specified in the config."
                raise ValueError(
                    msg,
                )
            save_planner_name, save_planner_config = save_planners[0]
            fsdp_config["save_planner"] = build_save_planner(
                save_planner_name,
                **save_planner_config,
            )

    eval_loader_config = (
        train_cfg.eval_loader
        if train_cfg.eval_loader is not None
        else train_cfg.eval_loaders
    )
    icl_tasks_config = train_cfg.icl_tasks or train_cfg.icl_tasks_str
    eval_gauntlet_config = train_cfg.eval_gauntlet or train_cfg.eval_gauntlet_str

    # Optional parameters will be set to default values if not specified.
    default_run_name: str = os.environ.get("RUN_NAME", "llm")
    run_name: str = train_cfg.run_name or default_run_name
    is_state_dict_sharded: bool = (
        (fsdp_config.get("state_dict_type", "full") == "sharded")
        if fsdp_config
        else False
    )
    save_latest_filename: str = train_cfg.save_latest_filename or (
        "latest-sharded-rank{rank}" if is_state_dict_sharded else "latest-rank{rank}.pt"
    )
    save_filename: str = train_cfg.save_filename or "ep{epoch}-ba{batch}-rank{rank}.pt"

    # Enable autoresume from model checkpoints if possible
    autoresume_default: bool = False
    if (
        train_cfg.save_folder is not None
        and not train_cfg.save_overwrite
        and not train_cfg.save_weights_only
    ):
        autoresume_default = True

    if not train_cfg.autoresume and autoresume_default:
        log(
            INFO,
            "As run_name, save_folder, and save_latest_filename are set, \
                changing autoresume default to True...",
        )

    # Optional tp config
    tp_config_dict: dict[str, Any] | None = train_cfg.tp_config

    # Warn if FSDP or TP is enabled but user only has 1 GPU
    if dist.get_world_size() == 1 and (
        fsdp_config is not None or tp_config_dict is not None
    ):
        parallelism = ""
        if fsdp_config is not None:
            parallelism += "FSDP"
        if tp_config_dict is not None:
            parallelism += "+TP" if fsdp_config is not None else "TP"
        warnings.warn(
            f"{parallelism} is not applicable for single-GPU training."
            " Reverting to DDP.",
            stacklevel=2,
        )
        fsdp_config = None
        tp_config_dict = None

    # Initialize context
    init_context = process_init_device(model_config, fsdp_config, tp_config_dict)
    logged_cfg.update({"fsdp_config": fsdp_config}, merge=True)
    logged_cfg.update({"tp_config": tp_config_dict}, merge=True)

    # Build tokenizer
    log(INFO, "Building tokenizer...")
    tokenizer_name = train_cfg.tokenizer["name"]
    tokenizer_kwargs = train_cfg.tokenizer.get("kwargs", {})
    tokenizer = build_tokenizer(tokenizer_name, tokenizer_kwargs)

    scheduler_params: dict[str, dict[str, Any]] = train_cfg.scheduler.pop(
        "schedulers",
        {},
    )

    scheduler = [
        build_scheduler(params.pop("name"), params)
        for params in scheduler_params.values()
    ]

    # Loggers
    loggers = (
        [
            build_logger(str(name), logger_cfg)
            for name, logger_cfg in train_cfg.loggers.items()
        ]
        if train_cfg.loggers
        else []
    )

    mosaicml_logger = find_mosaicml_logger(loggers)
    if mosaicml_logger is None:
        mosaicml_logger = maybe_create_mosaicml_logger()
        if mosaicml_logger is not None:
            # mosaicml_logger will be None if run isn't on MosaicML platform
            loggers.append(mosaicml_logger)

    if train_cfg.metadata is not None:
        # Optionally flatten the metadata for logging
        if train_cfg.flatten_metadata:
            logged_cfg.pop("metadata", None)
            common_keys = set(
                logged_cfg.keys(),
            ) & set(train_cfg.metadata.keys())
            if len(common_keys) > 0:
                msg = (
                    f"Keys {common_keys} are already present in the config."
                    " Please rename them in metadata or set flatten_metadata=False"
                    " to avoid flattening the metadata in the logged config."
                )
                raise ValueError(
                    msg,
                )

            logged_cfg.update(train_cfg.metadata, merge=True)

        if mosaicml_logger is not None:
            mosaicml_logger.log_metrics(train_cfg.metadata)
            mosaicml_logger._flush_metadata(force_flush=True)  # noqa: SLF001

    # Profiling
    profiler: Profiler | None = None
    profiler_cfg = train_cfg.profiler
    if profiler_cfg:
        profiler_schedule_cfg: dict = pop_config(
            profiler_cfg,
            "schedule",
            must_exist=True,
        )
        profiler_schedule = cyclic_schedule(**profiler_schedule_cfg)
        # Only support json trace handler
        profiler_trace_handlers: list[TraceHandler] = []
        profiler_trace_cfg: dict | None = pop_config(
            profiler_cfg,
            "json_trace_handler",
            must_exist=False,
            default_value=None,
        )
        if profiler_trace_cfg:
            profiler_trace_handlers.append(
                JSONTraceHandler(**profiler_trace_cfg),
            )
        profiler = Profiler(
            **profiler_cfg,
            trace_handlers=profiler_trace_handlers,
            schedule=profiler_schedule,
        )

    # Callbacks
    callback_configs = train_cfg.callbacks or {}
    callbacks: list[Callback] = [
        build_callback(
            name=str(name),
            kwargs=callback_cfg,
            train_config=logged_cfg,
        )
        for name, callback_cfg in callback_configs.items()
    ]

    use_async_eval = any(isinstance(c, AsyncEval) for c in callbacks)

    algorithm_configs = train_cfg.algorithms or {}

    # Algorithms
    algorithms = [
        build_algorithm(str(name), algorithm_cfg)
        for name, algorithm_cfg in algorithm_configs.items()
    ]

    # Dataloaders
    log(INFO, "Building train loader...")
    try:
        train_loader = None
        train_streams: dict[str, dict[str, Any]] | None = None
        if not no_data_loading:
            assert train_cfg.train_loader["dataset"] is not None, (
                "Dataset for train loader is not set."
            )
            train_streams = train_loader_config["dataset"]["streams"]

            if client_config.use_unigram_metrics:
                assert train_streams is not None, "Train streams must be provided."

                add_unigram_metrics(
                    client_config,
                    train_cfg,
                    train_streams,
                )
            assert isinstance(train_loader_config, dict), (
                "Expected train_loader_config to be a dict,"
                f" got {type(train_loader_config)}"
            )
            train_loader = build_dataloader(
                train_loader_config,
                tokenizer,
                train_cfg.device_train_batch_size,
            )
    except BaseContextualError as e:
        e.location = TrainDataLoaderLocation
        raise

    if mosaicml_logger is not None:
        mosaicml_logger.log_metrics({"data_validated": time.time()})

    # Evaluation
    if use_async_eval:
        evaluators = []
        if train_cfg.eval_first:
            warnings.warn(
                "AsyncEval callback does not support eval_first=True. Ignoring.",
                stacklevel=2,
            )
            train_cfg.eval_first = False
    else:
        try:
            log(INFO, "Building eval loader...")
            # Extracting the destination directory for the eval gauntlet
            destination_dir: str | None = None
            if eval_gauntlet_config is not None and not isinstance(
                eval_gauntlet_config,
                str,
            ):
                destination_dir = eval_gauntlet_config.pop("destination_dir", None)
            if (
                icl_tasks_config_dict is not None
                and icl_tasks_config is not None
                and not isinstance(icl_tasks_config, str)
            ):
                set_icl_tasks_root_dir(
                    icl_tasks_config,
                    icl_tasks_config_dict["root_dir"],
                )
            eval_icl_seq_len: int = train_cfg.icl_seq_len or train_cfg.max_seq_len
            evaluators, _, eval_gauntlet_callback = build_evaluators(
                eval_loader_config=eval_loader_config,
                icl_tasks_config=icl_tasks_config,
                eval_gauntlet_config=eval_gauntlet_config,
                tokenizer=tokenizer,
                device_eval_batch_size=train_cfg.device_eval_batch_size,
                icl_seq_len=eval_icl_seq_len,
                icl_subset_num_batches=train_cfg.icl_subset_num_batches,
                device_eval_microbatch_size=train_cfg.device_eval_microbatch_size,
                destination_dir=destination_dir,
            )
            if eval_gauntlet_callback is not None:
                callbacks.append(eval_gauntlet_callback)
        except BaseContextualError as e:
            e.location = EvalDataLoaderLocation
            raise

    if mosaicml_logger is not None:
        log_train_analytics(
            mosaicml_logger,
            model_config,
            train_loader_config,
            eval_loader_config,
            train_cfg.callbacks,
            tokenizer_name,
            train_cfg.load_path,
            icl_tasks_config,
            eval_gauntlet_config,
        )

    # Build Model
    log(INFO, "Initializing model...")
    assert isinstance(
        model_config,
        dict,
    ), f"Expected model_config to be a dict, got {type(model_config)}"
    # Add unigram metrics to the model
    if client_config.use_unigram_metrics:
        if "additional_train_metrics" not in model_config:
            model_config["additional_train_metrics"] = []
        model_config["additional_train_metrics"].extend(
            list(UNIGRAM_METRIC_NAMES_AND_CLASSES.keys()),
        )
    name = model_config.pop("name")
    assert isinstance(name, str)
    assert isinstance(model_config, dict)
    model = build_composer_model(
        name=name,
        tokenizer=tokenizer,
        init_context=init_context,
        master_weights_dtype=model_config.pop("master_weights_dtype", None),
        cfg=model_config,
    )

    _log_num_params(model, logged_cfg)

    if isinstance(client_config, FitConfig) and (
        client_config.frozen_layers is not None
        or client_config.unfrozen_layers is not None
    ):
        # Get the full list of parameter names
        all_param_names = get_list_of_parameters_names(model)
        # Apply freezing/unfreezing
        freeze_blocks(model, client_config.frozen_layers, client_config.unfrozen_layers)
        # Get list of unfrozen parameter names
        client_config.unfrozen_layers = get_list_of_parameters_names(model)
        # Get list of frozen parameter names
        client_config.frozen_layers = list(
            set(all_param_names) - set(client_config.unfrozen_layers),
        )

    # TP config
    tp_config: TPConfig | None = None
    if tp_config_dict is not None:
        strategy = tp_config_dict.pop("strategy")
        layer_plan = build_tp_strategies(strategy, model)
        tp_config = TPConfig(**tp_config_dict, layer_plan=layer_plan)

    # Parallelism config
    parallelism_config = {"fsdp": fsdp_config, "tp": tp_config}

    # Optimizer
    optimizer_name: str = train_cfg.optimizer.pop("name")
    optimizer_cfg = train_cfg.optimizer
    optimizer = build_optimizer(model, optimizer_name, optimizer_cfg)

    # Now add the eval metrics
    try:
        if eval_loader_config is not None and not use_async_eval:
            eval_metrics = model.get_metrics(is_train=False)
            non_icl_metrics = [
                metric_name
                for metric_name, metric in eval_metrics.items()
                if not isinstance(metric, InContextLearningMetric)
            ]
            evaluators = add_metrics_to_eval_loaders(
                evaluators,
                non_icl_metrics,
            )
    except BaseContextualError as e:
        e.location = EvalDataLoaderLocation
        raise

    compile_config = train_cfg.compile_config

    # Build the Trainer
    trainer = Trainer(
        run_name=run_name,
        seed=seed,
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=evaluators,
        optimizers=optimizer,
        schedulers=scheduler,
        max_duration=train_cfg.max_duration,
        eval_interval=train_cfg.eval_interval,
        eval_subset_num_batches=train_cfg.eval_subset_num_batches,
        progress_bar=train_cfg.progress_bar,
        log_to_console=train_cfg.log_to_console,
        console_log_interval=train_cfg.console_log_interval,
        loggers=loggers,
        callbacks=callbacks,
        precision=train_cfg.precision if not isinstance(device, DeviceCPU) else None,
        algorithms=algorithms,
        device_train_microbatch_size=train_cfg.device_train_microbatch_size,
        save_folder=train_cfg.save_folder,
        save_ignore_keys=train_cfg.save_ignore_keys,
        save_filename=save_filename,
        save_latest_filename=save_latest_filename,
        save_interval=train_cfg.save_interval,
        save_num_checkpoints_to_keep=train_cfg.save_num_checkpoints_to_keep,
        save_overwrite=train_cfg.save_overwrite,
        save_weights_only=train_cfg.save_weights_only,
        save_metrics=True,
        load_path=train_cfg.load_path,
        load_weights_only=train_cfg.load_weights_only,
        load_strict_model_weights=train_cfg.load_strict_model_weights,
        load_ignore_keys=train_cfg.load_ignore_keys,
        autoresume=train_cfg.autoresume,
        python_log_level=train_cfg.python_log_level,
        dist_timeout=train_cfg.dist_timeout,
        profiler=profiler,
        compile_config=compile_config,
        device=device,
        # NOTE: Force sync at the final batch
        ddp_sync_strategy=DDPSyncStrategy.FORCED_SYNC,
        train_subset_num_batches=train_cfg.train_subset_num_batches,
        parallelism_config=parallelism_config,
        spin_dataloaders=train_cfg.spin_dataloaders,
        accumulate_train_batch_on_tokens=train_cfg.accumulate_train_batch_on_tokens,
    )
    _sort_callbacks(trainer)
    return trainer, train_cfg, logged_cfg
