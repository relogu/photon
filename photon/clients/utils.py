"""Utility functions for model parameter handling and configurations in FL.

This module provides various utility functions to manage model parameters, configure
training settings, and handle data loading for federated learning. It includes functions
to set trainer timestamps, print trainable parameters, retrieve model parameters,
randomize and personalize model layers, and more.

Functions
---------
- set_trainer_timestamp(trainer: Trainer, timestamp: int) -> None
    Set the trainer's timestamp to a specific batch value.

- get_initial_parameters(cfg: BaseConfig) -> Parameters
    Retrieve the initial parameters for the federated learning server model.

- get_raw_model_parameters(
        cfg: DictConfig,
        verbose: bool = False,
        return_names: bool = False
    ) -> NDArrays | tuple[NDArrays, list[str]]
    Retrieve the raw model parameters from the configuration.

- randomize_layers(
        parameters: NDArrays,
        dummy_config: DictConfig,
        names: list[str],
        random_layers: list[str],
        truly_random_init: bool,
        cid: int = 0,
        server_round: int = 1
    ) -> None
    Randomize specified layers of the model parameters.

- personalize_layers(
        parameters: NDArrays,
        initial_trainer_parameters: NDArrays,
        personalized_layers: list[str],
        names: list[str],
        unfrozen_names: list[str]
    ) -> None
    Personalize specified layers of the model parameters.

Imports
-------
- copy
- random
- warnings
- torch
- logging.DEBUG
- typing.cast
- pathlib.Path
- composer.Trainer
- composer.utils.reproducibility
- flwr.common.logger.log
- flwr.common.typing.NDArrays
- flwr.common.parameters_to_ndarrays
- flwr.common.ndarrays_to_parameters
- flwr.common.Parameters
- llmfoundry.utils.builders.build_tokenizer
- llmfoundry.utils.builders.build_composer_model
- llmfoundry.utils.config_utils.process_init_device
- llmfoundry.utils.config_utils.make_dataclass_and_log_config
- llmfoundry.utils.config_utils.TRAIN_CONFIG_KEYS
- llmfoundry.utils.config_utils.TrainConfig
- llmfoundry.command_utils.train.validate_config
- omegaconf.DictConfig
- omegaconf.OmegaConf
- photon.conf.base_schema.BaseConfig
- photon.utils.get_list_of_parameters_names
- photon.utils.get_trainable_params_dict
- photon.utils.load_model_parameters_from_file

Example Usage
-------------
>>> from omegaconf import OmegaConf
>>> from composer import Trainer
>>> cfg = OmegaConf.create({...})
>>> trainer = Trainer(...)
>>> set_trainer_timestamp(trainer, timestamp=100)
>>> initial_params = get_initial_parameters(cfg)
>>> raw_params = get_raw_model_parameters(cfg)
>>> randomize_layers(
>>>     parameters, dummy_config, names, random_layers,
>>>     truly_random_init=True, cid=1, server_round=2,
>>> )
>>> personalize_layers(
>>>     parameters, initial_trainer_parameters,
>>>     personalized_layers, names, unfrozen_names,
>>> )
"""

# ruff: noqa: ERA001
import atexit
import copy
import gc
import os
import random
import time
import warnings
from dataclasses import asdict
from logging import DEBUG, WARNING
from pathlib import Path
from typing import Any, cast

import numpy as np
import streaming
import torch
from composer import Trainer
from composer.utils import dist, reproducibility
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.common.typing import NDArrays, Scalar
from llmfoundry.command_utils.train import (
    validate_config,
)
from llmfoundry.utils.builders import (
    build_composer_model,
    build_tokenizer,
)
from llmfoundry.utils.config_utils import (
    TRAIN_CONFIG_KEYS,
    TrainConfig,
    make_dataclass_and_log_config,
    process_init_device,
)
from omegaconf import DictConfig, OmegaConf
from streaming.base.shared.memory import SharedMemory, shared_memory_list

from photon.clients.configs import FitConfig
from photon.clients.llm_config_functions import set_client_load_path
from photon.conf.base_schema import BaseConfig
from photon.utils import (
    ClientState,
    get_list_of_parameters_names,
    get_parameters_from_state,
    get_trainable_params_dict,
    is_literal_for_ast,
    l2_norm_of_momenta,
    load_model_parameters_from_file,
    parameters_checker,
    sum_of_squares,
)


def get_client_state_struct(fit_config: FitConfig, cid: int | str) -> ClientState:
    """Retrieve the state of a specific client.

    Parameters
    ----------
    fit_config : FitConfig
        The configuration containing the client states and server steps.
    cid : int | str
        The client ID, which can be an integer or a string.

    Returns
    -------
    ClientState: The state structure of the client.

    Raises
    ------
    ValueError: If the client states or server steps are not provided in the
        configuration

    """
    # Retrieve the clients' states
    if fit_config.client_state is None:
        msg = "Client states must be provided."
        raise ValueError(msg)
    if fit_config.server_steps_cumulative is None:
        msg = "Server steps must be provided."
        raise ValueError(msg)

    # Extract current client's state
    return fit_config.client_state[str(cid)]


def set_initial_config_from_fit_config(
    fit_config: FitConfig,
    llm_config: DictConfig,
    cid: int | str,
) -> tuple[bool, bool, int | None]:
    """Set the initial configuration for a client based on the fit configuration.

    This function configures the local model based on the provided fit configuration,
    including setting the loading path, handling checkpoints, and adjusting model
    parameters.

    Parameters
    ----------
    fit_config : FitConfig
        The configuration containing the client states and server steps.
    llm_config : DictConfig
        The configuration dictionary for the local model.
    cid : int | str
        The client ID, which can be an integer or a string.

    Returns
    -------
        tuple[bool, bool, int | None]
            A tuple containing:
            - A boolean indicating whether to skip the iteration.
            - A boolean indicating whether a checkpoint exists.
            - The cumulative server steps.

    Raises
    ------
    ValueError: If the server steps cumulative is None and a checkpoint reset is
    requested.

    """
    # Get the number of local steps done by the current client
    num_batches_trained = int(str(llm_config["local_steps"]).replace("ba", ""))

    # Set the loading path
    server_steps_cumulative = fit_config.server_steps_cumulative

    skip_iteration = False
    checkpoint_exists = False
    if not fit_config.reset_checkpoint:
        if server_steps_cumulative is None:
            msg = "Server steps cumulative is None and we want to reset a checkpoint."
            raise ValueError(msg)

        skip_iteration, checkpoint_exists = set_client_load_path(
            llm_config,
            cid,
            server_steps_cumulative + num_batches_trained,
        )
    llm_config.load_ignore_keys = ["*scheduler*"]  # type: ignore[union-attr]
    if fit_config.reset_optimizer:
        # Ignoring the optimizer state if loading a checkpoint
        llm_config.load_ignore_keys += ["*optim*"]  # type: ignore[union-attr]
        # Ignoring the optimizer state when saving a checkpoint
        llm_config.save_ignore_keys = ["*optim*"]  # type: ignore[union-attr]

    if fit_config.reset_dataset_state:
        # Ignoring the dataset state if loading a checkpoint
        llm_config.load_ignore_keys += ["*dataset_state*"]

    # NOTE: The following, when re-loading from a checkpoint, returns a weird error
    # if not skip_iteration:
    #     # Ignoring loading the model as we need to set it from the server
    #     cfg.load_ignore_keys += ["*model*"]
    # Extract configs to build the trainer
    if (
        "vocab_size" in llm_config.model
        and (vocab_size := fit_config.resize_vocab) is not None
    ):
        llm_config.model.vocab_size = vocab_size
    return (
        skip_iteration,
        checkpoint_exists,
        server_steps_cumulative,
    )


def set_optimizer_state(  # noqa: PLR0914
    momenta: tuple[NDArrays, ...],
    trainer: Trainer,
    names: list[str],
    train_metrics: dict[str, Scalar],
    fit_config: FitConfig,
) -> float:
    """Replace the momenta of the optimizer with the given momenta.

    Called when using local adaptive optimizers.
    Serves to synchronize momenta across all clients.

    Parameters
    ----------
    momenta : tuple[NDArrays, ...]
        The momenta.
    trainer : Trainer
        The trainer object.
    names : list[str]
        The parameter names.
    train_metrics : dict[str, Scalar]
        The training metrics.
    fit_config : FitConfig
        The fit configuration.

    Returns
    -------
    float: The time taken to set the optimizer state.

    Raises
    ------
    ValueError: If the parameter is not found in the optimizer state.

    """
    log(DEBUG, "Setting optimizer state")

    start_time = time.time_ns()

    first_momentum, second_momentum = momenta

    # The structure of this is not well-documented at all
    # Essentially it is:
    # opt_state_dict.keys()
    # dict_keys(['DecoupledAdamW'])
    # opt_state_dict['DecoupledAdamW'].keys()
    # dict_keys(['state', 'param_groups'])
    # opt_state_dict['state'].keys()
    # List of parameter names
    # opt_state_dict['state']['param_name'].keys()
    # dict_keys(['step', 'exp_avg', 'exp_avg_sq'])
    # Where exp_avg and exp_avg_sq are the first and second moments

    optim_state_dict = trainer.state.get_optim_state_dict()
    state: dict[str, dict[str, torch.Tensor]] = optim_state_dict[
        next(iter(optim_state_dict))
    ].get("state", {})

    is_rank_0 = int(os.getenv("LOCAL_RANK", "")) == 0
    state_momenta_norms = l2_norm_of_momenta(state) if is_rank_0 else (0.0, 0.0)
    first_momentum_delta: float = 0.0
    second_momentum_delta: float = 0.0

    for i, name in enumerate(names):
        if fit_config.personalized_layers is not None and any(
            pl in name or name in pl for pl in fit_config.personalized_layers
        ):
            log(WARNING, "Skipping personalized layer %s", name)
            continue

        real_name = next((rn for rn in state if name in rn), None)

        if real_name is None:
            msg = f"Parameter {name} not found in optimizer state"
            raise ValueError(msg)

        if "step" in state[real_name]:
            # Track and change step
            # NOTE: needed for bias correction in AdamW
            # and/or time-dependent clipping Adopt
            state[real_name]["step"] = torch.as_tensor(
                fit_config.server_steps_cumulative,
                device=state[real_name]["step"].device,
                dtype=state[real_name]["step"].dtype,
            )
        else:
            log(WARNING, "Step not found in optimizer state for %s", name)

        if "exp_avg" in state[real_name]:
            # Replace first momentum
            new_first_momentum = torch.as_tensor(
                first_momentum[i],
                device=state[real_name]["exp_avg"].device,
                dtype=state[real_name]["exp_avg"].dtype,
            )
            first_momentum_delta += torch.sum(
                (state[real_name]["exp_avg"] - new_first_momentum) ** 2,
            ).item()
            state[real_name]["exp_avg"] = new_first_momentum
        else:
            log(WARNING, "First momentum not found in optimizer state for %s", name)

        if "exp_avg_sq" in state[real_name]:
            # Replace second momentum
            new_second_momentum = torch.as_tensor(
                second_momentum[i],
                device=state[real_name]["exp_avg_sq"].device,
                dtype=state[real_name]["exp_avg_sq"].dtype,
            )
            second_momentum_delta += torch.sum(
                (state[real_name]["exp_avg_sq"] - new_second_momentum.detach()) ** 2,
            ).item()
            state[real_name]["exp_avg_sq"] = new_second_momentum
        else:
            log(WARNING, "Second momentum not found in optimizer state for %s", name)

    first_momentum_delta = float(np.sqrt(first_momentum_delta))
    second_momentum_delta = float(np.sqrt(second_momentum_delta))

    dist.barrier()
    time_to_set = (time.time_ns() - start_time) * 1e-9
    log(DEBUG, "Optimizer state loaded")

    if is_rank_0:
        new_state_momenta_norms = l2_norm_of_momenta(
            trainer.state.get_optim_state_dict()[next(iter(optim_state_dict))]["state"],
        )
        train_metrics |= {
            "client/local_adopt/l2_norm/pre_first_moment": state_momenta_norms[0],
        }
        train_metrics |= {
            "client/local_adopt/l2_norm/pre_second_moment": state_momenta_norms[1],
        }
        train_metrics |= {
            "client/local_adopt/l2_norm/post_first_moment": new_state_momenta_norms[0],
        }
        train_metrics |= {
            "client/local_adopt/l2_norm/post_second_moment": new_state_momenta_norms[1],
        }
        train_metrics |= {
            "client/local_adopt/l2_norm/delta_first_moment": first_momentum_delta,
        }
        train_metrics |= {
            "client/local_adopt/l2_norm/delta_second_moment": second_momentum_delta,
        }

    return time_to_set


def manipulate_pre_training_ndarrays(  # noqa: PLR0913, PLR0917
    payload: NDArrays,
    trainer: Trainer,
    cid: int | str,
    configs: tuple[FitConfig, DictConfig],
    client_state_struct: ClientState,
    train_metrics: dict[str, Scalar],
) -> NDArrays:
    """Manipulate the initial parameters before training.

    This function personalizes and randomizes layers based on the fit configuration
    and client state, and checks the parameters before training.

    Parameters
    ----------
    payload : NDArrays
        The initial parameters of the model.
    trainer : Trainer
        The trainer instance used for training.
    cid : int | str
        The client ID, which can be an integer or a string.
    configs : tuple[FitConfig, DictConfig]
        A tuple containing the fit configuration and the local model configuration.
    client_state_struct : ClientState
        The state structure of the client.
    train_metrics : dict[str, Scalar]
        A dictionary to store training metrics.

    Returns
    -------
    tuple[NDArrays, list[str]]
        A tuple containing:
        - The initial parameters of the model after manipulation.
        - The list of parameter names.

    Raises
    ------
    ValueError: If unfrozen layers are not provided when personalizing.

    """
    fit_config, dummy_config = configs
    # Get initial parameters from trainer
    initial_trainer_parameters = get_parameters_from_state({}, trainer)
    # Get parameter names from trainer
    names = get_list_of_parameters_names(model=trainer.state.model)
    if not fit_config.aggregate_momenta:
        initial_parameters = payload
    else:
        # NOTE: We need to extract parameters
        # from the momenta list
        # the param structure is [params, momenta, momenta]
        # where each NDarrays has the same length
        initial_parameters, first_momentum, second_momentum = [
            payload[i : i + len(initial_trainer_parameters)]
            for i in range(0, len(payload), len(initial_trainer_parameters))
        ]
        time_to_set = set_optimizer_state(
            (first_momentum, second_momentum),
            trainer,
            names,
            train_metrics,
            fit_config,
        )
        train_metrics |= {"client/local_adopt/set_optimizer_state_time": time_to_set}

    # Get list of personalized layers from config
    personalized_layers: list[str] = (
        fit_config.personalized_layers
        if fit_config.personalized_layers is not None
        else []
    )
    # Personalize layers
    if personalized_layers:
        if fit_config.unfrozen_layers is None:
            msg = "Unfrozen layers must be provided when personalizing layers."
            raise ValueError(msg)

        personalize_layers(
            parameters=initial_parameters,
            initial_trainer_parameters=initial_trainer_parameters,
            personalized_layers=personalized_layers,
            names=names,
            unfrozen_names=fit_config.unfrozen_layers,
        )
    # Get list of random layers from config
    random_layers: list[str] = (
        fit_config.random_layers if fit_config.random_layers is not None else []
    )
    # Randomize layers
    if (
        random_layers
        and fit_config.random_init_freq > 0
        and client_state_struct.local_steps_cumulative % fit_config.random_init_freq
        == 0
    ):
        randomize_layers(
            parameters=initial_parameters,
            dummy_config=dummy_config,
            names=names,
            random_layers=random_layers,
            truly_random_init=fit_config.truly_random_init,
            cid=int(cid),
            server_round=fit_config.server_round,
        )
    # Check parameters before training
    parameters_checker(initial_trainer_parameters, initial_parameters, is_equal=False)
    return initial_trainer_parameters


def post_process_client_result(  # noqa: PLR0913, PLR0917
    train_metrics: dict[str, Scalar],
    client_state_struct: ClientState,
    llm_config: DictConfig,
    trainer: Trainer,
    initial_parameters: NDArrays,
    cid: int | str,
    fit_config: FitConfig,
) -> tuple[NDArrays, int]:
    """Compute post-training metrics on the client result.

    This function retrieves the model parameters after training, calculates the number
    of samples trained, updates the client state, and collects various training metrics.

    Parameters
    ----------
    train_metrics : dict[str, Scalar]
        A dictionary to store training metrics.
    client_state_struct : ClientState
        The state structure of the client.
    llm_config : DictConfig
        The configuration dictionary for the local model.
    trainer : Trainer
        The trainer instance used for training.
    initial_parameters : NDArrays
        The initial parameters of the model.
    cid : int | str
        The client ID, which can be an integer or a string.
    fit_config : FitConfig
        The configuration containing the client states and server steps.

    Returns
    -------
    tuple[NDArrays, int]
        A tuple containing:
        - The model parameters after training.
        - The number of samples trained.

    Raises
    ------
    ValueError: If the parameter is not found in the optimizer

    """
    # Retrieve model parameters
    start_time = time.time_ns()
    model_parameters = get_parameters_from_state({}, trainer)
    names = (
        get_list_of_parameters_names(model=trainer.state.model)
        if not fit_config.unfrozen_layers
        else fit_config.unfrozen_layers
    )

    # NOTE: We removed here the parameter checkers for efficiency reasons.
    # Re-add them if needed

    train_metrics |= {
        "client/fit_get_parameters_time": (time.time_ns() - start_time) * 1e-9,
    }
    # Retrieve the global train batch size
    global_train_batch_size = int(llm_config["global_train_batch_size"])
    # Get the number of local steps done by the current client
    num_batches_trained = int(str(llm_config["local_steps"]).replace("ba", ""))
    client_state_struct.steps_done = num_batches_trained
    # Retrieve number of samples trained
    # NOTE: We assume all the clients train with the same batch size,
    # so we just consider the number of local steps
    # NOTE: Assuming that this is the correct value of local steps
    # for the client to train in this particular round and no

    n_samples_trained: int = num_batches_trained * global_train_batch_size

    client_state_struct.local_steps_cumulative += num_batches_trained

    client_state_struct.local_timestamp = {
        k: v
        for k, v in trainer.state.timestamp.copy().state_dict().items()
        if is_literal_for_ast(repr(v))
    }

    # Retrieve training metrics
    train_metrics |= {
        k: v.detach().cpu().item()  # type: ignore[attr-defined]
        for k, v in trainer.state.train_metric_values.items()
    }
    # Only rank 0 collects metrics related to the pseudo gradients
    if int(os.getenv("LOCAL_RANK", "")) == 0:
        start_time = time.time_ns()
        per_layer_sum_of_squares = [
            sum_of_squares([x - y])
            for x, y in zip(initial_parameters, model_parameters, strict=False)
        ]

        for i, plss in enumerate(per_layer_sum_of_squares):
            train_metrics |= {
                f"client/layer/{i}/l2_norm_of_pseudo_gradient": float(np.sqrt(plss)),
            }

        l2_norm_of_pseudo_gradient: float = float(
            np.sqrt(sum(per_layer_sum_of_squares)),
        )

        train_metrics |= {"client/l2_norm_pseudo_gradient": l2_norm_of_pseudo_gradient}
        train_metrics |= {
            "client/fit_metrics_collection_time": (time.time_ns() - start_time) * 1e-9,
        }
        train_metrics |= {"client_state_acc": str({cid: asdict(client_state_struct)})}

        # NOTE: for some data parallelism implementations
        # the optimizer state dict may be empty
        # for all workers except rank 0
        # thus it only makes sense to add optimizer
        # states for rank 0
        if fit_config.aggregate_momenta:
            optim_state_dict = trainer.state.get_optim_state_dict()
            state: dict[str, dict[str, torch.Tensor]] = optim_state_dict[
                next(iter(optim_state_dict))
            ]["state"]
            first_momentum_acc: NDArrays = []
            second_momentum_acc: NDArrays = []

            # NOTE: these are considered to be the trainable parameters of the model
            for name in names:
                real_name = next((rn for rn in state if name in rn), None)
                if real_name is None:
                    msg = f"Parameter {name} not found in optimizer state"
                    raise ValueError(msg)
                if "exp_avg" in state[real_name]:
                    first_momentum_acc.append(
                        state[real_name]["exp_avg"].detach().to("cpu").numpy(),
                    )
                if "exp_avg_sq" in state[real_name]:
                    second_momentum_acc.append(
                        state[real_name]["exp_avg_sq"].detach().to("cpu").numpy(),
                    )

            model_parameters.extend(first_momentum_acc)
            model_parameters.extend(second_momentum_acc)

    return model_parameters, n_samples_trained


def streaming_shms_clean_up() -> None:
    """Clean up leaking and stale shared memories.

    This function performs the following clean-up tasks:
    - Cleans up leaking shared memories from the shared_memory_list.
    - Un-registers the cleanup function from atexit.
    - Clears the shared_memory_list.
    - Cleans up stale shared memory using the streaming library.
    - Runs the garbage collector to free up memory.
    """
    # NOTE: Clean up leaking shared memories
    for shm in shared_memory_list:
        SharedMemory.cleanup(shm)
        atexit.unregister(SharedMemory.cleanup)
    shared_memory_list.clear()
    # Cleaning stale shared memory
    streaming.base.util.clean_stale_shared_memory()  # type: ignore[reportAttributeAccessIssue]
    # Clean-up garbage collector
    gc.collect()


def get_initial_parameters(cfg: BaseConfig) -> Parameters:
    """Retrieve the initial parameters for the federated learning server model.

    This function returns the initial parameters of the model using the configuration.
    If a pretrained model path is specified in the configuration (`cfg`), it loads its
    parameters from the specified file. Otherwise, it returns random parameters
    based on the provided large language model (LLM) configuration. Also, it logs
    the shapes and names of the initial parameters for debugging purposes.

    Parameters
    ----------
    cfg : BaseConfig
        The configuration object containing the pretrained model path and LLM config.

    Returns
    -------
    'Parameters'
        The initial parameters of the model, either loaded from a pretrained model or
        initialized randomly based on the LLM configuration.

    Raises
    ------
    ValueError: If the number of pretrained parameters does not match the expected

    """
    llm_config = cfg.llm_config
    OmegaConf.resolve(llm_config)
    OmegaConf.set_struct(llm_config, value=False)
    initial_parameters_ndarrays: NDArrays
    (initial_parameters_ndarrays, _names) = cast(
        "tuple[NDArrays, list[str]]",
        get_raw_model_parameters(
            copy.deepcopy(llm_config),
            return_names=True,
            aggregate_momenta=cfg.fl.aggregate_momenta,
        ),
    )

    if cfg.pretrained_model_path:
        log(
            DEBUG,
            "FL server is loading pretrained model from %s",
            cfg.pretrained_model_path,
        )
        pretrained_params = load_model_parameters_from_file(
            Path(cfg.pretrained_model_path),
        )
        if len(pretrained_params) != len(initial_parameters_ndarrays):
            msg = (
                f"Expected {len(initial_parameters_ndarrays)} parameters, "
                f"but got {len(pretrained_params)}"
            )
            raise ValueError(msg)
        return ndarrays_to_parameters(pretrained_params)

    log(
        DEBUG,
        "FL server initializes model with random parameters.",
    )

    return ndarrays_to_parameters(initial_parameters_ndarrays)


def get_raw_model_parameters(
    cfg: DictConfig,
    *,
    verbose: bool = False,
    return_names: bool = False,
    aggregate_momenta: bool = False,
) -> NDArrays | tuple[NDArrays, list[str]]:
    """Retrieve the raw model parameters from the configuration.

    This function builds a model based on the provided configuration and retrieves its
    trainable parameters as numpy arrays. Optionally, it can also return the names of
    the parameters.

    Parameters
    ----------
    cfg : DictConfig
        The configuration object containing model and training parameters.
    verbose : bool, optional
        If True, logs the model summary (default is False).
    return_names : bool, optional
        If True, returns a tuple containing the parameter arrays and their names
        (default is False).
    aggregate_momenta : bool, optional
        If True, aggregates the momenta of the model parameters (default is False).

    Returns
    -------
    NDArrays or tuple[NDArrays, list[str]]
        If `return_names` is False, returns a list of numpy arrays representing the
        model parameters.
        If `return_names` is True, returns a tuple containing the list of numpy arrays
        and a list of parameter names.

    Raises
    ------
    TypeError: If the model configuration is not a dictionary.

    Example
    -------
    >>> from omegaconf import OmegaConf
    >>> cfg = OmegaConf.create({...})
    >>> parameters = get_raw_model_parameters(cfg)
    >>> print(parameters)

    """
    # Resolve all interpolation variables as early as possible
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, value=False)
    # Deep copy the configuration to prevent any dangerous modification
    internal_cfg = copy.deepcopy(cfg)

    # Filter deprecation warning from torch internal usage
    warnings.filterwarnings(
        action="ignore",
        category=UserWarning,
        message=(
            "torch.distributed.*_base is a private functionand will be deprecated.*"
        ),
    )

    # NOTE: We need to extract a set of global variables from the configuration object
    # to prevent the dataclass creator to crash
    internal_cfg.pop("data_local", None)
    internal_cfg.pop("data_remote", None)
    internal_cfg.pop("global_seed", None)
    internal_cfg.pop("local_steps", None)
    internal_cfg.pop("name", None)
    # NOTE: This contains OUR global parameters for the ICL tasks that the
    # `make_dataclass_and_log_config` cannot interpret, so we need to pop it
    _icl_tasks_config_dict: dict[str, Any] | None = internal_cfg.pop(
        "icl_tasks_config",
        None,
    )
    _logged_cfg, train_cfg = make_dataclass_and_log_config(
        internal_cfg,
        TrainConfig,
        TRAIN_CONFIG_KEYS,
        transforms="all",
        icl_tasks_required=internal_cfg.get("icl_tasks", None) is not None,
    )
    # Check for incompatibilities between the model and data loaders
    validate_config(train_cfg)
    # Build tokenizer
    tokenizer_name = train_cfg.tokenizer["name"]
    tokenizer_kwargs = train_cfg.tokenizer.get("kwargs", {})
    tokenizer = build_tokenizer(tokenizer_name, tokenizer_kwargs)
    # Get model while forcing cpu to prevent any GPU allocation
    model_config = train_cfg.model
    if not isinstance(model_config, dict):
        msg = f"Expected model_config to be a dict, got {type(model_config)}"
        raise TypeError(msg)

    model_config["init_device"] = "cpu"
    name = model_config.pop("name")
    if not isinstance(name, str):
        msg = f"Expected name to be a string, got {type(name)}"
        raise TypeError(msg)

    if not isinstance(model_config, dict):
        msg = f"Expected model_config to be a dict, got {type(model_config)}"
        raise TypeError(msg)

    init_context = process_init_device(model_config, None, None)
    model = build_composer_model(
        name=name,
        tokenizer=tokenizer,
        init_context=init_context,
        master_weights_dtype=model_config.pop("master_weights_dtype", None),
        cfg=model_config,
    )
    # Force model to cpu
    model.cpu()
    # Get model summary
    if verbose:
        log(DEBUG, model)
    parameters_ndarrays = [
        val.detach().to("cpu").numpy()
        for _, val in get_trainable_params_dict(model).items()
    ]

    if aggregate_momenta:
        # Create two zeroed-out copies of the parameters
        # to simulate the momenta shape
        zeros = [np.zeros_like(param) for param in parameters_ndarrays]
        parameters_ndarrays.extend(zeros)
        parameters_ndarrays.extend(zeros)

    if return_names:
        return parameters_ndarrays, get_list_of_parameters_names(model=model)
    return parameters_ndarrays


def randomize_layers(  # noqa: PLR0913
    parameters: NDArrays,
    dummy_config: DictConfig,
    names: list[str],
    random_layers: list[str],
    cid: int = 0,
    *,
    server_round: int = 1,
    truly_random_init: bool,
) -> None:
    """Randomize specified layers of the model parameters.

    This function randomizes the specified layers of the model parameters based on the
    provided configuration. It can use a truly random initialization if specified.

    Parameters
    ----------
    parameters : NDArrays
        The list of model parameters to be randomized.
    dummy_config : DictConfig
        The configuration object used to create a dummy model for randomization.
    names : list[str]
        The list of parameter names corresponding to the model parameters.
    random_layers : list[str]
        The list of layer names to be randomized.
    truly_random_init : bool
        If True, uses a truly random initialization for the specified layers.
    cid : int, optional
        The client ID used for seeding the random initialization (default is 0).
    server_round : int, optional
        The current server round used for seeding the random initialization
        (default is 1).


    Example
    -------
    >>> from omegaconf import OmegaConf
    >>> parameters = [...]
    >>> dummy_config = OmegaConf.create({...})
    >>> names = ["layer1.weight", "layer1.bias", "layer2.weight"]
    >>> random_layers = ["layer1.weight"]
    >>> randomize_layers(
    >>>     parameters, dummy_config, names, random_layers,
    >>>     truly_random_init=True, cid=1, server_round=2
    >>> )

    """
    new_dummy_config = copy.deepcopy(dummy_config)
    if truly_random_init:
        new_seed = 51550
        for _ in range(server_round):
            new_seed = random.randint(0, 2**32 - 1) ^ cid  # noqa: S311
        new_dummy_config.global_seed = new_seed
        new_dummy_config.seed = new_seed
        reproducibility.seed_all(new_seed)
        log(DEBUG, f"Randomizing layers with seed {new_seed}")

    # Guarantee it is false for pre-trained models
    new_dummy_config.model.pretrained = False
    tmp_dummy_config: BaseConfig = cast(
        "BaseConfig",
        DictConfig(
            {
                "pretrained_model_path": None,
                "llm_config": new_dummy_config,
            },
        ),
    )
    log(DEBUG, f"Creating random model with this config: {tmp_dummy_config}")

    random_parameters = parameters_to_ndarrays(get_initial_parameters(tmp_dummy_config))

    indices = [names.index(key) for key in random_layers]
    for i in indices:
        parameters[i] = random_parameters[i]

    log(DEBUG, f"Randomized layers: {random_layers} with indices: {indices}")


def personalize_layers(
    parameters: NDArrays,
    initial_trainer_parameters: NDArrays,
    personalized_layers: list[str],
    names: list[str],
    unfrozen_names: list[str],
) -> None:
    """Personalize specified layers of the model parameters.

    This function personalizes the specified layers of the model parameters by setting
    them to the initial trainer parameters for the given indices.

    Parameters
    ----------
    parameters : NDArrays
        The list of current model parameters to be personalized.
    initial_trainer_parameters : NDArrays
        The list of initial trainer parameters to use for personalization.
    personalized_layers : list[str]
        The list of layer names to be personalized.
    names : list[str]
        The list of parameter names corresponding to the model parameters.
    unfrozen_names : list[str]
        The list of parameter names corresponding to the unfrozen model parameters.

    Example
    -------
    >>> parameters = [...]
    >>> initial_trainer_parameters = [...]
    >>> personalized_layers = ["layer1.weight", "layer2.bias"]
    >>> names = ["layer1.weight", "layer1.bias", "layer2.weight", "layer2.bias"]
    >>> unfrozen_names = ["layer1.weight", "layer2.bias"]
    >>> personalize_layers(
    >>>     parameters,
    >>>     initial_trainer_parameters,
    >>>     personalized_layers,
    >>>     names,
    >>>     unfrozen_names
    >>> )

    """
    og_indices = [
        i
        for i, name in enumerate(names)
        if any(key in name for key in personalized_layers)
    ]
    indices = [
        i
        for i, name in enumerate(unfrozen_names)
        if any(key in name for key in personalized_layers)
    ]
    log(
        DEBUG,
        f"Personalized: {personalized_layers}, pre-freeze_indices: {og_indices}",
    )
    # Set the server parameters to the initial trainer parameters
    # for the given indices
    for i, j in zip(og_indices, indices, strict=True):
        parameters[i] = initial_trainer_parameters[j]
