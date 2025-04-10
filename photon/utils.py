"""Utility functions for FL and experiment management.

They assure compatibility with the Flower and wandb APIs.
"""

# ruff: noqa: ERA001
import ast
import os
import pickle  # noqa: S403
import threading
import time
import types
from collections import OrderedDict
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from logging import DEBUG, ERROR, WARNING
from pathlib import Path
from queue import Queue
from typing import Any, cast

import numpy as np
import psutil
import ray
import torch
import wandb
from composer import Timestamp, Trainer
from composer.loggers import RemoteUploaderDownloader
from composer.utils import dist
from flwr.common import Config, NDArray, NDArrays, log, parameters_to_ndarrays
from ray._private.internal_api import free as ray_free  # noqa: PLC2701
from torch import device as device_type
from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel

# NOTE: Setting the maximum value according to the documentation
# https://github.com/grpc/grpc/blob/eeae8e635a896bfa420d21e476221af652fd9986/include/grpc/impl/codegen/grpc_types.h#L150
PHOTON_LLM_MAX_MESSAGE_LENGTH = -1


@dataclass
class ClientState:
    """Dataclass for client state."""

    local_steps_cumulative: int
    local_timestamp: dict[str, Any] = field(
        default_factory=lambda: {
            k: v
            for k, v in Timestamp().state_dict().items()
            if is_literal_for_ast(repr(v))
        },
    )
    steps_done: int = 0


class NoOpContextManager:
    """A context manager that does nothing."""

    def __enter__(self) -> None:
        """Do nothing."""
        return

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        """Do nothing."""
        return


@contextmanager
def custom_ray_garbage_collector(
    garbage_queue: Queue[ray.ObjectRef],
    list_of_threads: list[threading.Thread],
    timeout: float = 300,
    *,
    join_at_the_end: bool = True,
) -> Generator[None, Any, None]:
    """Context manager for custom Ray garbage collection.

    This context manager creates a thread that continuously collects and frees
    Ray objects from a given queue. The thread runs in the background and stops
    when the context manager exits.

    Parameters
    ----------
    garbage_queue : Queue[ray.ObjectRef]
        A queue containing Ray object references to be freed.
    list_of_threads : list[threading.Thread]
        A list of threads to append the garbage collection thread to.
    timeout : float, optional
        The time in seconds to wait between garbage collection cycles, by default 300.
    join_at_the_end : bool, optional
        Whether to join the garbage collection thread at the end, by default
        True.

    Yields
    ------
    None
        The context manager yields control back to the caller.

    Notes
    -----
    The function assumes the existence of `ray_free` and `log` functions/utilities,
    as well as the `ERROR` constant for logging purposes.

    Example
    -------
    >>> garbage_queue = Queue()
    >>> with custom_ray_garbage_collector(garbage_queue):
    >>>     # Your code here
    >>>     pass

    """

    def collect_ray_garbage() -> None:
        while True:
            while not garbage_queue.empty():
                try:
                    obj = garbage_queue.get(block=True, timeout=0.1)
                    ray_free([obj])
                except Exception as e:  # noqa: BLE001, PERF203
                    log(ERROR, "Error in custom Ray garbage collection", exc_info=e)
                    break
            log(DEBUG, "Custom Ray garbage collection cycle completed.")
            time.sleep(timeout)

    if not list_of_threads:
        thread = threading.Thread(
            target=collect_ray_garbage,
            name="CustomRayGarbageCollector",
        )
        list_of_threads.append(thread)
        thread.start()

    try:
        yield
    finally:
        if join_at_the_end:
            log(DEBUG, "Closing custom Ray garbage collection.")
            for thread in list_of_threads:
                thread.join()


def parameters_checker(
    current_parameters: NDArrays,
    reference_parameters: NDArrays,
    *,
    is_equal: bool = False,
) -> None:
    """Checker trainer's parameters (in)compatibility with the given parameters.

    Parameters
    ----------
    current_parameters : NDArrays
        The current parameters.
    reference_parameters : NDArrays
        The reference parameters.
    is_equal : bool, optional
        Whether the parameters should be equal or not, by default False.

    Raises
    ------
    ValueError
        If the shapes don't match.
    AssertionError
        If the parameters are not equal

    """
    list_of_conditions = []
    for i, (cur_param, param) in enumerate(
        zip(current_parameters, reference_parameters, strict=True),
    ):
        current_param = cur_param
        # NOTE: Skip check if the size of the `current_param` is 0
        if current_param.size == 0:
            continue
        # Skip ranks > 0 b/c they are not meant to be consistent
        if int(os.getenv("LOCAL_RANK", "-1")) > 0:
            continue
        # Reshape the parameters if the shapes are not equal, which happens for
        # flattened parameters in FSDP)
        if current_param.shape != param.shape:
            try:
                current_param = current_param.reshape(param.shape)
            except Exception as e:  # noqa: BLE001
                log(
                    ERROR,
                    "Error in reshaping parameter, Rank %s, Component %s,"
                    " Trainer shape %s, Param shape %s",
                    int(os.getenv("LOCAL_RANK", "-1")),
                    i,
                    current_param.shape,
                    param.shape,
                    exc_info=e,
                )
                # If the reshaping fails, skip the check assuming split tensor by FSDP
                continue
        # Assert the shape of the parameters are equal
        if current_param.shape != param.shape:
            msg = f"Shapes don't match: {current_param.shape} != {param.shape}"
            raise ValueError(msg)
        # Append the condition to the list of conditions
        list_of_conditions.append(np.array_equal(current_param, param))
    # Assert all the conditions are true if `is_equal` is True
    local_rank = int(os.getenv("LOCAL_RANK", "-1"))
    if is_equal:
        list_of_conditions.append(True)
        if not all(list_of_conditions):
            err_string = (
                f"Parameters on rank {local_rank} are not equal: {list_of_conditions}"
            )
            raise AssertionError(err_string)
    # Assert not all the conditions are true (at least one is False) if `is_equal` is
    # False
    else:
        list_of_conditions.append(False)
        if all(list_of_conditions):
            err_string = (
                f"Parameters rank {local_rank} are not different: {list_of_conditions}"
            )
            raise AssertionError(err_string)


def get_parameters_from_state(_config: Config, trainer: Trainer) -> NDArrays:
    """Implement how to get parameters.

    Parameters
    ----------
    _config : Config
        The configuration.
    trainer : Trainer
        The trainer.

    Returns
    -------
    NDArrays
        The parameters.

    """
    model_parameters_dict = get_trainable_params_dict(trainer.state.model)
    return [val.detach().to("cpu").numpy() for _, val in model_parameters_dict.items()]


def get_trainable_params_dict(
    model: torch.nn.Module,
    *,
    sort_dict: bool = True,
    no_detach_and_clone: bool = False,
) -> dict[str, torch.nn.Parameter] | dict[str, torch.Tensor]:
    """Get the trainable parameters of a model as a dictionary.

    Parameters
    ----------
    model : torch.nn.Module
        The model.
    sort_dict : bool, optional
        Whether to sort the dictionary, by default True.
    no_detach_and_clone : bool, optional
        Whether to detach and clone the parameters, by default False.

    Returns
    -------
    dict[str, torch.nn.Parameter] | dict[str, torch.Tensor]
        The trainable parameters.

    Raises
    ------
    ValueError
        If the model is None.

    """
    params_dict: dict[str, torch.nn.Parameter] | dict[str, torch.Tensor] = {}
    # NOTE: This function is weird because the encapsulation done to support FSDP and
    # DDP is weird. Since they are both likely to change, we MUST maintain this very
    # well and implement as many checkers as we can.
    if hasattr(model, "model") and type(model.model) is FullyShardedDataParallel:
        if model.model is None:  # type: ignore[reportUnnecessaryComparison]
            error_message = "Model is None"
            raise ValueError(error_message)
        inner_model = model.model
        # NOTE: This doesn't work in the case in use_orig_params is True if the FSDP
        # configuration as the tensors returned are flattened breaking some assumptions
        # of the rest of the codebase
        with FullyShardedDataParallel.summon_full_params(
            inner_model,
            recurse=True,
            writeback=False,
            rank0_only=True,
            offload_to_cpu=True,
            with_grads=False,
        ):
            # NOTE: This parameter dict using the above parameters, i.e., (recurse=True,
            # writeback=False, rank0_only=True, offload_to_cpu=True, with_grads=False,),
            # will be complete only on rank 0. The other ranks will have zero-shaped
            # tensors for those layers that are not "living" in there.
            # NOTE: If the FSDP configuration use the original parameters
            # (use_orig_params=true), then the tensors in rank 0 have the correct
            # original shape. In the other ranks they are flattened anyway.
            # NOTE: On ranks > 0 the dictionary won't be empty. It will contain the
            # parameters that are "living" in that rank and will have zero-shaped
            # tensors for the others.
            params_dict = {
                name: param.detach().clone() if not no_detach_and_clone else param
                for name, param in inner_model.named_parameters()
                if param.requires_grad
            }
    else:
        params_dict = {
            name: param.detach().clone() if not no_detach_and_clone else param
            for name, param in model.named_parameters()
            if param.requires_grad
        }
    if sort_dict:
        params_dict = dict(sorted(params_dict.items()))
    dist.barrier()
    return params_dict


def freeze_blocks(
    model: torch.nn.Module,
    frozen_layers: list[str] | None,
    unfrozen_layers: list[str] | None,
) -> None:
    """Freeze the blocks of a model given a list of block indices.

    Parameters
    ----------
    model : torch.nn.Module
        The model.
    frozen_layers : list[str] | None
        The list of frozen layers.
    unfrozen_layers : list[str] | None
        The list of unfrozen layers.

    Raises
    ------
    ValueError
        If the model is None.

    """
    if hasattr(model, "model") and type(model.model) is FullyShardedDataParallel:
        if model.model is None:  # type: ignore[reportUnnecessaryComparison]
            error_message = "Model is None"
            raise ValueError(error_message)
        inner_model = model.model
        # NOTE: This doesn't work in the case in use_orig_params is True if the FSDP
        # configuration as the tensors returned are flattened breaking some assumptions
        # of the rest of the codebase
        with FullyShardedDataParallel.summon_full_params(
            inner_model,
            recurse=True,
            # Writing back is not compatible with rank 0 only
            writeback=True,
            rank0_only=False,
            # Prevent moving to CPU device
            offload_to_cpu=False,
            with_grads=False,
        ):
            # NOTE: !!! THIS REQUIRES INVESTIGATION AS IT DOESN'T WORK AS EXPECTED !!!
            # NOTE: This parameter dict using the above parameters, i.e.,
            # (recurse=True, writeback=True, rank0_only=False, offload_to_cpu=False,
            # with_grads=False,), won't be complete in any rank if the model i sharded.
            # Each rank will have zero-size tensors for those layers that are not
            # "living" in there and the flattened/unflattened complete tensors for those
            # blocks living there.
            # NOTE: If the FSDP configuration use the original parameters
            # (use_orig_params=true), then the tensors in rank 0 have the correct
            # original shape. In the other ranks they are flattened anyway.
            for name, param in inner_model.named_parameters():
                if param.requires_grad and (
                    (frozen_layers is not None and name in frozen_layers)
                    or (unfrozen_layers is not None and name not in unfrozen_layers)
                ):
                    param.requires_grad = False
                    log(DEBUG, "Freezing layer %s", name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad and (
                (frozen_layers is not None and name in frozen_layers)
                or (unfrozen_layers is not None and name not in unfrozen_layers)
            ):
                param.requires_grad = False
                log(DEBUG, "Freezing layer %s", name)
    dist.barrier()


def set_trainer_trainable_params_dict(
    trainer: Trainer,
    parameters_dict: OrderedDict[str, torch.Tensor],
) -> None:
    """Set the trainable parameters of a model.

    Parameters
    ----------
    trainer : Trainer
        The trainer object.
    parameters_dict : OrderedDict[str, torch.Tensor]
        The dictionary of parameters.

    Raises
    ------
    ValueError
        If the shapes don't match.

    """
    # NOTE: This function is weird because the encapsulation done to support FSDP and
    # DDP is weird. Since they are both likely to change, we MUST maintain this very
    # well and implement as many checkers as we can.
    if (
        hasattr(trainer.state.model, "model")
        and type(trainer.state.model.model) is FullyShardedDataParallel
    ):
        # Get the state dict of the model on rank 0 offloading to CPU
        # NOTE: This assumes there's enough RAM on rank 0 to hold the model state dict
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FullyShardedDataParallel.state_dict_type(
            trainer.state.model.model,
            StateDictType.FULL_STATE_DICT,
            save_policy,
        ):
            cpu_state = trainer.state.model.model.state_dict()
            # If the state dict exists (only on rank 0), modify ion place the parameters
            # to those passed as argument
            if cpu_state:
                # Set the parameters only if they require gradients
                for name, param in cpu_state.items():
                    if name not in parameters_dict:
                        error_message = f"""Parameter {name} not found
                        across list of parameters {parameters_dict.keys()}"""
                        raise ValueError(error_message)
                    param_from_dict = parameters_dict[name]
                    # Raise error if the shapes don't match
                    if param.shape != param_from_dict.shape:
                        error_message = f"""Shapes don't match:
                        {param.shape} != {param_from_dict.shape}"""
                        raise ValueError(error_message)
                    current_dtype = param.data.dtype
                    # NOTE: We need to add the prefix "model." to the name of the
                    # parameter to match the state dict
                    cpu_state[name] = param_from_dict.to(
                        device=param.device,
                        dtype=current_dtype,
                    )
            # Broadcast the state dict across all ranks
            # NOTE: This step is necessary as all the ranks must load the same state
            # dict concurrently
            list_of_objects = [cpu_state]
            dist.broadcast_object_list(list_of_objects, src=0)
            # Load the state dict back to the model
            trainer.state.model.model.load_state_dict(list_of_objects[0])
    else:
        for name, param in trainer.state.model.named_parameters():
            # Set the parameters only if they require gradients
            if param.requires_grad:
                lookup_name = name.replace("model.", "").replace("module.", "")
                if lookup_name not in parameters_dict:
                    log(
                        WARNING,
                        "Parameter %s not found in the list of parameters"
                        " and won't be set",
                        name,
                    )
                else:
                    param_from_dict = parameters_dict[lookup_name]
                    # Raise error if the shapes don't match
                    if param.shape != param_from_dict.shape:
                        error_message = f"""Shapes don't match:
                        {param.shape} != {param_from_dict.shape}"""
                        raise ValueError(error_message)
                    current_dtype = param.data.dtype
                    param.data = param_from_dict.to(
                        device=param.device,
                        dtype=current_dtype,
                    )
    dist.barrier()


def set_trainer_params_from_ndarrays(
    parameters: NDArrays,
    trainer: Trainer,
    key_to_filter: str = "transformer",
    *,
    filter_keys: bool = True,
) -> None:
    """Set the parameters of a trainer from a list of NDArrays.

    This function attempts to set the parameters of the trainer's model using
    the provided NDArrays. It first tries to set the parameters assuming they
    are ordered. If this fails due to shape mismatches, it retries with the
    parameters unordered.

    Args:
    ----
    parameters (NDArrays): The list of NDArrays representing the model parameters.
    trainer (Trainer): The trainer object whose model parameters are to be set.
    key_to_filter (str): The key to filter the parameters.
    filter_keys (bool): Whether to filter the keys.

    Raises:
    ------
    ValueError: If setting the parameters fails due to shape mismatches or other
        issues.

    """
    # Get the unordered and ordered list of parameter names
    parameters_names = get_list_of_parameters_names(
        trainer.state.model,
        sort_dict=False,
    )
    ordered_parameters_names = sorted(parameters_names)
    # Try to set the parameters a s if they are ordered
    try:
        parameters_dict = construct_parameters_dict(
            ordered_parameters_names,
            parameters,
            filter_keys=filter_keys,
            key_to_filter=key_to_filter,
        )
        set_trainer_trainable_params_dict(trainer, parameters_dict)
    except ValueError as e:
        if "Shapes don't match" in str(e):
            log(
                ERROR,
                "Error trying to set the parameters as ordered, trying unordered",
                exc_info=e,
                stack_info=True,
            )
            # If the ordered parameters failed, try to set the parameters as unordered
            parameters_dict = construct_parameters_dict(
                parameters_names,
                parameters,
                filter_keys=filter_keys,
                key_to_filter=key_to_filter,
            )
            set_trainer_trainable_params_dict(trainer, parameters_dict)
        else:
            raise


def get_wte_parameters_from_trainer(trainer: Trainer) -> NDArray:
    """Get the parameters of the WTE layer of a model from a trainer.

    Parameters
    ----------
    trainer : Trainer
        The trainer object.

    Returns
    -------
    NDArray
        The parameters of the WTE layer.

    Raises
    ------
    ValueError
        If there are no WTE parameters or if the WTE parameters are not unique.

    """
    # Get the parameter names of the model
    model_parameter_names = get_list_of_parameters_names(trainer.state.model)
    # Get the WTE parameters
    wte_parameters_dict = {
        name: param
        for name, param in zip(
            model_parameter_names,
            get_parameters_from_state({}, trainer),
            strict=False,
        )
        if "wte" in name
    }
    # Return the WTE parameters
    wte_parameters = list(wte_parameters_dict.values())
    if len(wte_parameters) <= 0:
        msg = "There are no WTE parameters"
        raise ValueError(msg)
    if len(wte_parameters) != 1:
        msg = "WTE parameters are not unique"
        raise ValueError(msg)
    return wte_parameters[0]


def set_wte_parameters_to_trainer(trainer: Trainer, wte_parameters: NDArray) -> None:
    """Set the parameters of the WTE layer of a model to a trainer."""
    # Get the parameter names of the model
    model_parameter_names = get_list_of_parameters_names(trainer.state.model)
    # Get the WTE parameters
    model_parameters: list[NDArray] = [
        param if "wte" not in name else wte_parameters
        for name, param in zip(
            model_parameter_names,
            get_parameters_from_state({}, trainer),
            strict=False,
        )
    ]
    # Set the WTE parameters
    set_trainer_params_from_ndarrays(model_parameters, trainer)


def get_list_of_parameters_names(
    model: torch.nn.Module,
    *,
    sort_dict: bool = True,
) -> list[str]:
    """Return the list of parameters names.

    Parameters
    ----------
    model : torch.nn.Module
        The model.
    sort_dict : bool, optional
        Whether to sort the dictionary, by default True.

    Returns
    -------
    list[str]
        The list of parameters names.

    """
    # Get named parameters dictionary of the model
    params_dict = {
        name: param for name, param in model.named_parameters() if param.requires_grad
    }
    # Trim some annoying prefixes
    params_dict = {k.replace("model.", ""): v for k, v in params_dict.items()}
    params_dict = {k.replace("module.", ""): v for k, v in params_dict.items()}
    params_dict = {k.replace("_fsdp_wrapped_", ""): v for k, v in params_dict.items()}
    params_dict = {
        k.replace("_checkpoint_wrapped_", ""): v for k, v in params_dict.items()
    }
    # Sort the dictionary if requested
    if sort_dict:
        params_dict = dict(sorted(params_dict.items()))
    # Return the list of parameter names
    return list(params_dict.keys())


def construct_parameters_dict(
    parameters_names: list[str],
    parameters: NDArrays,
    *,
    filter_keys: bool = True,
    key_to_filter: str = "transformer",
) -> OrderedDict[str, torch.Tensor]:
    """Construct a dictionary of parameters.

    Parameters
    ----------
    parameters_names : list[str]
        The list of parameters names.
    parameters : NDArrays
        The parameters.
    filter_keys : bool, optional
        Whether to filter the keys, by default True.
    key_to_filter : str, optional
        The key to filter, by default "transformer".

    Returns
    -------
    OrderedDict[str, torch.Tensor]
        The dictionary of parameters.

    """
    # Remove any non-transformer parameters from parameters_names
    if filter_keys:
        parameters_names = [name for name in parameters_names if key_to_filter in name]
    zipped_lists = zip(parameters_names, parameters, strict=True)
    return OrderedDict({k: torch.as_tensor(v) for k, v in zipped_lists})


def download_file_from_s3(
    remote_up_down: RemoteUploaderDownloader,
    remote_file_name: str,
    local_file_name: Path | str,
) -> None:
    """Download a file from S3."""
    remote_up_down._check_workers()  # noqa: SLF001
    remote_up_down.download_file(
        remote_file_name=remote_file_name,
        destination=str(local_file_name),
        overwrite=True,
    )


def upload_file_to_s3(
    remote_up_down: RemoteUploaderDownloader,
    remote_file_name: str,
    local_file_name: Path,
) -> None:
    """Download a file from S3."""
    remote_up_down._check_workers()  # noqa: SLF001
    remote_up_down.upload_file(
        state=None,
        remote_file_name=remote_file_name,
        file_path=local_file_name,
        overwrite=True,
    )


def load_model_parameters_from_file(file_path: Path) -> NDArrays:
    """Load model parameters from a file.

    Parameters
    ----------
    file_path : Path
        The file path.

    Returns
    -------
    NDArrays
        The model parameters.

    Raises
    ------
    ValueError
        If the file format is not supported.

    """
    if file_path.suffix in {".npz", ".npzc"}:
        with file_path.open("rb") as file:
            data = np.load(file)
            return [data[key] for key in data.files]
    elif file_path.suffix == ".bin":
        with file_path.open("rb") as file:
            return parameters_to_ndarrays(pickle.load(file))  # noqa: S301
    else:
        msg = f"Unsupported file format: {file_path.suffix}"
        raise ValueError(msg)


def dump_model_parameters_to_file(file_path: Path, model_parameters: NDArrays) -> None:
    """Load model parameters from a file.

    Parameters
    ----------
    file_path : Path
        The file path.
    model_parameters : NDArrays
        The model parameters.

    Raises
    ------
    ValueError
        If the file format is not supported.

    """
    # NOTE: Very slow for big models b/c compression. Good benchmark available here: https://stackoverflow.com/questions/30329726/fastest-save-and-load-options-for-a-numpy-array
    if file_path.suffix == ".npzc":
        with file_path.open("wb") as file:
            np.savez_compressed(file, *model_parameters)
    elif file_path.suffix == ".bin":
        with file_path.open("wb") as file:
            pickle.dump(model_parameters, file)
    elif file_path.suffix == ".npz":
        with file_path.open("wb") as file:
            np.savez(file, *model_parameters)
    else:
        msg = f"Unsupported file format: {file_path.suffix}"
        raise ValueError(msg)


def set_parameters(
    net: torch.nn.Module,
    parameters: NDArrays,
    device: str = "cpu",
) -> None:
    """Implement generic `set_parameters` for Flower Client."""
    net.eval()
    model_parameters_dict = get_trainable_params_dict(net)
    params_dict = zip(model_parameters_dict.keys(), parameters, strict=True)
    state_dict = OrderedDict(
        {k: torch.as_tensor(v, device=device) for k, v in params_dict},
    )
    net.load_state_dict(state_dict=state_dict, strict=False)
    del state_dict


def wandb_init(
    wandb_enabled: bool,  # noqa: FBT001
    *args: dict,
    **kwargs: dict,
) -> NoOpContextManager | Any | None:  # noqa: ANN401
    """Initialize wandb if enabled.

    Parameters
    ----------
    wandb_enabled : bool
        Whether wandb is enabled.
    args : dict
        The arguments.
    kwargs : dict
        The keyword arguments.

    Returns
    -------
    NoOpContextManager | Any | None
        The wandb context

    Raises
    ------
    ValueError
        If the name is not a string.

    """
    if wandb_enabled:
        # Add server suffix to the name of the run
        name = kwargs.pop("name", "")
        if type(name) is not str:
            error = f"Name must be a string, not {type(name)}"
            raise ValueError(error)
        name += "_server"
        return wandb.init(*args, **kwargs, name=name)  # type: ignore[arg-type,misc]

    return NoOpContextManager()


def chunks_idx(
    list_of_stuff: Sequence,
    n_chunks: int,
) -> Generator[tuple[int, int], Any, None]:
    """Split a list in n_chunks of equal length.

    Parameters
    ----------
    list_of_stuff : Sequence
        The list to split.
    n_chunks : int
        The number of chunks to split the list into.

    Yields
    ------
    Generator[tuple[int, int], Any, None]
        A generator yielding the start and end indices of the chunks

    """
    d, r = divmod(len(list_of_stuff), n_chunks)
    for i in range(n_chunks):
        si = (d + 1) * (min(r, i)) + d * (0 if i < r else i - r)
        yield si, si + (d + 1 if i < r else d)


def sum_of_squares(arrays: NDArrays) -> float:
    """Compute the sum of squares of a list of arrays.

    Parameters
    ----------
    arrays : NDArrays
        List of arrays to compute the sum of squares of.

    Returns
    -------
    float
        The sum of squares of the list of arrays.

    """
    return sum(np.sum(np.square(arr)) for arr in arrays)


def l2_norm(arrays: NDArrays) -> float:
    """Compute the L2 norm of a list of arrays.

    Parameters
    ----------
    arrays : NDArrays
        List of arrays to compute the L2 norm of.

    Returns
    -------
    float
        The L2 norm of the list of arrays.

    """
    return float(np.sqrt(sum_of_squares(arrays)))


def l2_norm_of_momenta(
    state: dict[str, dict[str, torch.Tensor]],
) -> tuple[float, float]:
    """Compute the L2 norm of optimizer momenta.

    Parameters
    ----------
    state : dict[str, dict[str, torch.Tensor]]
        The state of the optimizer.

    Returns
    -------
    float
        The L2 norm of the list of arrays.

    """
    first_moment_sums_of_squares = 0.0
    second_moment_sums_of_squares = 0.0

    for param_state in state.values():
        first_moment_sums_of_squares += torch.sum(
            param_state["exp_avg"].detach().to("cpu") ** 2,
        ).item()

        second_moment_sums_of_squares += torch.sum(
            param_state["exp_avg_sq"].detach().to("cpu") ** 2,
        ).item()

    return float(np.sqrt(first_moment_sums_of_squares)), float(
        np.sqrt(second_moment_sums_of_squares),
    )


def get_device() -> device_type:
    """Determine which device to use for PyTorch.

    Returns
    -------
        str: device for PyTorch

    """
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    return cast("device_type", device)


def get_n_cuda_devices() -> int:
    """Get the number of CUDA devices available.

    Returns
    -------
        int: number of CUDA devices available

    """
    if "cuda" in str(get_device()):
        return torch.cuda.device_count()
    return 0


def get_n_cpu_cores() -> int | None:
    """Get the number of CPU cores available.

    Returns
    -------
        int | None: number of CPU cores available

    """
    try:
        cpus = len(psutil.Process().cpu_affinity())  # type: ignore[reportArgumentType]
    except AttributeError:
        cpus = psutil.cpu_count()
    return cpus


def create_remote_up_down(  # noqa: PLR0913
    bucket_name: str,
    prefix: str,
    run_uuid: str | None,
    num_attempts: int,
    client_config: dict[str, Any],
    *,
    num_concurrent_uploads: int = 1,
    upload_staging_folder: str | None = None,  # Don't touch, it's /tmp by default
    use_procs: bool = True,
) -> RemoteUploaderDownloader:
    """Create the remote uploader/downloader.

    Parameters
    ----------
    bucket_name : str
        The name of the bucket.
    prefix : str
        The prefix of the bucket.
    run_uuid : str | None
        The UUID of the run.
    num_attempts : int
        The number of attempts.
    client_config : dict[str, Any]
        The configuration of the client.
    num_concurrent_uploads : int, optional
        The number of concurrent uploads, by default 1.
    upload_staging_folder : str | None, optional
        The upload staging folder, dont't touch, by default None.
    use_procs : bool, optional
        Whether to use processes, by default True. Don't touch.

    Returns
    -------
    RemoteUploaderDownloader
        The remote uploader/downloader.

    """
    bucket_uri = f"s3://{bucket_name}"
    remote_up_down = RemoteUploaderDownloader(
        bucket_uri=bucket_uri,
        backend_kwargs={
            "bucket": bucket_name,
            "prefix": prefix,  # Don't touch
            "region_name": None,  # Not necessary
            "endpoint_url": None,  # Will be read from env var
            "aws_access_key_id": None,  # Will be read from config file
            "aws_secret_access_key": None,  # Will be read from config file
            "aws_session_token": None,  # Will be automatically generated
            "client_config": client_config,  # And using defaults
            "transfer_config": None,  # Using defaults
        },
        file_path_format_string="{remote_file_name}",  # Don't touch
        num_concurrent_uploads=num_concurrent_uploads,
        upload_staging_folder=upload_staging_folder,  # Don't touch, default: /tmp
        use_procs=use_procs,  # Don't touch
        num_attempts=num_attempts,
    )
    remote_up_down.init(run_name=run_uuid)  # Don't touch
    return remote_up_down


def merge_freq_dicts(
    a: dict[int, int],
    b: dict[int, int],
) -> dict[int, int]:
    """Merge two frequency dictionaries.

    Parameters
    ----------
    a : dict[int, int]
        The first frequency dictionary.
    b : dict[int, int]
        The second frequency dictionary.

    Returns
    -------
    dict[int, int]
        The merged frequency dictionary.

    """
    return a | {k: (a.get(k, 0) + v) for k, v in b.items()}


def get_unigram_probabilities_tensor(
    stream_freq_dict: dict[int, int],
) -> torch.Tensor:
    """Get the unigram probabilities tensor.

    Parameters
    ----------
    stream_freq_dict : dict[int, int]
        The frequency dictionary.

    Returns
    -------
    torch.Tensor
        The unigram probabilities tensor.

    """
    total_tokens = float(sum(v for v in stream_freq_dict.values()))
    probabilities = {k: v / total_tokens for k, v in stream_freq_dict.items()}
    # Get the max token id
    max_token_id = max(stream_freq_dict.keys())
    # Convert to dense tensor
    probabilities_tensor = torch.zeros(max_token_id + 1)
    for k, v in probabilities.items():
        probabilities_tensor[k] = v
    return probabilities_tensor


def is_literal_for_ast(s: str) -> bool:
    """Check if the given str can be evaluate as a literal.

    Parameters
    ----------
    s : str
        The string to check.

    Returns
    -------
    bool
        Whether the string can be evaluated as a literal

    """
    try:
        ast.literal_eval(s)
    except ValueError:
        return False
    return True
