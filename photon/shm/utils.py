"""The module contains utility functions and classes for managing shared memories.

Functions
---------
    get_ndarrays_size_and_bounds(
        ndarrays: NDArrays
    ) -> tuple[int, list[tuple[int, int]]]
        Calculate the total size and memory bounds of a list of NDArrays.

    set_dict_configsrecord_shm(
        config: TypedDict[str, ConfigsRecord], shm: SharedMemory
    ) -> None
        Store a dictionary of ConfigsRecord in shared memory.

    old_get_parameters_shm(
        parameters: NDArrays, name: str, create: bool = False
    ) -> tuple[NDArrays, SharedMemory]
        Retrieve or create shared memory for model parameters.

    get_num_samples_shm(
        name: str, create: bool = False
    ) -> tuple[np.ndarray, SharedMemory]
        Retrieve or create shared memory for the number of samples.

    set_num_samples_shm(old_num_samples_sh: np.ndarray, new_num_samples: int) -> None
        Set the number of samples in shared memory.

    get_eval_loss_shm(
        name: str, create: bool = False
    ) -> tuple[np.ndarray, SharedMemory]
        Retrieve or create shared memory for evaluation loss.

    close_all_shms(process_uuid: str) -> None
        Close and unlink all shared memory segments associated with a process.

    remove_shm_from_resource_tracker() -> None
        Monkey-patch multiprocessing.resource_tracker so SharedMemory won't be tracked.

    get_dict_configsrecord_shm(
        name: str, config: TypedDict[str, ConfigsRecord], create: bool = False
    ) -> tuple[TypedDict[str, ConfigsRecord], SharedMemory]
        Retrieve or create shared memory for a dictionary of ConfigsRecord.

    get_config_shm(
        config: Config, name: str, create: bool = False
    ) -> tuple[Config, SharedMemory]
        Retrieve or create shared memory for a Config object.

    is_shm_existing(name: str) -> bool
        Check if a shared memory segment exists.

    set_config_shm(config: Config, shm: SharedMemory) -> None
        Store a Config object in shared memory.

    get_parameters_shm(
        parameters_metadata: ModelParametersMetadata, name: str, create: bool = False
    ) -> tuple[NDArrays, SharedMemory]
        Retrieve or create shared memory for model parameters.

    set_parameters_shm(old_parameters_sh: NDArrays, new_parameters: NDArrays) -> None
        Update model parameters in shared memory.

    set_eval_loss_shm(old_eval_loss_sh: np.ndarray, new_eval_loss: float) -> None
        Update evaluation loss in shared memory.

Classes
-------
    ModelParametersMetadata
        Metadata for model parameters.

Imports
-------
    - collections.abc
    - typing
    - dataclasses
    - pickle
    - logging
    - multiprocessing.shared_memory
    - numpy
    - flwr.common.logger
    - flwr.common
    - flwr.common.recordset_compat
    - flwr.common.record.typeddict
"""

import concurrent.futures
import pickle  # noqa: S403
from collections.abc import Sequence
from dataclasses import dataclass
from logging import ERROR
from multiprocessing import resource_tracker as res_track
from multiprocessing.shared_memory import SharedMemory
from typing import SupportsIndex

import numpy as np
from flwr.common import Config, NDArrays
from flwr.common.logger import log
from flwr.common.record.typeddict import TypedDict
from flwr.common.recordset_compat import ConfigsRecord

ShapeLike = SupportsIndex | Sequence[SupportsIndex]


def get_ndarrays_size_and_bounds(
    ndarrays: NDArrays,
) -> tuple[int, list[tuple[int, int]]]:
    """Calculate the total size and memory bounds of a list of NDArrays.

    This function computes the total number of bytes occupied by the given NDArrays
    and returns the memory bounds for each array.

    Parameters
    ----------
    ndarrays : NDArrays
        A list of NDArrays for which the size and bounds are to be calculated.

    Returns
    -------
        tuple[int, list[tuple[int, int]]]
            A tuple containing:
            - The total size in bytes of all the NDArrays.
            - A list of tuples, where each tuple represents the start and end byte
              positions of each ndarray in the concatenated memory space.

    Example
    -------
    >>> ndarrays = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    >>> total_size, bounds = get_ndarrays_size_and_bounds(ndarrays)
    >>> print(total_size)
    >>> print(bounds)

    """
    nbytes = [val.nbytes for val in ndarrays]
    array_bounds = [(sum(nbytes[:i]), sum(nbytes[: i + 1])) for i in range(len(nbytes))]
    return sum(nbytes), array_bounds


@dataclass
class ModelParametersMetadata:
    """Metadata for model parameters.

    This class stores metadata about model parameters, including their total size in
    bytes, memory bounds, shapes, and data types.

    Attributes
    ----------
        total_num_bytes : int
            The total number of bytes occupied by the model parameters.
        array_bounds : list[tuple[int, int]]
            A list of tuples representing the start and end byte positions of each
            ndarray
            in the concatenated memory space.
        shapes : list[ShapeLike]
            A list of shapes of the model parameters.
        dtypes : list[np.dtype]
            A list of data types of the model parameters.

    """

    total_num_bytes: int
    array_bounds: list[tuple[int, int]]
    shapes: list[ShapeLike]
    dtypes: list[np.dtype]

    @staticmethod
    def from_ndarrays(parameters: NDArrays) -> "ModelParametersMetadata":
        """Create ModelParametersMetadata from a list of NDArrays.

        Parameters
        ----------
        parameters : NDArrays
            A list of NDArrays representing the model parameters.

        Returns
        -------
            ModelParametersMetadata
                An instance of ModelParametersMetadata containing the metadata of the
                given NDArrays.

        """
        total_num_bytes, array_bounds = get_ndarrays_size_and_bounds(parameters)
        shapes = [x.shape for x in parameters]
        dtypes = [x.dtype for x in parameters]
        return ModelParametersMetadata(
            total_num_bytes=total_num_bytes,
            array_bounds=array_bounds,
            shapes=shapes,  # type: ignore[arg-type]
            dtypes=dtypes,
        )

    @staticmethod
    def to_str(parameters_metadata: "ModelParametersMetadata") -> str:
        """Convert ModelParametersMetadata to a string representation.

        Parameters
        ----------
        parameters_metadata : ModelParametersMetadata
            An instance of ModelParametersMetadata to be converted to a string.

        Returns
        -------
            str
                A string representation of the ModelParametersMetadata instance.

        """
        return (
            f"ModelParametersMetadata("
            f"total_num_bytes={parameters_metadata.total_num_bytes}, "
            f"array_bounds={parameters_metadata.array_bounds}, "
            f"shapes={parameters_metadata.shapes}, "
            f"dtypes={parameters_metadata.dtypes})"
        )

    @staticmethod
    def from_str(parameters_metadata_str: str) -> "ModelParametersMetadata":
        """Create ModelParametersMetadata from a string representation.

        Parameters
        ----------
        parameters_metadata_str : str
            A string representation of ModelParametersMetadata.

        Returns
        -------
            ModelParametersMetadata
                An instance of ModelParametersMetadata created from the string
                representation.

        """
        # Remove the class name and parentheses
        metadata_str = parameters_metadata_str[len("ModelParametersMetadata(") : -1]

        # Split the string into key-value pairs
        kv_pairs = metadata_str.split(", ")

        # Create a dictionary from the key-value pairs
        metadata_dict = {}
        for kv in kv_pairs:
            key, value = kv.split("=")
            metadata_dict[key] = eval(value)  # noqa: S307

        return ModelParametersMetadata(
            total_num_bytes=metadata_dict["total_num_bytes"],
            array_bounds=metadata_dict["array_bounds"],
            shapes=metadata_dict["shapes"],
            dtypes=metadata_dict["dtypes"],
        )


def set_dict_configsrecord_shm(
    config: TypedDict[str, ConfigsRecord],
    shm: SharedMemory,
) -> None:
    """Store a dictionary of ConfigsRecord in shared memory.

    This function serializes a dictionary of ConfigsRecord using pickle and stores it
    in the provided shared memory buffer.

    Parameters
    ----------
    config : TypedDict[str, ConfigsRecord]
        A dictionary containing ConfigsRecord objects.
    shm : SharedMemory
        The shared memory object where the serialized dictionary will be stored.

    """
    config_bytes = pickle.dumps(config, protocol=pickle.HIGHEST_PROTOCOL)
    shm.buf[:] = config_bytes


def get_num_samples_shm(
    name: str,
    *,
    create: bool = False,
) -> tuple[np.ndarray, SharedMemory]:
    """Retrieve or create shared memory for the number of samples.

    This function either creates a new shared memory segment or retrieves an existing
    one for storing the number of samples. It returns the number of samples as an
    ndarray and the shared memory object.

    Parameters
    ----------
    name : str
        The name of the shared memory segment.
    create : bool, optional
        If True, a new shared memory segment is created. If False, an existing
        shared memory segment is retrieved. Default is False.

    Returns
    -------
        tuple[np.ndarray, SharedMemory]
            A tuple containing:
            - The number of samples as an ndarray.
            - The shared memory object.

    Raises
    ------
        FileNotFoundError
            If the shared memory segment does not exist and create is False.

    """
    if create:
        shm = SharedMemory(create=True, size=np.dtype(np.int64).itemsize, name=name)
        shm.buf[:] = b"\0" * shm.size
    else:
        shm = SharedMemory(name=name)
    num_samples_sh: np.ndarray = np.ndarray((1,), dtype=np.int64, buffer=shm.buf)
    return num_samples_sh, shm


def set_num_samples_shm(
    old_num_samples_sh: np.ndarray,
    new_num_samples: int,
) -> None:
    """Set the number of samples in shared memory.

    This function updates the number of samples stored in the shared memory segment.

    Parameters
    ----------
    old_num_samples_sh : np.ndarray
        The ndarray representing the current number of samples in shared memory.
    new_num_samples : int
        The new number of samples to be set.

    """
    old_num_samples_sh[0] = new_num_samples


def get_eval_loss_shm(
    name: str,
    *,
    create: bool = False,
) -> tuple[np.ndarray, SharedMemory]:
    """Retrieve or create shared memory for evaluation loss.

    This function either creates a new shared memory segment or retrieves an existing
    one for storing the evaluation loss. It returns the evaluation loss as an ndarray
    and the shared memory object.

    Parameters
    ----------
    name : str
        The name of the shared memory segment.
    create : bool, optional
        If True, a new shared memory segment is created. If False, an existing
        shared memory segment is retrieved. Default is False.

    Returns
    -------
        tuple[np.ndarray, SharedMemory]
            A tuple containing:
            - The evaluation loss as an ndarray.
            - The shared memory object.

    Raises
    ------
        FileNotFoundError
            If the shared memory segment does not exist and create is False.

    """
    if create:
        shm = SharedMemory(create=True, size=np.dtype(np.float64).itemsize, name=name)
        shm.buf[:] = b"\0" * shm.size
    else:
        shm = SharedMemory(name=name)
    eval_loss_sh: np.ndarray = np.ndarray((1,), dtype=np.float64, buffer=shm.buf)
    return eval_loss_sh, shm


def close_all_shms(process_uuid: str) -> None:
    """Close and unlink all shared memory segments associated with a process.

    This function attempts to close and unlink all shared memory segments associated
    with the given process UUID.

    Parameters
    ----------
    process_uuid : str
        The unique identifier for the process.

    """
    shms_names = [
        process_uuid + "",
    ]
    for shm_name in shms_names:
        try:
            shm = SharedMemory(name=shm_name)
            shm.close()
            shm.unlink()
        except Exception as e:  # noqa: BLE001, PERF203
            if "[Errno 2] No such file or directory" in str(e):
                continue
            log(
                ERROR,
                "Removing Shared Memory %s failed because of %s",
                shm_name,
                e,
            )


def remove_shm_from_resource_tracker() -> None:
    """Monkey-patch multiprocessing.resource_tracker so SharedMemory won't be tracked.

    This function modifies the multiprocessing.resource_tracker to prevent it from
    tracking shared memory segments. This is a workaround for a known issue with
    shared memory tracking in Python.

    More details at: https://bugs.python.org/issue38119

    """

    def fix_register(name: str, rtype: str) -> None:
        if rtype == "shared_memory":
            return None
        return res_track._resource_tracker.register(name, rtype)  # noqa: SLF001

    res_track.register = fix_register  # type: ignore[assignment]

    def fix_unregister(name: str, rtype: str) -> None:
        if rtype == "shared_memory":
            return None
        return res_track._resource_tracker.unregister(name, rtype)  # noqa: SLF001

    res_track.unregister = fix_unregister  # type: ignore[assignment]

    if "shared_memory" in res_track._CLEANUP_FUNCS:  # type: ignore[attr-defined]  # noqa: SLF001
        del res_track._CLEANUP_FUNCS["shared_memory"]  # type: ignore[attr-defined]  # noqa: SLF001


def get_dict_configsrecord_shm(
    name: str,
    config: TypedDict[str, ConfigsRecord],
    *,
    create: bool = False,
) -> tuple[TypedDict[str, ConfigsRecord], SharedMemory]:
    """Retrieve or create shared memory for a dictionary of ConfigsRecord.

    This function either creates a new shared memory segment or retrieves an existing
    one for storing a dictionary of ConfigsRecord. It returns the dictionary and the
    shared memory object.

    Parameters
    ----------
    name : str
        The name of the shared memory segment.
    config : TypedDict[str, ConfigsRecord]
        A dictionary containing ConfigsRecord objects.
    create : bool, optional
        If True, a new shared memory segment is created. If False, an existing
        shared memory segment is retrieved. Default is False.

    Returns
    -------
        tuple[TypedDict[str, ConfigsRecord], SharedMemory]
            A tuple containing:
            - The dictionary of ConfigsRecord.
            - The shared memory object.

    Raises
    ------
        FileNotFoundError
            If the shared memory segment does not exist and create is False.

    """
    if create:
        config_bytes = pickle.dumps(config, protocol=pickle.HIGHEST_PROTOCOL)
        shm = SharedMemory(create=True, size=len(config_bytes), name=name)
        config_sh = config
    else:
        shm = SharedMemory(name=name)
        config_sh = pickle.loads(shm.buf)  # noqa: S301
    return config_sh, shm


def get_config_shm(
    config: Config,
    name: str,
    *,
    create: bool = False,
) -> tuple[Config, SharedMemory]:
    """Retrieve or create shared memory for a Config object.

    This function either creates a new shared memory segment or retrieves an existing
    one for storing a Config object. It returns the Config object and the shared memory
    object.

    Parameters
    ----------
    config : Config
        A Config object to be stored in shared memory.
    name : str
        The name of the shared memory segment.
    create : bool, optional
        If True, a new shared memory segment is created. If False, an existing
        shared memory segment is retrieved. Default is False.

    Returns
    -------
        tuple[Config, SharedMemory]
            A tuple containing:
            - The Config object.
            - The shared memory object.

    Raises
    ------
    ValueError
        If create is True and the config object is empty.

    """
    if create and config == {}:
        msg = "Cannot create config without config object."
        raise ValueError(msg)
    if create:
        config_bytes = pickle.dumps(config, protocol=pickle.HIGHEST_PROTOCOL)
        shm = SharedMemory(create=True, size=len(config_bytes), name=name)
        config_sh = config
    else:
        shm = SharedMemory(name=name)
        config_sh = pickle.loads(shm.buf)  # noqa: S301
    return config_sh, shm


def is_shm_existing(name: str) -> bool:
    """Check if a shared memory segment exists.

    This function checks if a shared memory segment with the given name exists.

    Parameters
    ----------
    name : str
        The name of the shared memory segment.

    Returns
    -------
    bool
        True if the shared memory segment exists, False otherwise.

    """
    try:
        _shm = SharedMemory(name=name, create=False)
    except FileNotFoundError:
        return False
    return True


def set_config_shm(
    config: Config,
    shm: SharedMemory,
) -> None:
    """Store a Config object in shared memory.

    This function serializes a Config object using pickle and stores it in the provided
    shared memory buffer.

    Parameters
    ----------
    config : Config
        The Config object to be stored in shared memory.
    shm : SharedMemory
        The shared memory object where the serialized Config will be stored.

    """
    config_bytes = pickle.dumps(config, protocol=pickle.HIGHEST_PROTOCOL)
    shm.buf[:] = config_bytes


def get_parameters_shm(
    parameters_metadata: ModelParametersMetadata,
    name: str,
    *,
    create: bool = False,
) -> tuple[NDArrays, SharedMemory]:
    """Retrieve or create shared memory for model parameters.

    This function either creates a new shared memory segment or retrieves an existing
    one for storing model parameters. It returns the parameters as NDArrays and the
    shared memory object.

    Parameters
    ----------
    parameters_metadata : ModelParametersMetadata
        Metadata for the model parameters, including their shapes, data types, and
        memory bounds.
    name : str
        The name of the shared memory segment.
    create : bool, optional
        If True, a new shared memory segment is created. If False, an existing
        shared memory segment is retrieved. Default is False.

    Returns
    -------
        tuple[NDArrays, SharedMemory]
            A tuple containing:
            - The model parameters as NDArrays.
            - The shared memory object.

    Raises
    ------
        FileNotFoundError
            If the shared memory segment does not exist and create is False.

    """
    if create:
        shm = SharedMemory(
            create=True,
            size=parameters_metadata.total_num_bytes,
            name=name,
        )
        shm.buf[:] = b"\0" * shm.size
    else:
        shm = SharedMemory(name=name)
    params_sh: NDArrays = [
        np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf[bounds[0] : bounds[1]])
        for shape, dtype, bounds in zip(
            parameters_metadata.shapes,
            parameters_metadata.dtypes,
            parameters_metadata.array_bounds,
            strict=False,
        )
    ]
    return params_sh, shm


def set_parameters_shm(
    old_parameters_sh: NDArrays,
    new_parameters: NDArrays,
) -> None:
    """Update model parameters in shared memory.

    This function updates the model parameters stored in the shared memory segment with
    new parameters.

    Parameters
    ----------
    old_parameters_sh : NDArrays
        The current model parameters stored in shared memory.
    new_parameters : NDArrays
        The new model parameters to be set in shared memory.

    """

    def update_parameter(i: int) -> None:
        if len(new_parameters[i].shape) == 0:
            old_parameters_sh[i] = new_parameters[i]
        else:
            old_parameters_sh[i][:] = new_parameters[i][:]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(update_parameter, range(len(new_parameters)))


def set_eval_loss_shm(
    old_eval_loss_sh: np.ndarray,
    new_eval_loss: float,
) -> None:
    """Update evaluation loss in shared memory.

    This function updates the evaluation loss stored in the shared memory segment with
    a new evaluation loss value.

    Parameters
    ----------
    old_eval_loss_sh : np.ndarray
        The current evaluation loss stored in shared memory.
    new_eval_loss : float
        The new evaluation loss value to be set in shared memory.

    """
    old_eval_loss_sh[0] = new_eval_loss
