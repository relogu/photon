"""The module contains utility functions and classes for managing Workers.

Functions
---------

    get_env_patcher(
        run_uuid: str, rank: str, master_port: str,
    ) -> Generator[None, Any, None]
        Context manager to patch environment variables for a distributed training setup.

Classes
-------
    WorkerResultMessage
        Data class for the result message sent by the worker.

Imports
-------
    - dataclasses
    - logging
    - os
    - collections.abc
    - typing
    - contextlib
    - torch
    - torch.distributed
    - streaming
    - composer.cli.launcher
    - composer.utils.misc
"""

import os
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from logging import DEBUG, ERROR
from typing import Any, SupportsIndex

import streaming
import torch
import torch.distributed as dist
from composer.cli.launcher import _patch_env  # noqa: PLC2701
from flwr.common.logger import log

ShapeLike = SupportsIndex | Sequence[SupportsIndex]


@dataclass
class WorkerResultMessage:
    """Data class for the result message sent by the worker."""

    n_samples: int
    delta: float
    device: str


@contextmanager
def get_env_patcher(
    run_uuid: str,
    rank: str,
    master_port: str,
) -> Generator[None, Any, None]:
    """Context manager to patch environment variables for a distributed training setup.

    This context manager sets up the necessary environment variables for a distributed
    training setup. It ensures that the PyTorch Distributed process group is properly
    initialized and recycled in case of a following task aims to use the same resources.

    Parameters
    ----------
    run_uuid : str
        The unique identifier for the run.
    rank : str
        The rank of the current process in the distributed setup.
    master_port : str
        The port used for communication in the distributed setup.

    Yields
    ------
        None
            The context manager yields control back to the caller with the environment
            variables patched.


    Example
    -------
    >>> with get_env_patcher(
    >>>     run_uuid="1234",
    >>>     rank="0",
    >>>     master_port="29500",
    >>> ):
    >>>     # Your code here

    """
    try:
        # Init environment variables
        environs: dict[str, str] = {}
        # Get the device type used in this settings
        devices = ",".join([str(i) for i in range(torch.cuda.device_count())])
        environs["APPOINTED_CUDA_DEVICE"] = devices
        environs["WORLD_SIZE"] = (
            str(torch.cuda.device_count())
        )
        environs["LOCAL_WORLD_SIZE"] = (
            str(torch.cuda.device_count())
        )
        # Set other environment variables
        if dist.is_initialized():
            master_port = os.environ["MASTER_PORT"]
        environs |= {
            # Shared
            "MASTER_ADDR": "127.0.0.1",
            "PYTHONUNBUFFERED": "1",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            "NODE_RANK": "0",
            "RUN_UUID": run_uuid,
            # Collaboration dependent
            "RANK": rank,
            "LOCAL_RANK": rank,
            "MASTER_PORT": master_port,
        }
        # Yield the context manager
        with _patch_env(**environs) as env_patcher:
            # NOTE: This is required most times to prevent the new StreamingDataset to
            # crash because it finds stale shared memories
            streaming.base.util.clean_stale_shared_memory()  # type: ignore[reportAttributeAccessIssue]
            log(
                DEBUG,
                "Environment variables patched for worker with rank"
                " %s.\n\t\tRANK=%s, WORLD_SIZE=%s, LOCAL_RANK=%s,"
                " LOCAL_WORLD_SIZE=%s, NODE_RANK=%s, MASTER_ADDR=%s,"
                " MASTER_PORT=%s, PYTHONUNBUFFERED=%s,"
                " TORCH_NCCL_ASYNC_ERROR_HANDLING=%s, RUN_UUID=%s,"
                " APPOINTED_CUDA_DEVICE=%s",
                rank,
                os.getenv("RANK"),
                os.getenv("WORLD_SIZE"),
                os.getenv("LOCAL_RANK"),
                os.getenv("LOCAL_WORLD_SIZE"),
                os.getenv("NODE_RANK"),
                os.getenv("MASTER_ADDR"),
                os.getenv("MASTER_PORT"),
                os.getenv("PYTHONUNBUFFERED"),
                os.getenv("TORCH_NCCL_ASYNC_ERROR_HANDLING"),
                os.getenv("RUN_UUID"),
                os.getenv("APPOINTED_CUDA_DEVICE"),
            )
            yield env_patcher
    except Exception as e:  # noqa: BLE001
        log(
            ERROR,
            "Error while patching the environment variables.",
            exc_info=e,
            stack_info=True,
        )
    finally:
        # NOTE: Setting the environment variables back to the latest value found to
        # recycle the PyTorch Process Group for the next task to perform
        if dist.is_initialized():
            os.environ["MASTER_PORT"] = master_port
