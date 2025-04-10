"""The module defines the Worker class and related functions for handling fit and eval.

Classes
-------
    Worker
        Worker process for handling training and evaluation tasks.

Functions
---------
    get_training_results_from_worker(worker: Worker | None) -> (
        tuple[
            tuple[NDArrays, int],
            tuple[int, dict],
            np.ndarray,
            tuple[SharedMemory, SharedMemory, SharedMemory],
        ] | None
    )
        Collect training results from a single worker.

    create_new_worker(
        config: BaseConfig,
        task_queue: QueueType,
        result_queue: QueueType,
        node_manager_uuid: str,
        run_uuid: str,
        parameters_metadata: ModelParametersMetadata,
        worker_rank: int,
        cpu_only: bool,
        cpu_concurrency: int,
    ) -> Worker
        Create a new Worker instance.

    start_worker(worker: Worker) -> None
        Start a Worker instance.

Imports
-------
    - contextlib
    - copy
    - gc
    - os
    - time
    - uuid
    - logging
    - multiprocessing.queues
    - multiprocessing.shared_memory
    - typing
    - multiprocess
    - numpy
    - omegaconf
    - torch
    - flwr.common
    - photon.clients.llm_client_functions
    - photon.conf.base_schema
    - photon.strategy.aggregation
    - photon.worker.utils
    - photon.shm.utils
    - photon.shm.constants
"""

import contextlib
import copy
import gc
import os
import time
import uuid
from logging import DEBUG, ERROR
from multiprocessing.queues import Queue as QueueType
from multiprocessing.shared_memory import SharedMemory
from typing import TYPE_CHECKING, Any, cast

import multiprocess as mp
import numpy as np
import torch
from flwr.common import Config, NDArrays
from flwr.common.logger import log, update_console_handler
from flwr.common.recordset_compat import ConfigsRecord
from omegaconf import DictConfig

from photon.clients.llm_client_functions import llm_eval, llm_fit
from photon.conf.base_schema import BaseConfig
from photon.shm.constants import (
    NM_CONFIG_SHM,
    NM_PARAMETERS_SHM,
    W_EVAL_LOSS_SHM,
    W_METRICS_SHM,
    W_N_SAMPLES_SHM,
    W_PARAMETERS_SHM,
)
from photon.shm.utils import (
    ModelParametersMetadata,
    close_all_shms,
    get_config_shm,
    get_dict_configsrecord_shm,
    get_eval_loss_shm,
    get_num_samples_shm,
    get_parameters_shm,
    remove_shm_from_resource_tracker,
    set_config_shm,
    set_eval_loss_shm,
    set_num_samples_shm,
    set_parameters_shm,
)
from photon.worker.utils import WorkerResultMessage, get_env_patcher

if TYPE_CHECKING:
    from composer import Trainer
    from flwr.common.record.typeddict import TypedDict


class Worker(mp.Process):  # type: ignore[reportAttributeAccessIssue]
    """Worker process for handling training and evaluation tasks.

    This class represents a worker process that handles training and evaluation tasks
    using shared memory for communication. It inherits from `multiprocessing.Process`.

    Attributes
    ----------
        worker_uuid : str
            The unique identifier for the worker.
        config : BaseConfig
            The configuration object for the worker.
        _llm_config : DictConfig
            The LLM-specific configuration extracted from the main config.
        task_queue : QueueType
            The queue for tasks assigned to the worker.
        result_queue : QueueType
            The queue for results produced by the worker.
        node_manager_uuid : str
            The UUID of the node manager.
        run_uuid : str
            The unique identifier for the run.
        parameters_metadata : ModelParametersMetadata
            Metadata for the model parameters.
        worker_rank : int
            The rank of the worker.
        worker_metrics_sh : SharedMemory | None
            Shared memory object for worker metrics.
        worker_metrics : Config
            Configuration object for worker metrics.
        auto_terminate : bool
            Flag indicating whether the worker should terminate after the current task.
        worker_parameters : NDArrays | None
            The model parameters used by the worker.
        worker_parameters_sh : SharedMemory | None
            Shared memory object for worker parameters.

    """

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        config: BaseConfig,
        worker_uuid: str,
        task_queue: QueueType,
        result_queue: QueueType,
        node_manager_uuid: str,
        run_uuid: str,
        parameters_metadata: ModelParametersMetadata,
        worker_rank: int,
    ) -> None:
        """Initialize the Worker instance.

        Parameters
        ----------
        config : BaseConfig
            The configuration object for the worker.
        worker_uuid : str
            The unique identifier for the worker.
        task_queue : QueueType
            The queue for tasks assigned to the worker.
        result_queue : QueueType
            The queue for results produced by the worker.
        node_manager_uuid : str
            The UUID of the node manager.
        run_uuid : str
            The unique identifier for the run.
        parameters_metadata : ModelParametersMetadata
            Metadata for the model parameters.
        worker_rank : int
            The rank of the worker.

        Raises
        ------
        TypeError
            If the LLM config is not a DictConfig.

        """
        super().__init__()
        self.worker_uuid = worker_uuid
        self.config = config
        # Extract the LLM part of the config
        self._llm_config = self.config.llm_config
        if not isinstance(self._llm_config, DictConfig):
            msg = "The LLM config is not a DictConfig."
            raise TypeError(msg)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.node_manager_uuid = node_manager_uuid
        self.run_uuid = run_uuid
        self.parameters_metadata = parameters_metadata
        self.worker_rank = worker_rank
        self.worker_metrics_sh: SharedMemory | None = None
        self.worker_metrics: Config = {}
        self.auto_terminate = False
        self.worker_parameters: NDArrays | None = None
        self.worker_parameters_sh: SharedMemory | None = None
        self.external_trainer: Trainer | None = None

    def fit_action(
        self,
        cid: int,
        config: ConfigsRecord,
    ) -> int:
        """Perform the fit action for the worker.

        This function retrieves the round parameters from shared memory, performs the
        fit action, and updates the shared memory with the results.

        Parameters
        ----------
        cid : int
            The client ID.
        config : ConfigsRecord
            The configuration record for the fit action.

        Raises
        ------
        ValueError
            If the worker parameters are None.

        Returns
        -------
        n_samples : int
            The number of samples processed by the worker.

        """
        # NOTE: Since we removed some timeouts for efficiency reasons, when this piece
        # of code is executed the SharedMemory may not be ready, so we need to wait
        # until it is
        # Shared memory for round parameters
        shm_not_written = True
        round_parameters: NDArrays | None = None
        while shm_not_written:
            try:
                round_parameters, _round_parameters_sh = get_parameters_shm(
                    parameters_metadata=self.parameters_metadata,
                    name=self.node_manager_uuid + NM_PARAMETERS_SHM,
                )
                shm_not_written = False
            except FileNotFoundError:  # noqa: PERF203
                pass
        assert round_parameters is not None
        # Call fit on shared parameters
        fit_trained_weights, fit_num_samples, train_metrics, self.external_trainer = (
            llm_fit(
                self.external_trainer,
                round_parameters,
                config,
                copy.deepcopy(self._llm_config),
                cid,
            )
        )
        if int(os.getenv("LOCAL_RANK", "")) == 0:
            # Creating the parameters shared memory
            self.create_parameters_shm()
            if self.worker_parameters is None:
                msg = "Worker parameters are None. Cannot perform partial aggregation."
                raise ValueError(msg)
            # Update shared memories
            set_num_samples_shm(self.worker_num_samples, fit_num_samples)
            set_parameters_shm(self.worker_parameters, fit_trained_weights)
            # Destroy the shared memory for the metrics dictionary because, since it was
            # created in a previous step (fit or eval of the same or different round) it
            # may have a different size.
            if self.worker_metrics_sh is not None:
                with contextlib.suppress(FileNotFoundError):
                    self.worker_metrics_sh.close()
                    self.worker_metrics_sh.unlink()
            # NOTE: Now we know the structure and we can create the train metrics
            # shared memory. Since the structure might changed, we cannot assume
            # a fixed size for the shared memory.
            (
                self.worker_metrics,
                self.worker_metrics_sh,
            ) = get_config_shm(
                config=train_metrics,
                create=True,
                name=self.worker_uuid + W_METRICS_SHM,
            )
            set_config_shm(train_metrics, self.worker_metrics_sh)
        return fit_num_samples

    def evaluate_action(
        self,
        _cid: int,
        config: ConfigsRecord,
    ) -> int:
        """Perform the evaluate action for the worker.

        This function retrieves the round parameters from shared memory, performs the
        evaluate action, and updates the shared memory with the results.

        Parameters
        ----------
        _cid : int
            The client ID.
        config : ConfigsRecord
            The configuration record for the evaluate action.

        Returns
        -------
        n_samples : int
            The number of samples processed by the worker.

        """
        # NOTE: Since we removed some timeouts for efficiency reasons, when this piece
        # of code is executed the SharedMemory may not be ready, so we need to wait
        # until it is
        # Shared memory for round parameters
        shm_not_written = True
        round_parameters: NDArrays | None = None
        while shm_not_written:
            try:
                round_parameters, _round_parameters_sh = get_parameters_shm(
                    parameters_metadata=self.parameters_metadata,
                    name=self.node_manager_uuid + NM_PARAMETERS_SHM,
                )
                shm_not_written = False
            except FileNotFoundError:  # noqa: PERF203
                pass
        assert round_parameters is not None
        # Call evaluate on shared parameters
        eval_loss, eval_num_samples, eval_metrics, self.external_trainer = llm_eval(
            self.external_trainer,
            round_parameters,
            config,
            copy.deepcopy(self._llm_config),
        )
        if int(os.getenv("LOCAL_RANK", "")) == 0:
            # Set the the values obtained to the SharedMemories
            set_num_samples_shm(self.worker_num_samples, eval_num_samples)
            set_eval_loss_shm(self.worker_eval_loss, eval_loss)
            # Destroy the shared memory for the metrics dictionary because, since it was
            # created in a previous step (fit or eval of the same or different round) it
            # may have a different size.
            if self.worker_metrics_sh is not None:
                with contextlib.suppress(FileNotFoundError):
                    self.worker_metrics_sh.close()
                    self.worker_metrics_sh.unlink()
            # NOTE: Now we know the structure and we can create the train metrics
            # shared memory. Since the structure might changed, we cannot assume
            # a fixed size for the shared memory.
            (
                self.worker_metrics,
                self.worker_metrics_sh,
            ) = get_config_shm(
                config=eval_metrics,
                create=True,
                name=self.worker_uuid + W_METRICS_SHM,
            )
            set_config_shm(eval_metrics, self.worker_metrics_sh)
        return eval_num_samples

    def process_task(self, client_id: int, action: str = "fit") -> None:
        """Process the received task.

        This function processes the received task by executing the fit or evaluate
        action based on the provided action parameter.

        Parameters
        ----------
        client_id : int
            The client ID.
        action : str, optional
            The action to be performed ("fit" or "evaluate"). Default is "fit".

        """
        # Take the timestamp before training a single client
        start_time = time.time_ns()
        # Loads a dict from the shared memory buffer
        # FL config shared memory
        # NOTE: We MUST keep the sh variable even if we don't use it
        fl_instructions_config, _fl_instructions_config_sh = get_dict_configsrecord_shm(
            config=cast("TypedDict[str, Any]", {}),
            name=self.node_manager_uuid + NM_CONFIG_SHM,
        )
        client_config: ConfigsRecord = fl_instructions_config[str(client_id)]
        # Load client
        # Patch the environment given the received instructions
        with get_env_patcher(
            run_uuid=(str(client_config["run_uuid"])),
            rank=str(self.worker_rank),
            master_port=str(client_config["MASTER_PORT"]),
        ):
            # Try to execute the task of the client
            try:
                if action == "fit":
                    # Launch the fit routine
                    n_samples = self.fit_action(client_id, client_config)
                    # Take the timestamp after the task is done
                    end_time = time.time_ns()
                    # Only rank 0 returns the result
                    if int(os.getenv("LOCAL_RANK", "")) == 0:
                        # Put the result in the result queue
                        self.result_queue.put(
                            WorkerResultMessage(
                                n_samples,
                                (end_time - start_time) * 1e-9,
                                self.worker_uuid,
                            ),
                        )
                elif action == "evaluate":
                    # Launch the evaluate routine
                    n_samples = self.evaluate_action(client_id, client_config)
                    # Only rank 0 returns the result
                    if int(os.getenv("LOCAL_RANK", "")) == 0:
                        # Take the timestamp after the task is done
                        end_time = time.time_ns()
                        # Put the result in the result queue
                        self.result_queue.put(
                            WorkerResultMessage(
                                n_samples,
                                (end_time - start_time) * 1e-9,
                                self.worker_uuid,
                            ),
                        )
            except Exception as e:  # noqa: BLE001
                log(
                    ERROR,
                    "Worker %s failed executing %s for client %s.",
                    self.worker_uuid,
                    action,
                    client_id,
                    exc_info=e,
                    stack_info=True,
                )
                # Always return the error to the result queue to notify the NodeManager
                self.result_queue.put(
                    WorkerResultMessage(
                        -1,
                        0.0,
                        "",
                    ),
                )
                # Append back to the task queue
                self.task_queue.put((client_id, action))
                # Set the auto_terminate flag to True for suicide
                self.auto_terminate = True

    def link_shms(self) -> None:
        """Link shared memory segments for the worker.

        This function creates and links the shared memory segments for the number of
        samples and evaluation loss.

        """
        # Number of samples shared memory
        self.worker_num_samples, self.worker_num_samples_sh = get_num_samples_shm(
            create=True,
            name=self.worker_uuid + W_N_SAMPLES_SHM,
        )
        set_num_samples_shm(self.worker_num_samples, 0)
        # Evaluation loss shared memory
        self.worker_eval_loss, self.worker_eval_loss_sh = get_eval_loss_shm(
            create=True,
            name=self.worker_uuid + W_EVAL_LOSS_SHM,
        )

    def create_parameters_shm(self) -> None:
        """Create shared memory for worker parameters.

        This function creates the shared memory segment for the worker's parameters.

        """
        # NOTE: This is the Worker's shared memory for the fit results.
        # NodeManager should only read this. Worker should only write this.
        # Shared memory for worker's parameters
        try:
            self.worker_parameters, self.worker_parameters_sh = get_parameters_shm(
                parameters_metadata=self.parameters_metadata,
                name=self.worker_uuid + W_PARAMETERS_SHM,
            )
        except FileNotFoundError:
            log(DEBUG, "Shared memory for parameters doesn't exists. Creating it.")
            self.worker_parameters, self.worker_parameters_sh = get_parameters_shm(
                create=True,
                parameters_metadata=self.parameters_metadata,
                name=self.worker_uuid + W_PARAMETERS_SHM,
            )
        except Exception as e:  # noqa: BLE001
            log(
                ERROR,
                "Error while creating the shared memory for the worker's parameters.",
                exc_info=e,
                stack_info=True,
            )

    def run(self) -> None:
        """Run the worker process.

        This function sets up the worker process, links shared memory segments, and
        processes tasks from the task queue.

        """
        # Fix the logger
        update_console_handler(level=DEBUG, colored=False, timestamps=True)
        # Create shared memories
        # Call the monkey-patch for the resource-register
        remove_shm_from_resource_tracker()
        # NOTE: This goes here because it needs to be done in the child process!
        # This is the first piece of code of the worker that live in the child
        # process, the `__init__()` function does not.
        self.link_shms()
        # Task loop
        task: tuple[int, str] | None = None
        for task in iter(self.task_queue.get, None):
            cid, action = task  # type: ignore[misc]
            self.process_task(cid, action)  # type: ignore[has-type]
            if self.auto_terminate:
                break
        torch.cuda.empty_cache()
        gc.collect()

    def soft_shutdown(self) -> None:
        """Perform a soft shutdown of the worker.

        This function closes and unlinks the worker's shared memory segments.

        """
        # Close and unlink Worker's shared memories
        close_all_shms(self.worker_uuid)


def get_training_results_from_worker(
    worker: Worker | None,
) -> (
    tuple[
        NDArrays,
        dict,
        np.ndarray,
        tuple[SharedMemory, SharedMemory, SharedMemory],
    ]
    | None
):
    """Collect training results from a single worker.

    This function retrieves the training results from a single worker, including the
    model parameters, sample counts, metrics, and shared memory objects.

    Parameters
    ----------
    worker : Worker | None
        The Worker object from which to collect training results. If None, the
        function logs an error and returns None.

    Returns
    -------
        tuple[
            NDArrays,
            dict,
            int,
            tuple[SharedMemory, SharedMemory, SharedMemory],
        ] | None
            A tuple containing:
            - The model parameters (NDArrays).
            - The metrics dictionary.
            - An ndarray containing the number of samples.
            - A tuple with the shared memory objects for parameters, number of samples,
                and metrics.
            If the worker is None or not alive, the function returns None.

    """
    if worker is None:
        log(
            ERROR,
            "Worker is None. Cannot collect its results.",
        )
        return None
    if worker.is_alive():
        w_parameters, w_parameters_shm = get_parameters_shm(
            parameters_metadata=worker.parameters_metadata,
            name=worker.worker_uuid + W_PARAMETERS_SHM,
        )
        w_num_samples, w_num_samples_shm = get_num_samples_shm(
            name=worker.worker_uuid + W_N_SAMPLES_SHM,
        )
        w_metrics, w_metrics_shm = get_config_shm(
            config={},
            name=worker.worker_uuid + W_METRICS_SHM,
        )
        return (
            w_parameters,
            w_metrics,
            w_num_samples,
            (w_parameters_shm, w_num_samples_shm, w_metrics_shm),
        )
    log(
        ERROR,
        "Worker %s is not alive anymore. Cannot collect its results.",
        worker.worker_uuid,
    )
    return None


def create_new_worker(  # noqa: PLR0913, PLR0917
    config: BaseConfig,
    task_queue: QueueType,
    result_queue: QueueType,
    node_manager_uuid: str,
    run_uuid: str,
    parameters_metadata: ModelParametersMetadata,
    worker_rank: int,
) -> Worker:
    """Create a new Worker instance.

    This function generates a unique UUID for the worker, creates a Worker object with
    the provided configuration and parameters, and returns the Worker object.

    Parameters
    ----------
    config : BaseConfig
        The configuration object for the worker.
    task_queue : QueueType
        The queue for tasks assigned to the worker.
    result_queue : QueueType
        The queue for results produced by the worker.
    node_manager_uuid : str
        The UUID of the node manager.
    run_uuid : str
        The unique identifier for the run.
    parameters_metadata : ModelParametersMetadata
        Metadata for the model parameters.
    worker_rank : int
        The rank of the worker.

    Returns
    -------
        Worker
            The created Worker object.

    """
    # Generate the Worker's UUID
    worker_uuid = node_manager_uuid + "-" + str(uuid.uuid4())
    # Create the Worker object
    return Worker(
        config=config,
        worker_uuid=worker_uuid,
        task_queue=task_queue,
        result_queue=result_queue,
        node_manager_uuid=node_manager_uuid,
        run_uuid=run_uuid,
        parameters_metadata=parameters_metadata,
        worker_rank=worker_rank,
    )
    # Return the Worker object


def start_worker(worker: Worker) -> None:
    """Start a Worker instance.

    This function starts the provided Worker instance and logs its UUID and rank.

    Parameters
    ----------
    worker : Worker
        The Worker instance to be started.

    """
    worker.start()
    log(
        DEBUG,
        "Worker %s with rank %s started.",
        worker.worker_uuid,
        worker.worker_rank,
    )
