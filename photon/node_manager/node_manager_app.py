"""Node Manager Application for Federated Learning.

This module defines the NodeManagerApp class, which is responsible for managing the
training and evaluation tasks on a node in a federated learning setup. The class
handles the creation and management of worker processes, the distribution of tasks
to these workers, and the aggregation of results. It also manages the configuration
and shared memory required for the tasks.

Classes
-------
NodeManagerApp
    Manages the training and evaluation tasks on a node.

Functions
---------
set_start_method
    Sets the start method for multiprocessing to 'spawn'.

Modules
-------
ast
    Provides functions to parse and process Python abstract syntax trees.
collections
    Implements specialized container datatypes.
copy
    Provides functions for shallow and deep copy operations.
gc
    Provides functions for garbage collection.
logging
    Provides a flexible framework for emitting log messages from Python programs.
multiprocessing.shared_memory
    Provides shared memory for multiprocessing.
os
    Provides a way of using operating system dependent functionality.
pickle
    Implements binary protocols for serializing and de-serializing objects.
tempfile
    Generates temporary files and directories.
time
    Provides various time-related functions.
typing
    Provides runtime support for type hints.
uuid
    Implements UUID objects as specified in RFC 4122.
cloudpickle
    Extended pickling support for Python objects.
multiprocessing.queues
    Provides a shared job queue implementation.
flwr.client
    Provides client-side functionality for Flower framework.
flwr.common
    Provides common utilities and types for Flower framework.
multiprocess
    Provides support for concurrent processing.
numpy
    Provides support for large, multi-dimensional arrays and matrices.
omegaconf
    Provides support for hierarchical configurations.
composer.loggers
    Provides logging utilities for ML training.
composer.utils.misc
    Provides miscellaneous utilities for ML training.
torch
    Provides support for tensor computation and deep learning.
contextlib
    Utilities for with-statement contexts.

Dependencies
------------
photon.conf.base_schema
    Provides the base configuration schema for the node manager.
photon.node_manager.utils
    Provides utility functions for the node manager.
photon.worker.worker
    Provides worker-related functionality for the node manager.
photon.resources_manager
    Provides resource management utilities for the node manager.
photon.utils
    Provides general utilities for the node manager.
flwr.server.strategy.aggregate
    Provides aggregation strategies for federated learning.
photon.strategy.aggregation
    Provides custom aggregation strategies for federated learning.
photon.clients.utils
    Provides client-related utilities for the node manager.
photon.shm.utils
    Provides shared memory utilities for the node manager.
photon.shm.constants
    Provides shared memory constants for the node manager.
photon.worker.utils
    Provides worker-related utilities for the node manager.
"""

import ast
import contextlib
import copy
import gc
import os
import pickle  # noqa: S403
import socket
import time
import uuid
from collections import defaultdict
from logging import DEBUG, ERROR
from queue import Queue as StandardQueue
from typing import TYPE_CHECKING, Any, cast

import cloudpickle
import ray
import torch
from composer.loggers import RemoteUploaderDownloader
from composer.utils.misc import get_free_tcp_port
from flwr.client import ClientApp
from flwr.client.typing import ClientFnExt, Mod
from flwr.common.logger import log
from flwr.common.record.typeddict import TypedDict
from flwr.common.recordset_compat import ConfigsRecord
from flwr.common.typing import NDArrays, Scalar
from flwr.server.strategy.aggregate import weighted_loss_avg
from multiprocess import (
    Queue,  # type: ignore[reportAttributeAccessIssue]
    set_start_method,  # type: ignore[reportAttributeAccessIssue]
)
from omegaconf import DictConfig, OmegaConf

from photon.clients.utils import get_raw_model_parameters
from photon.shm.constants import (
    NM_CONFIG_SHM,
    W_EVAL_LOSS_SHM,
    W_METRICS_SHM,
    W_N_SAMPLES_SHM,
)
from photon.shm.utils import (
    ModelParametersMetadata,
    close_all_shms,
    get_config_shm,
    get_dict_configsrecord_shm,
    get_eval_loss_shm,
    get_num_samples_shm,
    remove_shm_from_resource_tracker,
    set_dict_configsrecord_shm,
    set_num_samples_shm,
)
from photon.strategy.aggregation import weighted_average
from photon.utils import get_n_cuda_devices
from photon.worker.utils import WorkerResultMessage
from photon.worker.worker import (
    Worker,
    create_new_worker,
    get_training_results_from_worker,
    start_worker,
)

if TYPE_CHECKING:
    import threading
    from multiprocessing.queues import Queue as QueueType
    from multiprocessing.shared_memory import SharedMemory

    from ray import ObjectRef

    from photon.conf.base_schema import BaseConfig

set_start_method("spawn", force=True)
pickle.Pickler = cloudpickle.Pickler  # type: ignore[misc]


class NodeManagerApp(ClientApp):
    """NodeManagerApp is responsible for managing the training and evaluation on a node.

    This class handles the creation and management of worker processes, the distribution
    of tasks to these workers, and the aggregation of results. It also manages the
    configuration and shared memory required for the tasks.

    Attributes
    ----------
        cfg : BaseConfig
            The configuration object loaded from the config file.
        task_queue : QueueType
            The queue for tasks assigned to the workers.
        result_queue : QueueType
            The queue for results produced by the workers.
        node : NodeProperties
            The properties of the node, including hardware accelerators.
        properties : dict[str, ConfigsRecordValues]
            A dictionary containing the node properties.
        node_manager_uuid : str
            The unique identifier for the node manager.
        refresh_period : int
            The refresh period for the node manager.
        remote_up_down : RemoteUploaderDownloader | None
            The remote uploader/downloader for handling S3 communication.
        parameters_metadata : ModelParametersMetadata
            Metadata for the model parameters.
        round_parameters : NDArrays | None
            The model parameters for the current round.
        round_parameters_sh : SharedMemory | None
            Shared memory object for the round parameters.
        workers_dict : dict[int, Worker]
            A dictionary of worker processes.
        fl_instructions_config : TypedDict[str, ConfigsRecord] | None
            The configuration for the federated learning instructions.
        fl_instructions_config_sh : SharedMemory | None
            Shared memory object for the federated learning instructions.

    """

    def __init__(
        self,
        client_fn: ClientFnExt | None = None,  # Only for backward compatibility
        mods: list[Mod] | None = None,
    ) -> None:
        """Initialize the NodeManagerApp instance.

        This method sets up the NodeManagerApp by loading the configuration, setting up
        queues, getting node properties, creating workers, and initializing shared
        memories.

        Parameters
        ----------
        client_fn : ClientFnExt | None, optional
            A client function for backward compatibility. Default is None.
        mods : list[Mod] | None, optional
            A list of modifications to apply. Default is None.

        Raises
        ------
        ValueError
            If the environmental variable PHOTON_SAVE_PATH is not set.
        TypeError
            If the LLM configuration is not a DictConfig.

        """
        super().__init__(client_fn=client_fn, mods=mods)
        # Get the environmental variable for the dump folder
        save_path = os.environ.get("PHOTON_SAVE_PATH", "")
        # Raise an error if the environmental variable is not set
        if not save_path:
            msg = "The environmental variable PHOTON_SAVE_PATH is not set."
            raise ValueError(msg)
        # Load the configuration from the config file
        self.cfg = cast("BaseConfig", OmegaConf.load(save_path + "/config.yaml"))
        # Resolve the config and set it to be editable in place
        OmegaConf.resolve(self.cfg)
        OmegaConf.set_struct(self.cfg, value=False)
        # Set up Queues
        self.task_queue: QueueType = Queue()
        # One result_queue for all GPUs
        self.result_queue: QueueType = Queue()
        self.node_manager_uuid = self.cfg.run_uuid + "-" + str(uuid.uuid4())
        self.refresh_period = self.cfg.photon.refresh_period
        self.remote_up_down: RemoteUploaderDownloader | None = None
        self.create_remote_up_down()
        # Call the monkey-patch for the resource-register
        remove_shm_from_resource_tracker()
        # Extract the LLM part of the config
        llm_config = self.cfg.llm_config
        if not isinstance(llm_config, DictConfig):
            msg = "The LLM configuration is not a DictConfig."
            raise TypeError(msg)
        # Get initial model parameters
        parameters = cast(
            "NDArrays",
            get_raw_model_parameters(
                copy.deepcopy(llm_config),
                aggregate_momenta=self.cfg.fl.aggregate_momenta,
            ),
        )
        parameters_metadata = ModelParametersMetadata.from_ndarrays(parameters)
        del parameters
        # Parameter metadata
        self.parameters_metadata = parameters_metadata
        # Shared memory for round parameters
        self.round_parameters: NDArrays | None = None
        self.round_parameters_sh: SharedMemory | None = None
        # Create workers
        self.workers_dict: dict[int, Worker] = {}
        self.create_and_start_workers()
        # Initialize shared memories for the config
        self.fl_instructions_config: TypedDict[str, ConfigsRecord] | None = None
        self.fl_instructions_config_sh: SharedMemory | None = None
        # Master port for PyTorch Distributed
        self.nm_master_port: int | None = None
        # Initialize Ray
        ray.init(
            "auto",
        )
        # Ray objects to free
        self.list_of_ray_object_refs: list[ObjectRef] | None = None
        self.ray_garbage_queue: StandardQueue[ObjectRef] = StandardQueue()
        self.list_of_threads: list[threading.Thread] = []

    def create_and_start_workers(self) -> None:
        """Create and start worker processes.

        This method creates worker processes based on the number of CUDA devices or CPU
        concurrency and starts them.
        """
        for i in range(
            (get_n_cuda_devices()),
        ):
            worker = create_new_worker(
                config=self.cfg,
                task_queue=self.task_queue,
                result_queue=self.result_queue,
                node_manager_uuid=self.node_manager_uuid,
                run_uuid=self.cfg.run_uuid,
                parameters_metadata=self.parameters_metadata,
                worker_rank=i,
            )
            self.workers_dict[i] = worker
            log(
                DEBUG,
                "Created worker with rank %s",
                i,
            )
        # Start the workers
        for worker in self.workers_dict.values():
            start_worker(worker)
        log(
            DEBUG,
            "NodeManagerApp %s on node %s: all workers started.",
            self.node_manager_uuid,
            socket.gethostname(),
        )

    def check_workers_health(self) -> None:
        """Check the health of worker processes.

        This method checks if the worker processes are alive. If a worker is found to be
        dead, it restarts the worker.
        """
        for rank, worker in self.workers_dict.items():
            if not worker.is_alive():
                log(
                    DEBUG,
                    "NodeManagerApp %s on node %s: worker %s is dead. Restarting it...",
                    self.node_manager_uuid,
                    socket.gethostname(),
                    rank,
                )
                close_all_shms(worker.worker_uuid)
                self.workers_dict[rank] = create_new_worker(
                    config=self.cfg,
                    task_queue=self.task_queue,
                    result_queue=self.result_queue,
                    node_manager_uuid=self.node_manager_uuid,
                    run_uuid=self.cfg.run_uuid,
                    parameters_metadata=self.parameters_metadata,
                    worker_rank=rank,
                )
                start_worker(self.workers_dict[rank])

    def close_workers(self) -> None:
        """Perform a soft shutdown of all worker processes.

        This method initiates a soft shutdown for each worker process, waits until they
        are terminated, and then performs garbage collection and clears the CUDA cache.
        """
        # Wait until the worker is dead
        for worker in self.workers_dict.values():
            worker.soft_shutdown()
            while worker.is_alive():
                time.sleep(0.1)
                worker.terminate()
        log(
            DEBUG,
            "NodeManagerApp %s on node %s: workers are dead.",
            self.node_manager_uuid,
            socket.gethostname(),
        )
        gc.collect()
        torch.cuda.empty_cache()

    def create_remote_up_down(self) -> None:
        """Create and initialize the remote uploader/downloader for S3 communication.

        This method sets up the remote uploader/downloader using the S3 configuration
        provided in the configuration file.
        """
        if self.cfg.photon.comm_stack.s3:
            bucket_uri = f"s3://{self.cfg.s3_comm_config.bucket_name}"
            self.remote_up_down = RemoteUploaderDownloader(
                bucket_uri=bucket_uri,
                backend_kwargs={
                    "bucket": self.cfg.s3_comm_config.bucket_name,
                    "prefix": f"{self.cfg.run_uuid}/server",  # Don't touch
                    "region_name": None,  # Not necessary
                    "endpoint_url": None,  # Will be read from env var
                    "aws_access_key_id": None,  # Will be read from config file
                    "aws_secret_access_key": None,  # Will be read from config file
                    "aws_session_token": None,  # Will be automatically generated
                    "client_config": OmegaConf.to_container(
                        self.cfg.s3_comm_config.backend_kwargs.client_config,
                    ),  # And using defaults
                    "transfer_config": None,  # Using defaults
                },
                file_path_format_string="{remote_file_name}",  # Don't touch
                num_concurrent_uploads=1,
                upload_staging_folder=None,  # Don't touch, it's /tmp by default
                use_procs=True,  # Don't touch
                num_attempts=self.cfg.s3_comm_config.num_attempts,
            )
            self.remote_up_down.init(run_name=self.cfg.run_uuid)

    def fit(
        self,
        configs: TypedDict[str, ConfigsRecord],
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """Perform the fit (training) operation.

        This method distributes the training tasks to the workers, collects the results,
        and aggregates the training metrics.

        Parameters
        ----------
        configs : TypedDict[str, ConfigsRecord]
            The configuration dictionary for the training tasks.

        Returns
        -------
            tuple[NDArrays, int, dict[str, Scalar]]
                A tuple containing:
                - The aggregated model parameters (NDArrays).
                - The total number of samples.
                - The aggregated training metrics dictionary.

        Raises
        ------
        ValueError
            If no assignments are found in the config.

        """
        start_time = time.time()

        fit_ins_config = configs.pop("fitins.config")
        assignments = fit_ins_config.pop("client_ids")
        if assignments is None:  # type: ignore[reportUnnecessaryComparison]
            msg = "No assignments found in the config."
            raise ValueError(msg)
        list_of_cids_to_train: list[str] = ast.literal_eval(str(assignments))
        # Update each client config in `configs` with the shared parameters in
        # `fit_ins_config`
        for cid in list_of_cids_to_train:
            configs[str(cid)].update(fit_ins_config)

        node_train_metrics: dict[str, Scalar] = {}
        aggregated_params: NDArrays = []
        sum_of_samples: int = 0
        try:
            (
                aggregated_params,
                sum_of_samples,
                node_train_metrics,
            ) = self.collaborative_fit(configs, list_of_cids_to_train)
        except Exception as e:  # noqa: BLE001
            log(
                ERROR,
                "NodeManager %s",
                self.node_manager_uuid,
                exc_info=e,
                stack_info=True,
            )
        # Adding node training time in the metrics
        node_train_metrics.update(
            {
                "node_training_time_s": float(time.time() - start_time),
            },
        )
        log(
            DEBUG,
            "NodeManager %s: results have been processed. "
            "The time spent before collecting results was %s seconds.",
            self.node_manager_uuid,
            time.time() - start_time,
        )
        # Return results
        return (
            aggregated_params,
            int(sum_of_samples),
            node_train_metrics,
        )

    def collaborative_fit(  # noqa: C901
        self,
        configs: TypedDict[str, ConfigsRecord],
        list_of_cids_to_train: list[str],
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """Perform collaborative fit (training) for each client.

        This method distributes the training tasks to the workers collaboratively,
        collects the results, and aggregates the training metrics.

        Parameters
        ----------
        configs : TypedDict[str, ConfigsRecord]
            The configuration dictionary for the training tasks.
        list_of_cids_to_train : list[str]
            The list of client IDs to be trained.

        Returns
        -------
            tuple[NDArrays, int, dict[str, Scalar]]
                A tuple containing:
                - The aggregated model parameters (NDArrays).
                - The total number of samples.
                - The aggregated training metrics dictionary.

        """
        # Initialise partial aggregation variables
        aggregated_params: NDArrays = []
        sum_of_samples: int = 0
        node_train_metrics: dict = {}
        stats = defaultdict(list)
        # Here, workers are forced to collaborate with each other,
        # as such, we evaluate one client at a time
        while len(list_of_cids_to_train) > 0:
            self.check_workers_health()
            # Get the current cid
            current_cid = int(list_of_cids_to_train.pop(0))
            # Extract config
            config = cast("TypedDict", {str(current_cid): configs[str(current_cid)]})
            # Append NodeManager's config
            if self.nm_master_port is None:
                self.nm_master_port = get_free_tcp_port()
            config[str(current_cid)]["MASTER_PORT"] = self.nm_master_port
            # NOTE: Putting the node_manager_uuid in the config fails
            config[str(current_cid)]["run_uuid"] = self.cfg.run_uuid
            # Update instruction config shared memory
            _fl_instructions_config, fl_instructions_config_sh = (
                get_dict_configsrecord_shm(
                    config=config,
                    create=True,
                    name=self.node_manager_uuid + NM_CONFIG_SHM,
                )
            )
            set_dict_configsrecord_shm(config, fl_instructions_config_sh)
            # Send the collaborative task to the workers
            for _ in range(len(self.workers_dict)):
                self.task_queue.put((current_cid, "fit"))
            # Wait for the result
            worker_result: WorkerResultMessage | None = None
            while worker_result is None:
                try:
                    worker_result = self.result_queue.get(timeout=0.1)
                except Exception:  # noqa: BLE001, PERF203
                    for worker in self.workers_dict.values():
                        if not worker.is_alive():
                            worker_result = WorkerResultMessage(
                                -1,
                                0.0,
                                "",
                            )
            # Check if the training was successful
            if worker_result.n_samples > -1:
                # NOTE: The order here is important for compatibility with the server
                stats["device"].append(worker_result.device)
                stats["n_samples"].append(worker_result.n_samples)  # type: ignore[arg-type]
                stats["delta"].append(worker_result.delta)  # type: ignore[arg-type]
                # Get stuff from shared memories of the workers
                # NOTE: Keep a reference to the `*_shm` variables to prevent Seg Fault
                results = get_training_results_from_worker(self.workers_dict[0])
                if results is not None:
                    w_p, w_s_m, w_s, _shms = results
                    # NOTE: Not deep-copying here will trigger a segmentation fault
                    # because the SharedMemory object backing this array will be garbage
                    # collected after returning from this function
                    aggregated_params = copy.deepcopy(w_p)
                    sum_of_samples = w_s[0]
                    node_train_metrics = w_s_m
                    # Zero out the n_samples shared memory
                    set_num_samples_shm(w_s, 0)
                else:
                    log(ERROR, "Results received are invalid!")
                    list_of_cids_to_train.append(str(current_cid))
            else:
                # If the training was not successful, put the cid back in the list
                list_of_cids_to_train.append(str(current_cid))
                # Close all workers to refresh the state
                self.close_workers()
            with contextlib.suppress(FileNotFoundError):
                # Close the config shared memory
                fl_instructions_config_sh.close()
                fl_instructions_config_sh.unlink()
            # Empty the tasks list
            while not self.task_queue.empty():
                self.task_queue.get()
        # Return the results
        return (
            aggregated_params,
            sum_of_samples,
            node_train_metrics,
        )

    def eval(  # noqa: PLR0914, C901
        self,
        configs: TypedDict[str, ConfigsRecord],
    ) -> tuple[float, int, dict[Any, Any]]:
        """Perform the evaluation operation.

        This method distributes the evaluation tasks to the workers, collects the
        results, and aggregates the evaluation metrics.

        Parameters
        ----------
        configs : TypedDict[str, ConfigsRecord]
            The configuration dictionary for the evaluation tasks.

        Returns
        -------
            tuple[float, int, dict[Any, Any]]
                A tuple containing:
                - The aggregated evaluation loss.
                - The total number of samples.
                - The aggregated evaluation metrics dictionary.

        """
        start_time = time.time()

        evaluate_ins_config = configs.pop("evaluateins.config")
        assignments = evaluate_ins_config.pop("client_ids")
        assert assignments is not None, "No assignments found in the config."
        list_of_cids_to_eval: list[str] = ast.literal_eval(str(assignments))
        # Update each client config in `configs` with the shared parameters in
        # `fit_ins_config`
        for cid in list_of_cids_to_eval:
            configs[str(cid)].update(
                evaluate_ins_config,
                run_uuid=(self.cfg.run_uuid),
            )
        # Loop over virtual clients' results
        clients_eval_losses: list[tuple[int, float]] = []
        clients_eval_metrics: list[tuple[int, dict[str, Scalar]]] = []
        clients_eval_samples: list[int] = []
        # Here, workers are forced to collaborate with each other,
        # as such, we evaluate one client at a time
        while len(list_of_cids_to_eval) > 0:
            self.check_workers_health()
            # Get the current cid
            current_cid = int(list_of_cids_to_eval.pop(0))
            # Extract config
            config = cast("TypedDict", {str(current_cid): configs[str(current_cid)]})
            # Append NodeManager's config
            if self.nm_master_port is None:
                self.nm_master_port = get_free_tcp_port()
            config[str(current_cid)]["MASTER_PORT"] = self.nm_master_port
            # Update shared memories objects
            (
                self.fl_instructions_config,
                self.fl_instructions_config_sh,
            ) = get_dict_configsrecord_shm(
                config=config,
                create=True,
                name=self.node_manager_uuid + NM_CONFIG_SHM,
            )
            set_dict_configsrecord_shm(config, self.fl_instructions_config_sh)
            # Send the collaborative task to the workers
            for _ in range(len(self.workers_dict)):
                self.task_queue.put((current_cid, "evaluate"))
            # Wait for the result
            worker_results: WorkerResultMessage | None = None
            while worker_results is None:
                try:
                    worker_results = self.result_queue.get(timeout=10)
                except Exception:  # noqa: BLE001, PERF203
                    for worker in self.workers_dict.values():
                        if not worker.is_alive():
                            worker_results = WorkerResultMessage(
                                -1,
                                0.0,
                                "",
                            )
            # Check if the evaluation was successful
            if worker_results.n_samples > -1:
                # Get stuff from shared memories of the rank 0 worker
                # NOTE: Keep the `*_shm` variables to prevent Seg Fault
                w_eval_loss, _w_eval_loss_shm = get_eval_loss_shm(
                    name=self.workers_dict[0].worker_uuid + W_EVAL_LOSS_SHM,
                )
                w_num_samples, _w_num_samples_shm = get_num_samples_shm(
                    name=self.workers_dict[0].worker_uuid + W_N_SAMPLES_SHM,
                )
                w_metrics, _w_metrics_shm = get_config_shm(
                    config={},
                    name=self.workers_dict[0].worker_uuid + W_METRICS_SHM,
                )
                # Append eval losses to aggregate later
                clients_eval_losses.append((int(w_num_samples[0]), w_eval_loss[0]))
                # Append eval metrics to aggregate later
                clients_eval_metrics.append((int(w_num_samples[0]), w_metrics))
                # Append eval samples to aggregate later
                clients_eval_samples.append(int(w_num_samples[0]))
                # Zero out the n_samples shared memory
                set_num_samples_shm(w_num_samples, 0)
            else:
                list_of_cids_to_eval.append(str(current_cid))
                # Kill all the workers and restart
                self.close_workers()
            with contextlib.suppress(FileNotFoundError):
                # Close the config shared memory
                self.fl_instructions_config_sh.close()
                self.fl_instructions_config_sh.unlink()
            # Empty the tasks list
            while not self.task_queue.empty():
                self.task_queue.get()
        # Aggregation of eval losses and metrics
        node_eval_loss, node_eval_metrics, node_eval_samples = (
            weighted_loss_avg(clients_eval_losses),
            weighted_average(clients_eval_metrics),
            sum(clients_eval_samples),
        )
        node_eval_metrics.update({"node_eval_time_s": float(time.time() - start_time)})
        # Aggregation of eval samples
        log(
            DEBUG,
            "NodeManager %s: results have been processed. "
            "The time spent before collecting results was %s seconds.",
            self.node_manager_uuid,
            time.time() - start_time,
        )
        # Return results
        return (
            node_eval_loss,
            int(node_eval_samples),
            node_eval_metrics,
        )
