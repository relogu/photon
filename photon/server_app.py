"""Implementation of the Flower's ServerApp for orchestrating federate learning."""

import copy
import os
import random
import time
import timeit
import uuid
import warnings
from logging import DEBUG, INFO
from queue import Queue
from typing import TYPE_CHECKING, cast

import flwr as fl
import ray
import wandb
from flwr.common import (
    Context,
)
from flwr.common.logger import log, update_console_handler
from flwr.server import Driver
from omegaconf import OmegaConf

from photon.clients.configs import (
    get_photon_evaluate_config_fn,
    get_photon_fit_config_fn,
)
from photon.server.broadcast_utils import broadcast_parameters_to_nodes
from photon.server.evaluate_utils import evaluate_round
from photon.server.fit_utils import fit_round
from photon.server.init_utils import (
    get_centralized_run_parameters,
    initialize_round,
    resume_from_round,
)
from photon.server.s3_utils import (
    cleanup_checkpoints,
    import_checkpoints,
    upload_server_checkpoint,
)
from photon.server.server_util import (
    wait_for_nodes_to_connect,
)
from photon.strategy.dispatcher import dispatch_strategy
from photon.strategy.utils import initialize_strategy
from photon.utils import (
    create_remote_up_down,
    custom_ray_garbage_collector,
    wandb_init,
)

if TYPE_CHECKING:
    from composer.loggers import RemoteUploaderDownloader

    from photon.conf.base_schema import BaseConfig

# Fix the logger
update_console_handler(level=DEBUG, colored=False, timestamps=True)
# Filter user warning from configuration of MPT
warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    message=("If not using a Prefix Language Model*"),
    append=True,
)
# Filter deprecation warning from pkg_resources
warnings.filterwarnings(
    action="ignore",
    category=DeprecationWarning,
    message=("Deprecated call to *"),
    append=True,
)
warnings.filterwarnings(
    action="ignore",
    category=DeprecationWarning,
    message=("pkg_resources is deprecated*"),
    append=True,
)

# Run via `flower-server-app server:app`
app = fl.server.ServerApp()


@app.main()
def main(  # noqa: PLR0914, PLR0915, PLR0912, C901
    driver: Driver,
    context: Context,  # noqa: ARG001
) -> None:
    """Implement the main function for the Flower ServerApp.

    Parameters
    ----------
    driver : fl.server.Driver
        The driver object for the server.
    context : fl.common.Context
        The context object for the server.

    Raises
    ------
    ValueError
        If the environmental variable `PHOTON_SAVE_PATH` is not set.

    """
    start_up_time = timeit.default_timer()
    # Creating Queue for custom Ray garbage collector
    ray_garbage_queue: Queue[ray.ObjectRef] = Queue()
    # Initialize Ray
    ray.init("auto")
    # Get the environmental variable for the dump folder
    save_path = os.environ.get("PHOTON_SAVE_PATH", "")
    # Raise an error if the environmental variable is not set
    if not save_path:
        msg = "PHOTON_SAVE_PATH is not set."
        raise ValueError(msg)
    # Load the configuration from the config file
    cfg = cast("BaseConfig", OmegaConf.load(save_path + "/config.yaml"))
    log(INFO, "Initializing Photon Server")

    # Get FL setting parameters from the config file
    n_total_clients = cfg.fl.n_total_clients
    n_clients_per_round = cfg.fl.n_clients_per_round
    num_rounds = cfg.fl.n_rounds
    # Instantiate a PRNG
    rng = random.Random(cfg.seed)  # noqa: S311
    # Get Photon parameters
    n_nodes = cfg.photon.n_nodes

    strategy = dispatch_strategy(
        cfg,
    )
    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    with (
        wandb_init(  # type: ignore[union-attr,misc]
            cfg.use_wandb,
            **cfg.wandb.setup,  # type: ignore[reportCallIssue]
            settings=wandb.Settings(start_method="thread"),  # type: ignore[arg-type]
            config=wandb_config,  # type: ignore[arg-type]
        ) as wandb_run,
        custom_ray_garbage_collector(
            garbage_queue=ray_garbage_queue,
            list_of_threads=[],
        ),
    ):
        log(INFO, f"Wandb run initialized: {wandb_run}")
        # Create RemoteUploaderDownloader
        remote_up_down: RemoteUploaderDownloader | None = None
        if cfg.photon.checkpoint or cfg.photon.comm_stack.s3:
            remote_up_down = create_remote_up_down(
                bucket_name=cfg.s3_comm_config.bucket_name,
                prefix=f"{cfg.run_uuid}/server",
                run_uuid=cfg.run_uuid,
                num_attempts=cfg.s3_comm_config.num_attempts,
                client_config=OmegaConf.to_container(
                    cfg.s3_comm_config.backend_kwargs.client_config,
                ),  # type: ignore[reportArgumentType, arg-type]
            )

        # Import another experiment checkpoints for restoration
        if cfg.photon.restore_run_uuid is not None:
            if remote_up_down is None:
                msg = "Cannot restore without a RemoteUploaderDownloader object"
                raise ValueError(msg)
            import_checkpoints(
                cfg=cfg,
                remote_up_down=remote_up_down,
                strategy=strategy,
            )

        # Resume experiment from a previously saved checkpoint
        if cfg.photon.resume_round is not None:
            if cfg.photon.checkpoint is None:  # type: ignore[reportUnnecessaryComparison]
                msg = "Cannot resume if `cfg.photon.checkpoint` is None"
                raise ValueError(msg)
            if remote_up_down is None:
                msg = "Cannot resume without a RemoteUploaderDownloader object"
                raise ValueError(msg)
            (
                parameters,
                history,
                start_round,
                time_offset,
                server_steps_cumulative,
                client_state,
                momentum_vector,
                second_momentum_vector,
            ) = resume_from_round(cfg, remote_up_down, strategy=strategy)
            # Loop over the PRNG to get to the correct round
            for _ in range(start_round):
                sampled_clients = rng.sample(
                    range(n_total_clients),
                    n_clients_per_round,
                )
            sampled_clients = []
        elif cfg.photon.restore_cent_run_uuid is not None:
            if remote_up_down is None:
                msg = "Cannot restore without a RemoteUploaderDownloader object"
                raise ValueError(msg)
            parameters = get_centralized_run_parameters(copy.deepcopy(cfg))
            (
                parameters,
                history,
                start_round,
                time_offset,
                server_steps_cumulative,
                client_state,
                momentum_vector,
                second_momentum_vector,
            ) = initialize_round(cfg, remote_up_down, parameters=parameters)
        else:
            (
                parameters,
                history,
                start_round,
                time_offset,
                server_steps_cumulative,
                client_state,
                momentum_vector,
                second_momentum_vector,
            ) = initialize_round(cfg, remote_up_down)

        # Initialize the strategy
        initialize_strategy(
            strategy=strategy,
            parameters=parameters,
            momentum_vector=momentum_vector,
            second_momentum_vector=second_momentum_vector,
        )

        # Wait for the minimum number of nodes to connect
        wait_for_nodes_to_connect(driver, n_nodes)

        log(
            INFO,
            "Start-up time for the server is %s",
            timeit.default_timer() - start_up_time,
        )
        # Run federated learning for number of rounds
        log(INFO, "FL starting from round %s", start_round + 1)
        start_time = timeit.default_timer()

        # Broadcast model parameters to all NodeManagers
        broadcast_time = time.time_ns()
        all_node_ids = driver.get_node_ids()
        server_uuid = cfg.run_uuid + "-server-" + str(uuid.uuid4())
        list_of_ray_object_refs = broadcast_parameters_to_nodes(
            driver=driver,
            parameters=parameters,
            node_ids=all_node_ids,
            current_round=start_round,
            remote_uploader_downloader=remote_up_down,
            comm_stack=cfg.photon.comm_stack,
            server_uuid=server_uuid,
        )
        time_to_broadcast = time.time_ns() - broadcast_time

        if cfg.fl.eval_period:
            # Launch the evaluate process for the starting round
            sampled_clients = [0]
            history = evaluate_round(
                driver_and_nodes=(driver, all_node_ids),
                evaluate_config_fn=get_photon_evaluate_config_fn(cfg),
                clients_and_states=(sampled_clients, client_state),
                cfg=cfg,
                server_state=(
                    strategy,
                    history,
                    start_round,
                    server_steps_cumulative,
                ),
            )
        history.add_metrics_centralized(
            server_round=start_round + 1,
            metrics={"server/broadcast_pre_time": time_to_broadcast * 1e-9},
        )
        # Nullify assignments
        sampled_clients = []

        # Federated learning loop
        for current_round in range(start_round + 1, num_rounds + 1):
            start_round_time = time.time_ns()
            log(DEBUG, f"Commencing server round {current_round}")

            # Check NodeManagers health
            first_check_nm_time = time.time_ns()
            all_node_ids = driver.get_node_ids()
            history.add_metrics_centralized(
                server_round=current_round,
                metrics={
                    "server/first_check_nm_time": (time.time_ns() - first_check_nm_time)
                    * 1e-9,
                },
            )

            # List of sampled Client IDs in this round
            sampled_clients = rng.sample(range(n_total_clients), n_clients_per_round)
            log(DEBUG, f"Sampled {len(sampled_clients)} Client IDs: {sampled_clients}")

            # Launch the federated fit process
            (
                parameters,
                client_state,
                server_steps_cumulative,
                history,
            ) = fit_round(
                sampled_clients=sampled_clients,
                all_node_ids=all_node_ids,
                driver=driver,
                fit_config_fn=get_photon_fit_config_fn(cfg),
                current_round=current_round,
                client_state=client_state,
                server_steps_cumulative=server_steps_cumulative,
                cfg=cfg,
                strategy=strategy,
                remote_up_down=remote_up_down,
                history=history,
                parameters=parameters,
            )
            # Nullify sampled clients
            sampled_clients = []

            # Broadcast model parameters to all NodeManagers
            broadcast_time = time.time_ns()
            if list_of_ray_object_refs:
                for ray_objec_ref_to_collect in list_of_ray_object_refs:
                    ray_garbage_queue.put(ray_objec_ref_to_collect)
            list_of_ray_object_refs = None
            list_of_ray_object_refs = broadcast_parameters_to_nodes(
                driver=driver,
                parameters=parameters,
                node_ids=all_node_ids,
                current_round=current_round,
                remote_uploader_downloader=remote_up_down,
                comm_stack=cfg.photon.comm_stack,
                server_uuid=server_uuid,
            )
            history.add_metrics_centralized(
                server_round=current_round,
                metrics={
                    "server/broadcast_post_time": (time.time_ns() - broadcast_time)
                    * 1e-9,
                },
            )

            # Check for changes in connected NodeManagers
            second_check_nm_time = time.time_ns()
            all_node_ids = driver.get_node_ids()
            history.add_metrics_centralized(
                server_round=current_round,
                metrics={
                    "server/second_check_nm_time": (
                        time.time_ns() - second_check_nm_time
                    )
                    * 1e-9,
                },
            )
            if (
                cfg.fl.eval_period is not None
                and current_round % cfg.fl.eval_period == 0
            ):
                # Launch the evaluate process
                sampled_clients = [0]
                history = evaluate_round(
                    driver_and_nodes=(driver, all_node_ids),
                    clients_and_states=(sampled_clients, client_state),
                    evaluate_config_fn=get_photon_evaluate_config_fn(cfg),
                    cfg=cfg,
                    server_state=(
                        strategy,
                        history,
                        current_round,
                        server_steps_cumulative,
                    ),
                )
            # Nullify assignments
            sampled_clients = []

            # Save the checkpoint to S3 Object Store
            if cfg.photon.checkpoint or cfg.photon.comm_stack.s3:
                if remote_up_down is None:
                    msg = "Cannot checkpoint without a RemoteUploaderDownloader object"
                    raise ValueError(msg)

                upload_server_checkpoint(
                    parameters=parameters,
                    server_state=(
                        history,
                        current_round,
                        time_offset,
                        server_steps_cumulative,
                    ),
                    momenta=(momentum_vector, second_momentum_vector),
                    client_state=client_state,
                    remote_up_down=remote_up_down,
                )

            # Log the time taken for the round
            history.add_metrics_centralized(
                server_round=current_round,
                metrics={
                    "server/round_time": (time.time_ns() - start_round_time) * 1e-9,
                },
            )
            # Clean up checkpoints if asked to
            if cfg.cleanup_checkpoints_per_round:
                cleanup_checkpoints(cfg.run_uuid, strategy.state_keys, end_idx=-1)

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time + time_offset
        log(INFO, "FL finished in %s", elapsed)

        log(DEBUG, "app_fit: losses_distributed %s", str(history.losses_distributed))
        log(
            DEBUG,
            "app_fit: metrics_distributed_fit %s",
            str(history.metrics_distributed_fit),
        )
        log(DEBUG, "app_fit: metrics_distributed %s", str(history.metrics_distributed))
        log(DEBUG, "app_fit: losses_centralized %s", str(history.losses_centralized))
        log(DEBUG, "app_fit: metrics_centralized %s", str(history.metrics_centralized))

        # Clean up checkpoints if asked to
        if cfg.cleanup_checkpoints:
            cleanup_checkpoints(cfg.run_uuid, strategy.state_keys, end_idx=None)
