"""Provides the internal functions used by the LLM client.

This module contains utility functions that handle local training and evaluation for
Large Language Model (LLM) clients in a federated learning setup. It sets up and
configures Trainers, loads checkpoints, applies parameters, runs training or evaluation,
gathers metrics, and cleans up resources.
"""

import copy
import gc
import os
import time
from logging import DEBUG, ERROR
from typing import Any

import torch

# NOTE: We need this if we want to compile the model because the attention
# implementation in the MPT code is dispatched using a dictionary that raises:
# `AssertionError: Dict types must use ConstDictVariable.`
import torch._dynamo
from composer import Trainer
from flwr.common.logger import log
from flwr.common.recordset_compat import ConfigsRecord
from flwr.common.typing import NDArrays, Scalar
from omegaconf import DictConfig

from photon.clients.configs import EvaluateConfig, FitConfig
from photon.clients.llm_config_functions import get_train_config
from photon.clients.trainer_utils import (
    get_trainer_mutables_from_config,
    get_trainer_object,
    load_trainer_checkpoint,
    set_mutables_trainer,
    trainer_clean_up,
)
from photon.clients.utils import (
    get_client_state_struct,
    manipulate_pre_training_ndarrays,
    post_process_client_result,
    set_initial_config_from_fit_config,
    streaming_shms_clean_up,
)
from photon.utils import (
    get_parameters_from_state,
    parameters_checker,
    set_trainer_params_from_ndarrays,
)

torch._dynamo.config.suppress_errors = True  # type: ignore[reportAttributeAccessIssue]  # noqa: SLF001


def llm_fit(  # noqa: PLR0914
    external_trainer: Trainer | None,
    payload: NDArrays,
    config: ConfigsRecord,
    llm_config: DictConfig,
    cid: int | str,
) -> tuple[NDArrays, int, dict[str, Scalar] | dict[Any, Any], Trainer]:
    """Perform local training on the LLM client.

    This function sets up the model and trainer environment, handles checkpoint
    loading if available, sets model parameters, performs local training, and
    returns the updated model parameters, the number of trained samples, training
    metrics, and the final Trainer object.

    Parameters
    ----------
    external_trainer : Trainer | None
        An optional external Trainer object. If provided, it will be used instead
        of creating a new Trainer.
    payload : NDArrays
        The model parameters to be loaded into the trainer before training.
    config : ConfigsRecord
        Configuration information for federated training.
    llm_config : DictConfig
        The local LLM configuration specifying trainer and model details.
    cid : int | str
        The client ID.

    Returns
    -------
    tuple[NDArrays, int, dict[str, Scalar] | dict[Any, Any], Trainer]
        A tuple containing:
        - The updated model parameters after training.
        - The number of samples used during training.
        - A dictionary of training metrics.
        - The final Trainer object.

    Raises
    ------
    ValueError
        If the server steps cumulative is None and we want to skip this iteration

    """
    try:
        fit_config = FitConfig(**config)  # type: ignore[reportArgumentType,arg-type]
        client_state_struct, start_time = (
            get_client_state_struct(fit_config, cid),
            time.time_ns(),
        )
        train_metrics: dict[str, Scalar] = {}

        skip_iteration, checkpoint_exists, server_steps_cumulative = (
            set_initial_config_from_fit_config(fit_config, llm_config, cid)
        )
        dummy_config = copy.deepcopy(llm_config)

        if external_trainer is not None:
            log(DEBUG, "External trainer object exists.")
            train_cfg, device, logged_cfg, icl_tasks_config_dict = get_train_config(
                cfg=llm_config,
                cid=cid,
            )
            trainer_mutable_attributes = get_trainer_mutables_from_config(
                trainer=external_trainer,
                train_cfg=train_cfg,
                client_config=fit_config,
                icl_tasks_config_dict=icl_tasks_config_dict,
                device=device,
                logged_cfg=logged_cfg,
            )
            set_mutables_trainer(
                external_trainer,
                trainer_mutable_attributes,
                client_config=fit_config,
            )
            if checkpoint_exists:
                log(DEBUG, "Checkpoint exists.")
                load_trainer_checkpoint(external_trainer, train_cfg)
        else:
            log(DEBUG, "External trainer object doesn't exit.")
            (
                external_trainer,
                train_cfg,
                _,
            ) = get_trainer_object(
                cfg=llm_config,
                cid=cid,
                client_config=fit_config,
            )

        initial_trainer_parameters = manipulate_pre_training_ndarrays(
            payload=payload,
            trainer=external_trainer,
            configs=(fit_config, dummy_config),
            cid=cid,
            client_state_struct=client_state_struct,
            train_metrics=train_metrics,
        )

        # NOTE: Extract parameters
        # if our payload also contains momenta
        parameters = (
            payload
            if not fit_config.aggregate_momenta
            else payload[: len(initial_trainer_parameters)]
        )

        log(DEBUG, "Trainer object obtained.")
        train_metrics["client/fit_init_time"] = (time.time_ns() - start_time) * 1e-9

        if not skip_iteration:
            client_timestamp = external_trainer.state.timestamp.copy()
            if client_timestamp.batch == 0:
                client_timestamp.load_state_dict(client_state_struct.local_timestamp)

            if server_steps_cumulative is None:
                msg = "Server steps cumulative is None and we want to skip iteration."
                raise ValueError(msg)  # noqa: TRY301

            external_trainer.state.timestamp = client_timestamp.copy(
                batch=server_steps_cumulative if not fit_config.reset_timestamp else 0,
            )

            start_time = time.time_ns()
            set_trainer_params_from_ndarrays(
                parameters,
                external_trainer,
                fit_config.set_trainer_key_to_filter,
                filter_keys=fit_config.set_trainer_params_filter_keys,
            )

            # NOTE: We removed here the parameter checkers for efficiency reasons.
            # Re-add them if needed

            train_metrics["client/fit_set_parameters_time"] = (
                time.time_ns() - start_time
            ) * 1e-9

            if train_cfg.eval_first:
                start_time = time.time_ns()
                external_trainer.eval()
                train_metrics["client/fit_pre_eval_time"] = (
                    time.time_ns() - start_time
                ) * 1e-9
                train_metrics.update(
                    {
                        f"PrePersonalization{k}": v.detach().cpu().item()  # type: ignore[attr-defined]
                        for k, v in external_trainer.state.eval_metric_values.items()
                    },
                )

            try:
                start_time = time.time_ns()
                external_trainer.fit(
                    duration=0 if skip_iteration else llm_config["local_steps"],
                )
                train_metrics["client/fit_time"] = (time.time_ns() - start_time) * 1e-9
            except Exception as e:
                log(ERROR, "llm_fit::trainer.fit", exc_info=e, stack_info=True)
                raise

        model_parameters, n_samples_trained = post_process_client_result(
            train_metrics=train_metrics,
            trainer=external_trainer,
            initial_parameters=parameters,
            client_state_struct=client_state_struct,
            llm_config=llm_config,
            cid=cid,
            fit_config=fit_config,
        )

    except Exception as e:
        log(ERROR, "Error in llm_fit function", exc_info=e, stack_info=True)
        raise
    else:
        return model_parameters, n_samples_trained, train_metrics, external_trainer


def llm_eval(
    external_trainer: Trainer | None,
    payload: NDArrays,
    config: ConfigsRecord,
    llm_config: DictConfig,
) -> tuple[float, int, dict[str, Scalar], Trainer]:
    """Perform local evaluation on the LLM client.

    This function sets up the trainer environment for evaluation, applies model
    parameters, runs the evaluation, collects evaluation metrics, and cleans up
    resources.

    Parameters
    ----------
    external_trainer : Trainer | None
        An optional external Trainer object. If provided, it will be used instead
        of creating a new Trainer.
    payload : NDArrays
        The model parameters to be used for evaluation.
    config : ConfigsRecord
        Configuration information for federated evaluation.
    llm_config : DictConfig
        The local LLM configuration specifying trainer and model details.

    Returns
    -------
    tuple[float, int, dict[str, Scalar], Trainer]
        A tuple containing:
        - The (unused) loss value set to 0.0 in this function.
        - The number of samples evaluated.
        - A dictionary of evaluation metrics.
        - The final Trainer object.

    """
    start_time = time.time_ns()
    num_samples = 0
    eval_metrics: dict[str, Scalar] = {}

    llm_config.autoresume = False  # type: ignore[union-attr]
    llm_config.save_folder = None  # type: ignore[union-attr]
    llm_config.load_path = None  # type: ignore[union-attr]
    llm_config.loggers = None  # type: ignore[union-attr]

    client_eval_config = EvaluateConfig(**dict(config))  # type: ignore[reportArgumentType,arg-type]

    (
        external_trainer,
        _,
        _,
    ) = get_trainer_object(
        cfg=llm_config,
        cid=None,
        client_config=client_eval_config,
    )

    initial_trainer_parameters = get_parameters_from_state({}, external_trainer)

    # NOTE: Extract only the parameters
    # for eval, no momenta needed
    # since we are not training
    parameters = (
        payload
        if not client_eval_config.aggregate_momenta
        else payload[: len(initial_trainer_parameters)]
    )

    parameters_checker(initial_trainer_parameters, parameters, is_equal=False)

    eval_metrics["client/eval_init_time"] = (time.time_ns() - start_time) * 1e-9

    start_time = time.time_ns()
    set_trainer_params_from_ndarrays(
        parameters,
        external_trainer,
        client_eval_config.set_trainer_key_to_filter,
        filter_keys=client_eval_config.set_trainer_params_filter_keys,
    )

    current_trainer_parameters = get_parameters_from_state({}, external_trainer)
    parameters_checker(
        current_trainer_parameters,
        initial_trainer_parameters,
        is_equal=False,
    )
    parameters_checker(
        current_trainer_parameters,
        parameters,
        is_equal=True,
    )

    gc.collect()
    torch.cuda.empty_cache()
    eval_metrics["client/eval_set_parameters_time"] = (
        time.time_ns() - start_time
    ) * 1e-9

    start_time = time.time_ns()
    external_trainer.eval()
    eval_metrics["client/eval_time"] = (time.time_ns() - start_time) * 1e-9
    start_time = time.time_ns()
    if int(os.getenv("LOCAL_RANK", "")) == 0:
        num_samples = (
            external_trainer.state.eval_timestamp._sample.value  # noqa: SLF001
        )
        eval_metrics.update(
            {
                "Val" + k: v.detach().cpu().item()  # type: ignore[attr-defined]
                for k, v in external_trainer.state.eval_metric_values.items()
            },
        )
        eval_metrics["client/eval_metrics_collection_time"] = (
            time.time_ns() - start_time
        ) * 1e-9

    start_time = time.time_ns()
    trainer_clean_up(trainer=external_trainer)
    streaming_shms_clean_up()
    if int(os.getenv("LOCAL_RANK", "")) == 0:
        eval_metrics["client/eval_trainer_closing_time"] = (
            time.time_ns() - start_time
        ) * 1e-9

    return 0.0, num_samples, eval_metrics, external_trainer
