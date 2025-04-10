"""Centralised training script for LLMFoundry models.

Slightly adapted from the original https://github.com/mosaicml/llm-foundry/blob/25599294c942cfed2c6f8329e14791e4a2f91539/scripts/train/train.py
Copyright 2022 MosaicML LLM Foundry authors
SPDX-License-Identifier: Apache-2.0
"""

import gc
import os
from logging import INFO
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
from composer import Trainer
from flwr.common import NDArray, log
from llmfoundry.callbacks import EvalGauntlet
from omegaconf import OmegaConf

from photon.clients.configs import CentralizedConfig
from photon.clients.llm_client_functions import (
    get_parameters_from_state,
    get_trainer_object,
)
from photon.server.s3_utils import load_pretrained_model_from_path
from photon.utils import (
    dump_model_parameters_to_file,
    get_wte_parameters_from_trainer,
    set_wte_parameters_to_trainer,
)

if TYPE_CHECKING:
    from photon.conf.base_schema import BaseConfig


def main() -> Trainer:  # noqa: C901
    """Execute centralized training.

    This function sets up the training environment, loads configurations, initializes
    the trainer, optionally loads pretrained models, performs evaluation if requested,
    and starts the training process. It also handles dumping model parameters to files
    before and after training.

    Returns
    -------
        Trainer
            The initialized trainer object.

    Raises
    ------
    ValueError
        If the environmental variable PHOTON_SAVE_PATH is not set.

    """
    # Get the environmental variable for the dump folder
    save_path = os.environ.get("PHOTON_SAVE_PATH", "")
    # Raise an error if the environmental variable is not set
    if not save_path:
        msg = "The environmental variable PHOTON_SAVE_PATH is not set."
        raise ValueError(msg)
    # Load the configuration from the config file
    cfg_ = cast("BaseConfig", OmegaConf.load(save_path + "/config.yaml"))
    # Resolve all interpolation variables as early as possible
    OmegaConf.resolve(cfg_)
    OmegaConf.set_struct(cfg_, value=False)
    cfg = cfg_.llm_config
    # Resolve all interpolation variables as early as possible
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, value=False)
    # Creating ClientConfig object
    client_config = CentralizedConfig(
        allow_unigram_metrics_failures=cfg_.fl.allow_unigram_metrics_failures,
        resize_vocab=cfg_.fl.resize_vocab,
        split_eval=cfg_.centralized.split_eval,
        set_trainer_params_filter_keys=cfg_.fl.set_trainer_params_filter_keys,
        set_trainer_key_to_filter=cfg_.fl.set_trainer_key_to_filter,
        use_unigram_metrics=cfg_.fl.use_unigram_metrics,
        s3_comm_config=cfg_.s3_comm_config,
    )

    # NOTE: The cid passed her is use to appoint the position for the stream used for
    # creating the streaming dataset object
    log(
        INFO,
        "Creating trainer object using stream_id: %s...",
        cfg_.centralized.stream_id,
    )
    trainer, train_config, *_ = get_trainer_object(
        client_config=client_config,
        cfg=cfg,
        cid=cfg_.centralized.stream_id,
        log_name="_centralised",
        split_eval=client_config.split_eval,
    )
    torch.cuda.empty_cache()
    gc.collect()

    wte_parameters: NDArray | None = None
    if cfg_.wte_parameters_path:
        load_pretrained_model_from_path(
            trainer=trainer,
            pretrained_model_path=cfg_.wte_parameters_path,
            run_uuid=cfg_.run_uuid,
            s3_comm_config=cfg_.s3_comm_config,
        )
        wte_parameters = get_wte_parameters_from_trainer(trainer)

    if cfg_.pretrained_model_path:
        load_pretrained_model_from_path(
            trainer=trainer,
            pretrained_model_path=cfg_.pretrained_model_path,
            run_uuid=cfg_.run_uuid,
            s3_comm_config=cfg_.s3_comm_config,
        )

    if wte_parameters is not None:
        set_wte_parameters_to_trainer(trainer, wte_parameters)

    # Eval first if requested
    if train_config.eval_first:
        trainer.eval()
        eval_gauntlet_callback: EvalGauntlet | None = None
        for callback in trainer.state.callbacks:
            if isinstance(callback, EvalGauntlet):
                eval_gauntlet_callback = callback
        if eval_gauntlet_callback is not None:
            assert isinstance(eval_gauntlet_callback, EvalGauntlet)
            composite_scores = eval_gauntlet_callback.eval_after_all(
                trainer.state,
                trainer.logger,
            )
            log(
                INFO,
                "Evaluated model with the Gauntlet before training: %s",
                composite_scores,
            )

    # Dump model parameters to file
    if cfg_.centralized.store_init_model:
        # Get model parameters from trainer object
        model_parameters = get_parameters_from_state({}, trainer)
        # Get number of steps executed
        n_steps = trainer.state.timestamp.batch.value
        # Dump the compressed model parameters to file
        dump_model_parameters_to_file(
            model_parameters=model_parameters,
            file_path=Path(f"{cfg_.run_uuid}-{n_steps}-checkpoint.npz"),
        )
        log(INFO, "Dumped initial model parameters to file.")

    if not cfg_.centralized.eval_only:
        log(INFO, "Starting training...")
        trainer.fit(reset_time=cfg_.centralized.reset_timestamp)

    # Dump model parameters to file
    if cfg_.centralized.store_final_model:
        # Get model parameters from trainer object
        model_parameters = get_parameters_from_state({}, trainer)
        # Get number of steps executed
        n_steps = trainer.state.timestamp.batch.value
        # Dump the compressed model parameters to file
        dump_model_parameters_to_file(
            model_parameters=model_parameters,
            file_path=Path(f"{cfg_.run_uuid}-{n_steps}-checkpoint.npz"),
        )
        log(INFO, "Dumped final model parameters to file.")

    log(INFO, "Done.")
    return trainer


if __name__ == "__main__":
    main()
