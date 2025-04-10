"""Base configuration schema."""

from enum import Enum, auto
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig
from pydantic.dataclasses import dataclass


@dataclass(config={"arbitrary_types_allowed": True})
class CommStack(DictConfig):
    """Communication stack configuration.

    Attributes
    ----------
    shm: bool = MISSING
        Whether to use SharedMemory communication
    s3: bool = MISSING
        Whether to use S3 communication
    ray: bool = MISSING
        Whether to use Ray communication

    """

    s3: bool = MISSING
    shm: bool = MISSING
    ray: bool = MISSING


@dataclass(config={"arbitrary_types_allowed": True})
class Centralized(DictConfig):
    """Centralized configuration.

    Attributes
    ----------
    store_init_model: bool = MISSING
        Whether to store the initial model
    store_final_model: bool = MISSING
        Whether to store the final model
    stream_id: str | None = MISSING
        Stream id to pass to the data configuration
    eval_only: bool = MISSING
        Whether to only execute the evaluation
    split_eval: bool = MISSING
        Whether to report performance separately on each stream
    reset_timestamp: bool = MISSING
        Whether to reset the timestamp

    """

    store_init_model: bool = MISSING
    store_final_model: bool = MISSING
    stream_id: str | None = MISSING
    eval_only: bool = MISSING
    split_eval: bool = MISSING
    reset_timestamp: bool = MISSING


@dataclass(config={"arbitrary_types_allowed": True})
class Photon(DictConfig):
    """Photon configuration.

    Attributes
    ----------
    n_nodes: int = MISSING
        Number of nodes in the cluster
    saving_path: str = MISSING
        Path to save the models
    refresh_period: int = MISSING
        Refresh period for the client workers
    checkpoint: bool = MISSING
        Whether to checkpoint the model
    restore_run_uuid: str = MISSING
        Run UUID to restore the model
    resume_round: int | None = MISSING
        Round to resume from, None implies start anew
        negative indices are counted from the last round
    restore_cent_run_uuid: str = MISSING
        Run UUID to restore the centralized model
    restore_cent_run_batches: int = MISSING
        Number of batches to restore from the centralized model
    copy_client_checkpoints: bool = MISSING
        Whether to copy the client checkpoints

    """

    n_nodes: int = MISSING
    saving_path: str = MISSING
    refresh_period: int = MISSING
    checkpoint: bool = MISSING
    restore_run_uuid: str | None = MISSING
    resume_round: int | None = MISSING
    restore_cent_run_uuid: str | None = MISSING
    restore_cent_run_batches: int | None = MISSING
    copy_client_checkpoints: bool = MISSING
    comm_stack: CommStack = MISSING


class StrategyName(str, Enum):
    """Strategy type."""

    @staticmethod
    def _generate_next_value_(
        name: str,
        start: int,  # noqa: ARG004
        count: int,  # noqa: ARG004
        last_values: list[int],  # noqa: ARG004
    ) -> str:
        """Generate the next value.

        Replacement for StrEnum to support python 3.10

        Parameters
        ----------
        name: str
            Name of the strategy
        start: int
            Start value
        count: int
            Count value
        last_values: list[int]
            List of last values

        Returns
        -------
        str
            Lowercase name

        """
        return name.lower()

    NESTOROV = auto()
    FEDMOM = auto()
    FEDAVG = auto()
    FEDYOGI = auto()
    FEDADAM = auto()


class StrategyKWArgs(dict[str, Any], DictConfig):  # type: ignore[reportIncompatibleMethodOverride,misc]
    """StrategyKWArgs configuration."""


@dataclass(config={"arbitrary_types_allowed": True})
class FL(DictConfig):
    """Federated learning configuration.

    Attributes
    ----------
    n_total_clients: int = MISSING
        Total number of clients
    n_clients_per_round: int = MISSING
        Number of clients per round
    n_rounds: int = MISSING
        Number of rounds
    reset_checkpoint: bool = MISSING
        Whether to reset the checkpoint
    reset_optimizer: bool = MISSING
        Whether to reset the optimizer
    reset_dataset_state: bool = MISSING
        Whether to reset the dataset state
    resize_vocab: int | None = MISSING
        Resize the vocabulary
    n_local_epochs: int = MISSING
        Number of local epochs
    n_local_steps: int = MISSING
        Number of local steps
    use_unigram_metrics: bool = MISSING
        Whether to use unigram metrics
    allow_unigram_metrics_failures: bool = MISSING
        Whether to allow missing frequency dictionaries
    random_layers: list[str] = MISSING
        List of random layers
    random_init_freq: int = MISSING
        Random initialization frequency, 0 means never
    truly_random_init: bool = MISSING
        Whether to truly random initialization
    personalized_layers: list[str] = MISSING
        List of personalized layers
    frozen_layers : list[str] | None
        Specify layers to freeze.
    unfrozen_layers : list[str] | None
        Specify layers to keep unfrozen and freeze everything else.
    split_eval : bool
    ignore_failed_rounds: bool = MISSING
        Whether to ignore failed rounds
    accept_failures_cnt: int = MISSING
        Number of acceptable failures
    eval_period: int | None = MISSING
        Federated evaluation period, None means no evaluation is executed
    split_eval: bool = MISSING
        Whether to report performance separately on each stream
    strategy_name: StrategyName = MISSING
        Strategy name
    strategy_kwargs: StrategyKWArgs = MISSING
        Strategy kwargs
    set_trainer_params_filter_keys: bool = MISSING
        If to filter out parameter names without the following key
    set_trainer_key_to_filter: str = MISSING
        Key to filter
    use_noise_scale_metric: bool = MISSING
        Whether to use noise scale metric
    noise_scale_beta: float = MISSING
        Beta used to compute EMA for noise scale

    """

    n_total_clients: int = MISSING
    n_clients_per_round: int = MISSING
    n_rounds: int = MISSING
    reset_checkpoint: bool = MISSING
    reset_optimizer: bool = MISSING
    reset_dataset_state: bool = MISSING
    reset_timestamp: bool = MISSING
    resize_vocab: int | None = MISSING
    n_local_epochs: int = MISSING
    n_local_steps: int = MISSING
    use_unigram_metrics: bool = MISSING
    allow_unigram_metrics_failures: bool = MISSING
    random_layers: list[str] = MISSING
    random_init_freq: int = MISSING
    truly_random_init: bool = MISSING
    personalized_layers: list[str] = MISSING
    frozen_layers: list[str] | None = MISSING
    unfrozen_layers: list[str] | None = MISSING

    ignore_failed_rounds: bool = MISSING
    accept_failures_cnt: int = MISSING
    eval_period: int | None = MISSING
    split_eval: bool = MISSING

    strategy_name: StrategyName = MISSING
    strategy_kwargs: StrategyKWArgs = MISSING
    set_trainer_params_filter_keys: bool = MISSING
    set_trainer_key_to_filter: str = MISSING

    use_noise_scale_metric: bool = MISSING
    noise_scale_beta: float = MISSING
    aggregate_momenta: bool = MISSING


@dataclass(config={"arbitrary_types_allowed": True})
class ClientConfig(DictConfig):
    """Client configuration."""

    connect_timeout: int = MISSING
    read_timeout: int = MISSING


@dataclass(config={"arbitrary_types_allowed": True})
class BackendKwargs(DictConfig):
    """Backend configuration.

    Attributes
    ----------
    client_config: DictConfig
        Configuration for the client

    """

    client_config: ClientConfig = MISSING


@dataclass(config={"arbitrary_types_allowed": True})
class S3CommConfig(DictConfig):
    """S3 communication configuration.

    Attributes
    ----------
    bucket_name: str = MISSING
        Name of the S3 bucket
    num_attempts: int = MISSING
        Number of attempts
    backend_kwargs: BackendKwargs
        Backend configuration

    """

    bucket_name: str = MISSING
    num_attempts: int = MISSING
    backend_kwargs: BackendKwargs = MISSING


@dataclass(config={"arbitrary_types_allowed": True})
class WandbSetup(DictConfig):
    """Wand setup configuration.

    Attributes
    ----------
    project: str = MISSING
        Name of the project
    group: str = MISSING
        Name of the group
    tags: list[str] = MISSING
        List of tags
    entity: str = MISSING
        Name of the entity
    mode: str = MISSING
        Mode of the run: "online", "offline"
    name: str = MISSING
        Name of the run, {Config.run_uuid}
    resume: str = MISSING
        Weather to allow resumption
    id: str = MISSING
        ID of the run, {Config.run_uuid}
    allow_val_change: bool = MISSING
        Allows changing the value of the config when resuming

    """

    project: str = MISSING
    group: str = MISSING
    tags: list[str] = MISSING
    entity: str | None = MISSING
    mode: str = MISSING
    name: str = MISSING
    resume: str = MISSING
    id: str = MISSING
    allow_val_change: bool = MISSING


@dataclass(config={"arbitrary_types_allowed": True})
class Wandb(DictConfig):
    """Wandb configuration.

    Attributes
    ----------
    setup: WandbSetup
        Wandb setup configuration

    """

    setup: WandbSetup = MISSING


class Dataset(dict[str, Any], DictConfig):  # type: ignore[reportIncompatibleMethodOverride,misc]
    """Dataset configuration."""


class LLMConfig(dict[str, Any], DictConfig):  # type: ignore[reportIncompatibleMethodOverride,misc]
    """LLM configuration."""


@dataclass(config={"arbitrary_types_allowed": True})
class BaseConfig(DictConfig):
    """Base configuration.

    Attributes
    ----------
    run_uuid: str = MISSING
        Run UUID
    seed: int = MISSING
        Seed
    pretrained_model_path: str = MISSING
        Path to the pretrained model
    wte_parameters_path: str = MISSING
        Path to the model from which to take the WTE parameters
    centralized: Centralized = MISSING
        Centralized configuration
    photon: Photon
        Photon configuration
    fl: FL
        Federated learning configuration
    s3_comm_config: S3CommConfig
        S3 communication configuration
    use_wandb: bool = MISSING
        Whether to use Wandb
    cleanup_checkpoints: bool = MISSING
        Whether to clean up all the checkpoints at the end
    cleanup_checkpoints_per_round: bool = MISSING
        Whether to clean up the checkpoints at the end of each round
    wandb: Wandb
        Wandb configuration

    """

    run_uuid: str = MISSING
    seed: int = MISSING
    pretrained_model_path: str | None = MISSING
    wte_parameters_path: str | None = MISSING
    centralized: Centralized = MISSING
    photon: Photon = MISSING
    fl: FL = MISSING
    s3_comm_config: S3CommConfig = MISSING
    use_wandb: bool = MISSING
    cleanup_checkpoints: bool = MISSING
    cleanup_checkpoints_per_round: bool = MISSING
    wandb: Wandb = MISSING

    # NOTE: MosaicML specific, do not include in the base schema
    llm_config: LLMConfig = MISSING
    dataset: Dataset = MISSING


def register_config(name: "str") -> None:
    """Register the base configuration schema."""
    cs = ConfigStore.instance()
    cs.store(name=name, node=BaseConfig)
