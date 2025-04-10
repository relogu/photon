"""Pydantic models for the configs used by the clients."""

import ast
from collections.abc import Callable
from typing import Any, TypeVar, cast

from flwr.common.typing import ConfigsRecordValues
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, field_validator
from pydantic.functional_validators import FieldValidatorModes

from photon.conf.base_schema import BaseConfig, DictConfig
from photon.utils import ClientState

F = TypeVar("F", bound=Callable[..., Any])


def typed_field_validator(
    field: str,
    /,
    *field_names: str,
    mode: FieldValidatorModes = "after",
) -> Callable[[F], F]:
    """Wrap Pydantic's validator.

    Used to avoid the "Untyped function decorator obscures type of function" warning.

    Parameters
    ----------
    field : str
        The field to validate.
    field_names : str
        The field names to validate.
    mode : FieldValidatorModes
        The mode to use for the validator.

    Returns
    -------
    Callable[[F], F]
        The decorator.

    """

    def decorator(func: F) -> F:
        # Apply the Pydantic validator decorator
        return field_validator(
            field,
            *field_names,
            mode=mode,
        )(func)

    return decorator


class FitConfig(BaseModel):
    """Pydantic model for the fit configuration.

    Attributes
    ----------
    cid : int | str
        The client ID.
    server_round : int
        The server round.
    batch_size : int
        The batch size.
    n_local_steps : int
        The number of local steps.
    n_local_epochs : int
        The number of local epochs.
    reset_checkpoint : bool
        Whether to reset the checkpoint.
    reset_optimizer : bool
        Whether to reset the optimizer.
    reset_dataset_state : bool
        Whether to reset the dataset state.
    reset_timestamp : bool
        Whether to reset the timestamp.
    use_unigram_metrics : bool
        Whether to use unigram metrics.
    allow_unigram_metrics_failures : bool
        Whether to allow freq dict failures.
    aggregate_momenta: bool
        Whether to aggregate momenta.
    resize_vocab : int | None
        The vocabulary size.
    s3_comm_config : DictConfig
        The S3 communication configuration.
    random_layers : list[str] | None
        The random layers.
    random_init_freq : int
        The random init frequency.
    personalized_layers : list[str] | None
        The personalized layers.
    truly_random_init : bool
        Whether to use a different seed for random layer init.
    frozen_layers : list[str] | None
        Specify layers to freeze.
    unfrozen_layers : list[str] | None
        Specify layers to keep unfrozen and freeze everything else.
    split_eval : bool
        Whether to split the evaluation across streams.
    set_trainer_params_filter_keys : bool
        Weather to filter the keys in the model dictionary.
    set_trainer_key_to_filter : str
        The key to filter in the model dictionary.
    client_state: ClientState | None = None
        The client state per round, added by server
    server_steps_cumulative: int | None = None
        The cumulative server steps, added by server

    """

    cid: int | str
    server_round: int
    batch_size: int
    n_local_steps: int
    n_local_epochs: int
    reset_checkpoint: bool
    reset_optimizer: bool
    reset_dataset_state: bool
    reset_timestamp: bool
    use_unigram_metrics: bool
    allow_unigram_metrics_failures: bool
    aggregate_momenta: bool
    resize_vocab: int | None = None
    s3_comm_config: DictConfig
    random_layers: list[str] | None = None
    random_init_freq: int
    personalized_layers: list[str] | None = None
    truly_random_init: bool
    frozen_layers: list[str] | None = None
    unfrozen_layers: list[str] | None = None
    split_eval: bool
    set_trainer_params_filter_keys: bool
    set_trainer_key_to_filter: str
    client_state: dict[str, ClientState] | None = None
    server_steps_cumulative: int | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @typed_field_validator("s3_comm_config", mode="before")
    @classmethod
    def validates3_comm_config(
        cls: Any,
        v: Any,  # noqa: ANN401
    ) -> DictConfig | Any:  # noqa: ANN401
        """Convert s3 comm config to OmegaConf object.

        Parameters
        ----------
        v : Any
            The value to convert.

        Returns
        -------
        Any
            The converted value.

        """
        if isinstance(v, str):
            return cast("DictConfig", OmegaConf.create(ast.literal_eval(v)))
        return v

    @typed_field_validator("client_state", mode="before")
    @classmethod
    def validate_client_state(
        cls: Any,
        v: Any,  # noqa: ANN401
    ) -> dict[str | int, ClientState] | Any:  # noqa: ANN401
        """Convert s3 comm config to OmegaConf object.

        Parameters
        ----------
        v : Any
            The value to convert.

        Returns
        -------
        Any
            The converted value.

        """
        if isinstance(v, str):
            client_states_dict: dict[int | str, dict[str, Any]] = ast.literal_eval(v)
            return {str(k): ClientState(**v) for k, v in client_states_dict.items()}
        if isinstance(v, dict):
            return {str(k): ClientState(**v) for k, v in v.items()}
        return v

    @typed_field_validator(
        "random_layers",
        "personalized_layers",
        "frozen_layers",
        "unfrozen_layers",
        "resize_vocab",
        mode="before",
    )
    @classmethod
    def validate_ast(cls: Any, v: Any) -> Any:  # noqa: ANN401
        """Convert strings to python objects.

        Parameters
        ----------
        v : Any
            The value to convert.

        Returns
        -------
        Any
            The converted value.

        """
        if isinstance(v, str):
            return ast.literal_eval(v)
        return v


def get_photon_fit_config_fn(
    cfg: BaseConfig,
) -> Callable[[int, str | int], dict[str, ConfigsRecordValues]]:
    """Get a fit config fn for the client.

    Parameters
    ----------
    cfg : BaseConfig
        The configuration object.

    Returns
    -------
    Callable[[int, str | int], dict[str, ConfigsRecordValues]]
        The fit configuration fn.

    """

    def photon_fit_config(
        server_round: int,
        client_id: str | int,
    ) -> dict[str, ConfigsRecordValues]:
        """Create a fit config for the client.

        Parameters
        ----------
        server_round : int
                The server round.
        client_id : str | int
            The client ID.
        cfg : BaseConfig
            The configuration object.

        Returns
        -------
        dict[str, ConfigsRecordValues]
            The fit configuration.

        """
        config = {
            "cid": client_id,
            "server_round": server_round,
            "batch_size": cfg.llm_config.global_train_batch_size,
            "n_local_steps": cfg.fl.n_local_steps,
            "n_local_epochs": cfg.fl.n_local_epochs,
            "reset_checkpoint": cfg.fl.reset_checkpoint,
            "reset_optimizer": cfg.fl.reset_optimizer,
            "reset_dataset_state": cfg.fl.reset_dataset_state,
            "reset_timestamp": cfg.fl.reset_timestamp,
            "use_unigram_metrics": cfg.fl.use_unigram_metrics,
            "allow_unigram_metrics_failures": cfg.fl.allow_unigram_metrics_failures,
            "resize_vocab": str(cfg.fl.resize_vocab),
            "s3_comm_config": str(
                OmegaConf.to_container(cfg.s3_comm_config, resolve=True),
            ),
            "random_layers": str(cfg.fl.random_layers),
            "random_init_freq": cfg.fl.random_init_freq,
            "personalized_layers": str(cfg.fl.personalized_layers),
            "truly_random_init": cfg.fl.truly_random_init,
            "frozen_layers": str(cfg.fl.frozen_layers),
            "unfrozen_layers": str(cfg.fl.unfrozen_layers),
            "split_eval": cfg.fl.split_eval,
            "set_trainer_params_filter_keys": cfg.fl.set_trainer_params_filter_keys,
            "set_trainer_key_to_filter": cfg.fl.set_trainer_key_to_filter,
            "aggregate_momenta": cfg.fl.aggregate_momenta,
        }
        # Validate
        FitConfig(**config)
        return config

    return photon_fit_config


class EvaluateConfig(BaseModel):
    """Pydantic model for the fit configuration.

    Attributes
    ----------
    cid : int | str
        The client ID.
    server_round : int
        The server round.
    batch_size : int
        The batch size.
    n_local_steps : int
        The number of local steps.
    n_local_epochs : int
        The number of local epochs.
    reset_checkpoint : bool
        Whether to reset the checkpoint.
    reset_optimizer : bool
        Whether to reset the optimizer.
    reset_dataset_state : bool
        Whether to reset the dataset state.
    reset_timestamp : bool
        Whether to reset the timestamp.
    use_unigram_metrics : bool
        Whether to use unigram metrics.
    allow_unigram_metrics_failures : bool
        Whether to allow freq dict failures.
    resize_vocab : int | None
        The vocabulary size.
    s3_comm_config : DictConfig
        The S3 communication configuration.
    random_layers : list[str] | None
        The random layers.
    random_init_freq : int
        The random init frequency, 0 means never.
    personalized_layers : list[str] | None
        The personalized layers.
    truly_random_init : bool
        Weather to use a different seed for random layer init.
    split_eval : bool
        Whether to split the evaluation across streams.
    set_trainer_params_filter_keys : bool
        Weather to filter the keys in the model dictionary.
    set_trainer_key_to_filter : str
        The key to filter in the model dictionary.
    aggregate_momenta: bool
        Whether to aggregate momenta.
    client_state: ClientState | None = None
        The client state per round, added by server
    server_steps_cumulative: int | None = None
        The cumulative server steps, added by server

    """

    cid: int | str
    server_round: int
    batch_size: int
    use_unigram_metrics: bool
    allow_unigram_metrics_failures: bool
    resize_vocab: int | None = None
    s3_comm_config: DictConfig
    split_eval: bool
    set_trainer_params_filter_keys: bool
    set_trainer_key_to_filter: str
    aggregate_momenta: bool
    client_state: dict[str | int, ClientState] | None = None
    server_steps_cumulative: int | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @typed_field_validator("s3_comm_config", mode="before")
    @classmethod
    def validates3_comm_config(
        cls: Any,
        v: Any,  # noqa: ANN401
    ) -> DictConfig | Any:  # noqa: ANN401
        """Convert s3 comm config to OmegaConf object.

        Parameters
        ----------
        v : Any
            The value to convert.

        Returns
        -------
        Any
            The converted value.

        """
        if isinstance(v, str):
            return cast("DictConfig", OmegaConf.create(ast.literal_eval(v)))
        return v

    @typed_field_validator("client_state", mode="before")
    @classmethod
    def validate_client_state(
        cls: Any,
        v: Any,  # noqa: ANN401
    ) -> dict[str | int, ClientState] | Any:  # noqa: ANN401
        """Convert s3 comm config to OmegaConf object.

        Parameters
        ----------
        v : Any
            The value to convert.

        Returns
        -------
        Any
            The converted value.

        """
        if isinstance(v, str):
            client_states_dict: dict[int | str, dict[str, Any]] = ast.literal_eval(v)
            return {int(k): ClientState(**v) for k, v in client_states_dict.items()}
        if isinstance(v, dict):
            return {int(k): ClientState(**v) for k, v in v.items()}
        return v

    @typed_field_validator("resize_vocab", mode="before")
    @classmethod
    def validate_ast(cls: Any, v: Any) -> Any:  # noqa: ANN401
        """Convert strings to python objects.

        Parameters
        ----------
        v : Any
            The value to convert.

        Returns
        -------
        Any
            The converted value.

        """
        if isinstance(v, str):
            return ast.literal_eval(v)
        return v


def get_photon_evaluate_config_fn(
    cfg: BaseConfig,
) -> Callable[[int, str | int], dict[str, ConfigsRecordValues]]:
    """Get an evaluate config fn for the client.

    Parameters
    ----------
    cfg : BaseConfig
        The configuration object.

    Returns
    -------
    Callable[[int, str | int], dict[str, ConfigsRecordValues]]
        The evaluate configuration fn.

    """

    def photon_evaluate_config(
        server_round: int,
        client_id: str | int,
    ) -> dict[str, ConfigsRecordValues]:
        """Create an evaluate config for the client.

        Parameters
        ----------
        server_round : int
                The server round.
        client_id : str | int
            The client ID.
        cfg : BaseConfig
            The configuration object.

        Returns
        -------
        dict[str, ConfigsRecordValues]
            The evaluate configuration

        """
        config = {
            "cid": client_id,
            "server_round": server_round,
            "batch_size": cfg.llm_config.device_eval_batch_size,
            "use_unigram_metrics": cfg.fl.use_unigram_metrics,
            "allow_unigram_metrics_failures": cfg.fl.allow_unigram_metrics_failures,
            "resize_vocab": str(cfg.fl.resize_vocab),
            "s3_comm_config": str(
                OmegaConf.to_container(cfg.s3_comm_config, resolve=True),
            ),
            "split_eval": cfg.fl.split_eval,
            "set_trainer_params_filter_keys": cfg.fl.set_trainer_params_filter_keys,
            "set_trainer_key_to_filter": cfg.fl.set_trainer_key_to_filter,
            "aggregate_momenta": cfg.fl.aggregate_momenta,
        }
        # Validate
        EvaluateConfig(**config)
        return config

    return photon_evaluate_config


class CentralizedConfig(BaseModel):
    """Configuration model for centralized training.

    This class defines the configuration parameters for centralized training, including
    settings for unigram metrics, vocabulary resizing, S3 communication, evaluation
    splitting, and trainer parameters.

    Attributes
    ----------
    use_unigram_metrics : bool
        Whether to use unigram metrics.
    allow_unigram_metrics_failures : bool
        Whether to allow failures in unigram metrics.
    resize_vocab : int | None, optional
        The size to resize the vocabulary to. Default is None.
    s3_comm_config : DictConfig
        The configuration for S3 communication.
    split_eval : bool
        Whether to split evaluation data.
    set_trainer_params_filter_keys : bool
        Whether to set trainer parameters filter keys.
    set_trainer_key_to_filter : str
        The key to filter trainer parameters.
    model_config : ConfigDict
        The model configuration, allowing arbitrary types.

    Methods
    -------
        validates3_comm_config(cls, v)
            Convert S3 communication config to OmegaConf object.
        validate_ast(cls, v)
            Convert strings to Python objects.

    """

    use_unigram_metrics: bool
    allow_unigram_metrics_failures: bool
    resize_vocab: int | None = None
    s3_comm_config: DictConfig
    split_eval: bool
    set_trainer_params_filter_keys: bool
    set_trainer_key_to_filter: str
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @typed_field_validator("s3_comm_config", mode="before")
    @classmethod
    def validates3_comm_config(
        cls: Any,
        v: Any,  # noqa: ANN401
    ) -> DictConfig | Any:  # noqa: ANN401
        """Convert s3 comm config to OmegaConf object.

        Parameters
        ----------
        v : Any
            The value to convert.

        Returns
        -------
        Any
            The converted value.

        """
        if isinstance(v, str):
            return cast("DictConfig", OmegaConf.create(ast.literal_eval(v)))
        return v

    @typed_field_validator("resize_vocab", mode="before")
    @classmethod
    def validate_ast(cls: Any, v: Any) -> Any:  # noqa: ANN401
        """Convert strings to python objects.

        Parameters
        ----------
        v : Any
            The value to convert.

        Returns
        -------
        Any
            The converted value.

        """
        if isinstance(v, str):
            return ast.literal_eval(v)
        return v
