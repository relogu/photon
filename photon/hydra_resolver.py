"""Script for resolving the hydra configuration."""

import os

import hydra
from omegaconf import OmegaConf

from photon.conf import base_schema
from photon.conf.base_schema import BaseConfig

base_schema.register_config(name="base_schema")


# Define strategy
@hydra.main(config_path="conf/", config_name="base", version_base=None)
def main(cfg: BaseConfig) -> None:
    """Resolve the configuration and dump it to a YAML file.

    Parameters
    ----------
    cfg : BaseConfig
        The configuration object.

    Raises
    ------
    ValueError
        If the environmental variable PHOTON_SAVE_PATH is not set.

    """
    # Resolve the configuration
    OmegaConf.resolve(cfg)
    # Get the environmental variable for the dump folder
    save_path = os.environ.get("PHOTON_SAVE_PATH", "")
    # Raise an error if the environmental variable is not set
    if not save_path:
        msg = "The environmental variable PHOTON_SAVE_PATH is not set."
        raise ValueError(msg)
    # Dump the configuration to a YAML file under the dump folder
    OmegaConf.save(cfg, save_path + "/config.yaml")


if __name__ == "__main__":
    main()
