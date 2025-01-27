import logging
import os
import typing as t
from pathlib import Path

import yaml


def load_config(config_path: str = "configs/config.yaml") -> t.Dict[str, t.Any]:
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def load_general_config(config_path: str = "configs/general.yaml") -> t.Dict[str, t.Any]:
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def load_experiment_config(experiment_name: str) -> t.Dict[str, t.Any]:
    all_configs = load_config()
    try:
        exp_config = all_configs[experiment_name]
    except KeyError:
        logging.error(f"Experiment {experiment_name} not found")
        raise KeyError()

    exp_config["dir"] = Path("experiments") / exp_config["name"]
    os.makedirs(exp_config["dir"], exist_ok=True)

    exp_config.update(load_general_config())
    exp_config["data_dir"] = Path(exp_config["data_dir"])

    return exp_config
