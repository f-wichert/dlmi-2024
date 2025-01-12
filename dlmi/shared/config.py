import typing as t

import yaml


def load_config(config_path: str = "configs/config.yaml") -> t.Dict[str, t.Any]:
    """Loads a yaml config and returns it as a dict

    Args:
        config_path (str): absolute or relative path to config file

    Returns:
        t.Dict[str, t.Any]: nested dictionary of configurations
    """
    with open(config_path, "r") as f:
        config: t.Dict[str, t.Any] = yaml.load(f, Loader=yaml.FullLoader)
    return config
