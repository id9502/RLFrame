import os
import random
from core.common.config import Config


__all__ = ["makedir", "assets_dir", "model_dir", "log_dir", "demo_dir"]


def makedir(path):
    if not os.path.exists(path):
        print(f"Making dir '{path}'")
        os.makedirs(path, exist_ok=True)


def assets_dir():
    assets_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets"))
    makedir(assets_path)
    return assets_path


def _name_str_from_cfg(config: Config):
    naming_param = config.require_field("naming")
    name = ""
    seed = random.randint(0, 10000)
    if "tag" in naming_param:
        name += f"{naming_param.pop('tag')}-"
    if "env name" in naming_param:
        name += f"{naming_param.pop('env name')}-"
    if "short" in naming_param:
        name += f"{naming_param.pop('short')}-"
    if "seed" in naming_param:
        seed = naming_param.pop("seed")

    for k in naming_param:
        name += f"{naming_param[k]}-"

    name += f"{seed}"
    return name


def model_dir(config: Config):
    name = _name_str_from_cfg(config)
    model_path = os.path.join(assets_dir(), name, "model")
    makedir(model_path)
    return model_path


def log_dir(config: Config):
    name = _name_str_from_cfg(config)
    log_path = os.path.join(assets_dir(), name, "log")
    makedir(log_path)
    return log_path


def demo_dir(*sub_path):
    """
    :param sub_path: sub folder under '/dataset', will be concatenated together
    :return: full path to demo folder or demo package
    """
    demo_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../dataset"))
    demo_path = os.path.join(demo_path, *sub_path)
    return demo_path
