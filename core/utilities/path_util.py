import os
from core.common.config import ParamDict


def _makedir(path):
    if not os.path.exists(path):
        print(f"Making dir '{path}'")
        os.makedirs(path, exist_ok=True)


def assets_dir():
    assets_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets"))
    _makedir(assets_path)
    return assets_path


def model_dir(config: ParamDict):
    env_name, tag, short, seed = config.require("env name", "tag", "short", "seed")
    model_path = os.path.join(assets_dir(), f"{tag}-{env_name}-{short}-{seed}", "model")
    _makedir(model_path)
    return model_path


def log_dir(config: ParamDict):
    env_name, tag, short, seed = config.require("env name", "tag", "short", "seed")
    log_path = os.path.join(assets_dir(), f"{tag}-{env_name}-{short}-{seed}", "log")
    _makedir(log_path)
    return log_path


def demo_dir(*sub_path):
    """
    :param sub_path: sub folder under '/dataset', will be concatenated together
    :return: full path to demo folder or demo package
    """
    demo_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../dataset"))
    demo_path = os.path.join(demo_path, *sub_path)
    return demo_path
