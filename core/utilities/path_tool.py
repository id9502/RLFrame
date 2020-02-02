#!/usr/bin/env python3

import os
from glob import glob
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


if __name__ == "__main__":
    assets_path = assets_dir()
    os.system("clear")
    print("Experiment manager >>>")
    a = glob(os.path.join(assets_path, "*"), recursive=True)
    print(a)
    #os.symlink()
    #print("↑/↓ for selecting item, D for delete, c for ")

