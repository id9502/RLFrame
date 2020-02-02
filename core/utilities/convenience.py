import os
from copy import deepcopy
from core.common import Config
from core.utilities.path_tool import model_dir


def loadInitConfig(default_cfg: Config):
    """
    This function will deepcopy the cfg and add some fields used for (continue) training
    """
    cfg = deepcopy(default_cfg)
    saved_name = cfg.require("load name")

    cfg.register_item("current training iter", 0, fields=["save"])
    cfg.register_item("policy state dict", None, fields=["save"])
    cfg.register_item("filter state dict", None, fields=["save"])

    if os.path.isfile(saved_name):
        cfg.load(saved_name, "this")
        print(f"Find saved model at {saved_name}, try loading from it")
    else:
        saved_name = os.path.join(model_dir(cfg), saved_name)
        if os.path.isfile(saved_name):
            cfg.load(saved_name, "this")
            print(f"Find saved model at {saved_name}, try loading from it")

    return cfg
