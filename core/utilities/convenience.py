import os
from copy import deepcopy
from core.utilities import model_dir
from inspect import getmembers, isclass
from core.common import Config, Type, ModuleType, Tuple


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


def cpu_state_dict(state_dict):
    """
    Warning! the state_dict will be modified inplace
    :param state_dict: state dict from any device
    :return: state_dict on cpu
    """
    for k, v in state_dict.items():
        state_dict[k] = v.cpu()
    return state_dict


def get_named_class(class_name: str, model: ModuleType) -> Type:
    all_class_dict = {}
    for o in getmembers(model):
        if isclass(o[1]):
            all_class_dict[o[0]] = o[1]

    if class_name not in all_class_dict:
        raise NotImplementedError(f"No class {class_name} found in {model.__name__} !")
    return all_class_dict[class_name]


def all_class_names(model: ModuleType) -> Tuple[str]:
    return tuple(o[0] for o in getmembers(model) if isclass(o[1]))
