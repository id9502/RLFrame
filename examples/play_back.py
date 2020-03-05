#!/usr/bin/env python3
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.model import Policy
from core.agent import Agent as Agent
from core.filter.zfilter import ZFilter
from core.utilities import loadInitConfig
from core.common import ParamDict, ARGConfig, ARG
from environment import FakeGym
from environment import FakeRLBench


default_config = ARGConfig(
    "PyTorch Demo Replay example",

    ARG("load name", "dataset/learned/final.pkl", desc="name of pre-trained model"),

    # ---- program config ---- #
    ARG("verify iter", 5, desc="maximal number of training iterations (default: {})"),
    ARG("verify display", True, desc="whither display the GUI form or not (default: {})"),
    ARG("gpu", (0, 1, 2, 3), save=False, desc="tuple of available GPUs, empty for cpu only"),
)


def replay_loop(cfg, agent):
    max_iter, display = cfg.require("verify iter", "verify display")

    validate_cfg = ParamDict({
        "policy state dict": cfg["policy state dict"],
        "filter state dict": cfg["filter state dict"],
        "trajectory max step": 128,
        "max iter": max_iter,
        "display": display,
        "fixed environment": False,
        "fixed policy": True,
        "fixed filter": True
    })

    agent.verify(validate_cfg)


def main(cfg):
    env_name, gamma, tau, policy_state, filter_state = \
        cfg.require("env name", "advantage gamma", "advantage tau", "policy state dict", "filter state dict")

    filter_op = ZFilter(gamma, tau)
    # env = FakeGym(env_name)
    env = FakeRLBench(env_name)
    policy = Policy(cfg, env.info())
    agent = Agent(cfg, env, policy, filter_op)

    # ---- start training ---- #
    if policy_state is not None:
        agent.policy().reset(policy_state)
    if filter_state is not None:
        agent.filter().reset(filter_state)

    print("Info: Start replaying saved model")
    replay_loop(cfg, agent)

    print("Done")


if __name__ == "__main__":
    default_config.parser()

    saved_config = loadInitConfig(default_config)

    main(saved_config)

    exit(0)
