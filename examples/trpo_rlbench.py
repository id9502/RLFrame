#!/usr/bin/env python3
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.logger import Logger
from core.filter.zfilter import ZFilter
from core.algorithm.trpo import trpo_step
from core.agent import Agent_sync as Agent
from core.model import PolicyWithValue as Policy
from core.common import ParamDict, ARGConfig, ARG
from core.utilities import running_time, model_dir, loadInitConfig
from environment import FakeRLBench


default_config = ARGConfig(
    "PyTorch TRPO example",
    ARG("env name", "ReachTarget", critical=True, fields=["naming"],
        desc="name of the environment to run"),
    ARG("action mode", "delta joint position", critical=True, fields=["naming"],
        desc="name of the action mode, (default: {})"),
    ARG("tag", "default", fields=["naming"],
        desc="tag of this experiment"),
    ARG("short", "trpo", critical=True, fields=["naming"],
        desc="short name of this method"),
    ARG("seed", 1, critical=True, fields=["naming"], desc="random seed (default: {})"),

    ARG("load name", "~final.pkl", desc="name of pre-trained model"),
    # ---- model parameters ---- #
    ARG("activation", "tanh", critical=True,
        desc="activation function name('tanh', 'sigmoid', 'relu')"),
    ARG("gamma", 0.99, critical=True, key_name="advantage gamma", fields=["policy init"],
        desc="discount factor (default: {})"),
    ARG("tau", 0.95, critical=True, key_name="advantage tau", fields=["policy init"],
        desc="gae (default: {})"),
    ARG("damping", 1.e-2, critical=True, desc="damping (default: {})"),
    ARG("l2 reg", 1.e-3, critical=True, desc="l2 regularization regression (default: {})"),
    ARG("lr", 1.e-4, critical=True, desc="Learning rate (default: {})"),
    ARG("max kl", 1.e-2, critical=True, desc="max kl value (default: {})"),
    ARG("use zfilter", True, critical=True, desc="filter the state when running (default {})"),

    # ---- program config ---- #
    ARG("batch size", 32, desc="batch size per update (default: {})"),
    ARG("max iter", 5000, desc="maximal number of training iterations (default: {})"),
    ARG("eval batch size", 4, desc="batch size used for evaluations (default: {})"),
    ARG("eval interval", 1, desc="interval between evaluations (default: {})"),
    ARG("save interval", 100, desc="interval between saving (default: {}, 0, means never save)"),
    ARG("threads", 8, desc="number of threads for agent (default: {})"),
    ARG("gpu threads", 2, desc="number of threads for agent (default: {})"),
    ARG("gpu", (0, 1, 2, 3), desc="tuple of available GPUs, empty for cpu only"),
)


def train_loop(cfg, agent, logger):
    curr_iter, max_iter, eval_iter, eval_batch_sz, batch_sz, save_iter =\
        cfg.require("current training iter", "max iter", "eval interval",
                    "eval batch size", "batch size", "save interval")

    training_cfg = ParamDict({
        "policy state dict": agent.policy().getStateDict(),
        "filter state dict": agent.filter().getStateDict(),
        "trajectory max step": 64,
        "batch size": batch_sz,
        "fixed environment": False,
        "fixed policy": False,
        "fixed filter": False
    })
    validate_cfg = ParamDict({
        "policy state dict": None,
        "filter state dict": None,
        "trajectory max step": 64,
        "batch size": eval_batch_sz,
        "fixed environment": False,
        "fixed policy": True,
        "fixed filter": True
    })

    for i_iter in range(curr_iter, max_iter):

        s_time = float(running_time(fmt=False))

        """sample new batch and perform TRPO update"""
        batch_train, info_train = agent.rollout(training_cfg)
        trpo_step(cfg, batch_train, agent.policy())

        e_time = float(running_time(fmt=False))

        logger.train()
        info_train["duration"] = e_time - s_time
        info_train["epoch"] = i_iter
        logger(info_train)

        cfg["current training iter"] = i_iter + 1
        cfg["policy state dict"] = training_cfg["policy state dict"] = validate_cfg["policy state dict"] = agent.policy().getStateDict()
        cfg["filter state dict"] = training_cfg["filter state dict"] = validate_cfg["filter state dict"] = agent.filter().getStateDict()

        if i_iter % eval_iter == 0:
            batch_eval, info_eval = agent.rollout(validate_cfg)

            logger.train(False)
            info_eval["duration"] = e_time - s_time
            info_eval["epoch"] = i_iter
            logger(info_eval)

        if i_iter != 0 and i_iter % save_iter == 0:
            file_name = os.path.join(model_dir(cfg), f"I_{i_iter}.pkl")
            cfg.save(file_name)
            print(f"Saving current step at {file_name}")

    file_name = os.path.join(model_dir(cfg), f"final.pkl")
    cfg.save(file_name)
    print(f"Total running time: {running_time(fmt=True)}, result saved at {file_name}")


def main(cfg):
    env_name, action_mode, use_zf, gamma, tau, policy_state, filter_state =\
        cfg.require("env name", "action mode", "use zfilter", "advantage gamma", "advantage tau", "policy state dict", "filter state dict")

    logger = Logger()
    logger.init(cfg)

    filter_op = ZFilter(gamma, tau, enable=use_zf)
    env = FakeRLBench(env_name, action_mode=action_mode)
    policy = Policy(cfg, env.info())
    agent = Agent(cfg, env, policy, filter_op)

    # ---- start training ---- #
    if policy_state is not None:
        agent.policy().reset(policy_state)
    if filter_state is not None:
        agent.filter().reset(filter_state)

    train_loop(cfg, agent, logger)

    print("Done")


if __name__ == '__main__':
    import torch.multiprocessing as multiprocessing
    multiprocessing.set_start_method('spawn')

    default_config.parser()

    train_cfg = loadInitConfig(default_config)

    main(train_cfg)

    exit(0)
