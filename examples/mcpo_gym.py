#!/usr/bin/env python3
import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.model import Policy
from core.logger import Logger
from core.filter.zfilter import ZFilter
from core.algorithm.mcpo import mcpo_step
from core.agent import Agent_sync as Agent
from core.common import ParamDict, ARGConfig, ARG
from core.utilities import running_time, model_dir, loadInitConfig
from environment import FakeGym
# from environment import FakeRLBench


default_config = ARGConfig(
    "PyTorch MC-PO example",
    ARG("env name", "SparseMountainCarContinuous-v0", critical=True, desc="name of the environment to run"),
    ARG("tag", "default", desc="tag of this experiment"),
    ARG("short", "mcpo", critical=True, desc="short name of this method"),

    ARG("load name", "~final.pkl", desc="name of pre-trained model"),
    ARG("demo path", "imitation/imperfect_MountainCarContinuous-v0.demo.pkl", desc="demo package path"),
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
    ARG("bc method", "l2", critical=True, desc="method for determining distance (default: {})"),
    ARG("constraint", 0., critical=True, desc="constraint limit of behavior discrepancy (default: {})"),
    ARG("constraint factor", 1.e-4, critical=True, desc="constraint limit growth along iter (default: {})"),
    ARG("constraint max", 10., critical=True, desc="constraint max growth along iter (default: {})"),
    ARG("seed", 1, critical=True, desc="random seed (default: {})"),
    ARG("use zfilter", True, critical=True, desc="filter the state when running (default {})"),

    # ---- program config ---- #
    ARG("batch size", 32, desc="batch size per update (default: {})"),
    ARG("max iter", 5000, desc="maximal number of training iterations (default: {})"),
    ARG("eval batch size", 4, desc="batch size used for evaluations (default: {})"),
    ARG("eval interval", 1, desc="interval between evaluations (default: {})"),
    ARG("save interval", 500, desc="interval between saving (default: {}, 0, means never save)"),
    ARG("threads", 4, desc="number of threads for agent (default: {})"),
    ARG("gpu threads", 2, desc="number of threads for agent (default: {})"),
    ARG("gpu", (0, 1, 2, 3), desc="tuple of available GPUs, empty for cpu only"),
)


def train_loop(cfg, agent, logger):
    curr_iter, max_iter, eval_iter, eval_batch_sz, batch_sz, save_iter, demo_loader =\
        cfg.require("current training iter", "max iter", "eval interval",
                    "eval batch size", "batch size", "save interval", "demo loader")

    training_cfg = ParamDict({
        "policy state dict": agent.policy().getStateDict(),
        "filter state dict": agent.filter().getStateDict(),
        "trajectory max step": 1024,
        "batch size": batch_sz,
        "fixed environment": False,
        "fixed policy": False,
        "fixed filter": False
    })
    validate_cfg = ParamDict({
        "policy state dict": None,
        "filter state dict": None,
        "trajectory max step": 1024,
        "batch size": eval_batch_sz,
        "fixed environment": False,
        "fixed policy": True,
        "fixed filter": True
    })

    # we use the entire demo set without sampling
    demo_trajectory = demo_loader.generate_all()
    if demo_trajectory is None:
        print("Warning: No demo loaded, fall back compatible with TRPO method")
    else:
        print("Info: Demo loaded successfully")
        demo_actions = []
        demo_states = []
        for p in demo_trajectory:
            demo_actions.append(torch.as_tensor([t['a'] for t in p], dtype=torch.float32, device=agent.policy().device))
            demo_states.append(torch.as_tensor([t['s'] for t in p], dtype=torch.float32, device=agent.policy().device))
        demo_states = torch.cat(demo_states, dim=0)
        demo_actions = torch.cat(demo_actions, dim=0)
        demo_trajectory = (demo_states, demo_actions)

    for i_iter in range(curr_iter, max_iter):

        s_time = float(running_time(fmt=False))

        """sample new batch and perform TRPO update"""
        batch_train, info_train = agent.rollout(training_cfg)
        mcpo_step(cfg, batch_train, agent.policy(), demo_trajectory)

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
    env_name, gamma, tau, policy_state, filter_state =\
        cfg.require("env name", "advantage gamma", "advantage tau", "policy state dict", "filter state dict")

    logger = Logger()
    logger.init(cfg)

    filter_op = ZFilter(gamma, tau)
    env = FakeGym(env_name)
    # env = FakeRLBench("ReachTarget")
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
