#!/usr/bin/env python3
import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.logger import Logger
from core.algorithm.bc import bc_step
from core.filter.zfilter import Filter
from core.model import Policy as Policy
from core.agent import Agent_sync as Agent
from core.common import ParamDict, ARGConfig, ARG
from core.utilities import running_time, model_dir, loadInitConfig
# from environment import FakeGym
from environment import FakeRLBench


default_config = ARGConfig(
    "PyTorch Behavior Cloning example",
    ARG("env name", "ReachTarget", critical=True, fields=["naming"],
        desc="name of the environment to run"),
    ARG("action mode", "delta joint position", critical=True, fields=["naming"],
        desc="name of the action mode, (default: {})"),
    ARG("tag", "default", fields=["naming"],
        desc="tag of this experiment"),
    ARG("short", "bc", critical=True, fields=["naming"],
        desc="short name of this method"),
    ARG("seed", 1, critical=True, fields=["naming"], desc="random seed (default: {})"),

    ARG("load name", "~final.pkl", desc="name of pre-trained model"),
    ARG("demo path", "RLBench/1000_djp_ReachTarget.demo.pkl", desc="demo package path"),
    # ---- model parameters ---- #
    ARG("activation", "tanh", critical=True,
        desc="activation function name('tanh', 'sigmoid', 'relu')"),
    ARG("l2 reg", 1.e-3, critical=True, desc="l2 regularization regression (default: {})"),
    ARG("lr", 1.e-3, critical=True, desc="Learning rate (default: {})"),
    ARG("lr factor", -0.e-4, critical=True, desc="Learning rate (default: {})"),
    ARG("bc method", "l2", critical=True, desc="method for determining distance (default: {})"),

    # ---- program config ---- #
    ARG("batch size", 2048, desc="batch size per update (number of s-a pairs) (default: {})"),
    ARG("max iter", 100, desc="maximal number of training iterations (default: {})"),
    ARG("eval batch size", 4, desc="batch size used for evaluations (default: {})"),
    ARG("eval interval", 1, desc="interval between evaluations (default: {})"),
    ARG("save interval", 5, desc="interval between saving (default: {}, 0, means never save)"),
    ARG("threads", 2, desc="number of threads for agent (default: {})"),
    ARG("gpu threads", 2, desc="number of threads for agent (default: {})"),
    ARG("gpu", (0, 1, 2, 3), desc="tuple of available GPUs, empty for cpu only"),
)


def train_loop(cfg, agent, logger):
    curr_iter, max_iter, eval_iter, eval_batch_sz, save_iter, demo_loader =\
        cfg.require("current training iter", "max iter", "eval interval",
                    "eval batch size", "save interval", "demo loader")

    validate_cfg = ParamDict({
        "policy state dict": None,
        "filter state dict": None,
        "trajectory max step": 64,
        "batch size": eval_batch_sz,
        "fixed environment": False,
        "fixed policy": True,
        "fixed filter": True
    })

    # we use the entire demo set without sampling
    demo_trajectory = demo_loader.generate_all()
    if demo_trajectory is None:
        raise FileNotFoundError("Demo file not exists or cannot be loaded, abort !")
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

        """sample new data batch and perform Behavior Cloning update"""
        loss = bc_step(cfg, agent.policy(), demo_trajectory)

        e_time = float(running_time(fmt=False))

        cfg["current training iter"] = i_iter + 1
        cfg["policy state dict"] = validate_cfg["policy state dict"] = agent.policy().getStateDict()
        cfg["filter state dict"] = validate_cfg["filter state dict"] = agent.filter().getStateDict()

        if i_iter % eval_iter == 0:
            batch_eval, info_eval = agent.rollout(validate_cfg)
            logger.train(False)
            info_eval["duration"] = e_time - s_time
            info_eval["epoch"] = i_iter
            info_eval["loss"] = loss
            logger(info_eval)

        if i_iter != 0 and i_iter % save_iter == 0:
            file_name = os.path.join(model_dir(cfg), f"I_{i_iter}.pkl")
            cfg.save(file_name)
            print(f"Saving current step at {file_name}")

    file_name = os.path.join(model_dir(cfg), f"final.pkl")
    cfg.save(file_name)
    print(f"Total running time: {running_time(fmt=True)}, result saved at {file_name}")


def main(cfg):
    env_name, action_mode, policy_state, filter_state =\
        cfg.require("env name", "action mode", "policy state dict", "filter state dict")

    logger = Logger()
    logger.init(cfg)

    filter_op = Filter()
    # env = FakeGym(env_name)
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
