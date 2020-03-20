import os
import bz2
import random
import pickle
import numpy as np
from environment import FakeRLBench
from core.common import ARGConfig, ARG
from core.utilities import assets_dir, makedir


default_config = ARGConfig(
    "RLBench demo generator example",
    ARG("env name", "ReachTarget", critical=True, fields=["naming"],
        desc="name of the environment to run"),
    ARG("action mode", "delta joint position", critical=True, fields=["naming"],
        desc="name of the action mode, (default: {})"),
    ARG("seed", 1, critical=True, fields=["naming"], desc="random seed (default: {})"),

    ARG("save name", os.path.join("RLBench", "50_ReachTarget.demo.pkl"), desc="path for saving generated demos"),

    # ---- simulation config
    ARG("observation mode", "state", desc="observation mode, could be ['state', 'vision', 'all'] (default: {})"),
    ARG("robot name", "panda", desc="robot to be used (default: {})"),

    # ---- program config ---- #
    ARG("demo size", 50, desc="number of demo trajectories to be generated (default: {})"),
    ARG("demo random", True, desc="whether the generated demos are random inited (default: {})"),
    ARG("display", False, desc="whether use GUI form or not (default: {}"),
)


def main(cfg):
    env_name, save_name, seed, display, demo_size, demo_random, obs_mode, act_mode, robot = \
        cfg.require("env name", "save name", "seed", "display", "demo size",
                    "demo random", "observation mode", "action mode", "robot name")

    save_path = os.path.join(assets_dir(), save_name)
    makedir(os.path.dirname(save_path))

    random.seed(seed)

    env = FakeRLBench(env_name, observation_mode=obs_mode, action_mode=act_mode, robot_name=robot)
    env.init(display=display)
    pack = env.live_demo(demo_size, random=demo_random)
    with bz2.BZ2File(save_path, "wb") as f:
        pickle.dump(pack, f)
    env.finalize()

    traj_states_num = np.asarray([len(traj) for traj in pack["trajectory"]], dtype=np.float32)
    print(f"---------- Generating Done --------------")
    print(f"num trajectories: {len(traj_states_num)}")
    print(f"max traj length: {traj_states_num.max()}")
    print(f"min traj length: {traj_states_num.min()}")
    print(f"avg traj length: {traj_states_num.mean()}")
    print(f"saved at '{save_path}'")
    print(f"-----------------------------------------")


if __name__ == "__main__":
    default_config.parser()
    main(default_config)
