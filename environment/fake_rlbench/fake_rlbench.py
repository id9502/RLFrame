import numpy as np
from rlbench import tasks
from rlbench.environment import SUPPORTED_ROBOTS
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.environment import Environment as RLEnvironment
from rlbench.task_environment import _DT, Quaternion
from core.common import StepDict, SampleBatch
from core.environment.environment import Environment
from core.utilities.convenience import all_class_names, get_named_class
from .stdout_footpad import suppress_stdout
import random as rnd


class FakeRLBenchEnv(Environment):

    ROBOT_NAME = SUPPORTED_ROBOTS.keys()
    OBSERVATION_MODE = ("state", "vision", "all")
    ACTION_MODE = {"joint velocity": ArmActionMode.ABS_JOINT_VELOCITY,
                   "delta joint velocity": ArmActionMode.DELTA_JOINT_VELOCITY,
                   "joint position": ArmActionMode.ABS_JOINT_POSITION,
                   "delta joint position": ArmActionMode.DELTA_JOINT_POSITION,
                   "joint torque": ArmActionMode.ABS_JOINT_TORQUE,
                   "delta joint torque": ArmActionMode.DELTA_JOINT_TORQUE,
                   "effector velocity": ArmActionMode.ABS_EE_VELOCITY,
                   "delta effector velocity": ArmActionMode.DELTA_EE_VELOCITY,
                   "effector position": ArmActionMode.ABS_EE_POSE,
                   "delta effector position": ArmActionMode.DELTA_EE_POSE}

    def __init__(self, task_name: str, observation_mode: str = "state",
                 action_mode: str = "delta joint position", robot_name: str = "panda"):
        super(FakeRLBenchEnv, self).__init__(task_name)
        if task_name not in all_class_names(tasks):
            raise KeyError(f"Error: unknown task name {task_name}")
        if observation_mode not in FakeRLBenchEnv.OBSERVATION_MODE:
            raise KeyError(f"Error: unknown observation mode {observation_mode}, available: {FakeRLBenchEnv.OBSERVATION_MODE}")
        if action_mode not in FakeRLBenchEnv.ACTION_MODE:
            raise KeyError(f"Error: unknown action mode {action_mode}, available: {FakeRLBenchEnv.ACTION_MODE.keys()}")
        if robot_name not in FakeRLBenchEnv.ROBOT_NAME:
            raise KeyError(f"Error: unknown robot name {robot_name}, available: {FakeRLBenchEnv.ROBOT_NAME}")

        # TODO: modify the task/robot/arm/gripper to support early instantiation before v-rep launched
        self._observation_mode = observation_mode
        self._action_mode = action_mode
        self._task_name = task_name
        self._robot_name = robot_name

        self._observation_config = ObservationConfig()
        if self._observation_mode == "state":
            self._observation_config.set_all_low_dim(True)
            self._observation_config.set_all_high_dim(False)
        elif self._observation_mode == "vision":
            self._observation_config.set_all_low_dim(False)
            self._observation_config.set_all_high_dim(True)
        elif self._observation_mode == "all":
            self._observation_config.set_all(True)

        self._action_config = ActionMode(FakeRLBenchEnv.ACTION_MODE[self._action_mode])

        self.env = None
        self.task = None

        self._update_info_dict()

    def init(self, display=False):
        with suppress_stdout():
            self.env = RLEnvironment(action_mode=self._action_config, obs_config=self._observation_config,
                                     headless=not display, robot_configuration=self._robot_name)
            self.env.launch()
            self.task = self.env.get_task(get_named_class(self._task_name, tasks))

    def reset(self, random: bool = True) -> StepDict:
        self.task._static_positions = not random
        descriptions, obs = self.task.reset()
        # Returns a list of descriptions and the first observation
        next_step = {"opt": descriptions}

        if self._observation_mode == "state" or self._observation_mode == "all":
            next_step['s'] = obs.get_low_dim_data()
        if self._observation_mode == "vision" or self._observation_mode == "all":
            next_step["left shoulder rgb"] = obs.left_shoulder_rgb
            next_step["right_shoulder_rgb"] = obs.right_shoulder_rgb
            next_step["wrist_rgb"] = obs.wrist_rgb
        return next_step

    def step(self, last_step: StepDict) -> (StepDict, bool):
        assert 'a' in last_step, "Key 'a' for action not in last_step, maybe you passed a wrong dict ?"

        obs, reward, terminate = self.task.step(last_step['a'])
        last_step['r'] = reward
        last_step["info"] = {}
        next_step = {"opt": None}

        if self._observation_mode == "state" or self._observation_mode == "all":
            next_step['s'] = obs.get_low_dim_data()
        if self._observation_mode == "vision" or self._observation_mode == "all":
            next_step["left shoulder rgb"] = obs.left_shoulder_rgb
            next_step["right_shoulder_rgb"] = obs.right_shoulder_rgb
            next_step["wrist_rgb"] = obs.wrist_rgb
        return last_step, next_step, terminate

    def finalize(self) -> bool:
        with suppress_stdout():
            self.env.shutdown()
        self.task = None
        self.env = None
        return True

    def name(self) -> str:
        return self._task_name

    # ------------- private methods ------------- #

    def _update_info_dict(self):
        # update info dict
        self._info["action mode"] = self._action_mode
        self._info["observation mode"] = self._observation_mode
        # TODO: action dim should related to robot, not action mode, here we fixed it temporally
        self._info["action dim"] = (self._action_config.action_size,)
        self._info["action low"] = -np.ones(self._info["action dim"], dtype=np.float32)
        self._info["action high"] = np.ones(self._info["action dim"], dtype=np.float32)
        if self._observation_mode == "state" or self._observation_mode == "all":
            # TODO: observation should be determined without init the entire environment
            with suppress_stdout():
                env = RLEnvironment(action_mode=self._action_config, obs_config=self._observation_config,
                                    headless=True, robot_configuration=self._robot_name)
                env.launch()
                task = env.get_task(get_named_class(self._task_name, tasks))
                _, obs = task.reset()
                env.shutdown()
                del task
                del env
            self._info["time step"] = _DT
            self._info["state dim"] = tuple(obs.get_low_dim_data().shape)
            self._info["state low"] = np.ones(self._info["state dim"], dtype=np.float32) * -np.inf
            self._info["state high"] = np.ones(self._info["state dim"], dtype=np.float32) * np.inf
        if self._observation_mode == "vision" or self._observation_mode == "all":
            self._info["left shoulder rgb dim"] = tuple(self._observation_config.left_shoulder_camera.image_size) + (3,)
            self._info["left shoulder rgb low"] = np.zeros(self._info["left shoulder rgb dim"], dtype=np.float32)
            self._info["left shoulder rgb high"] = np.ones(self._info["left shoulder rgb dim"], dtype=np.float32)
            self._info["right shoulder rgb  dim"] = tuple(self._observation_config.right_shoulder_camera.image_size) + (3,)
            self._info["right shoulder rgb  low"] = np.zeros(self._info["right shoulder rgb  dim"], dtype=np.float32)
            self._info["right shoulder rgb  high"] = np.ones(self._info["right shoulder rgb  dim"], dtype=np.float32)
            self._info["wrist rgb dim"] = tuple(self._observation_config.wrist_camera.image_size) + (3,)
            self._info["wrist rgb low"] = np.zeros(self._info["wrist rgb dim"], dtype=np.float32)
            self._info["wrist rgb high"] = np.ones(self._info["wrist rgb dim"], dtype=np.float32)
        self._info["reward low"] = -np.inf
        self._info["reward high"] = np.inf

    def live_demo(self, amount: int, random: bool = True) -> SampleBatch:
        """
        :param amount: number of demonstration trajectories to be generated
        :param random: if the starting position is random
        :return: observation list : [amount x [(steps-1) x [s, a] + [s_term, None]]],
                 WARNING: that the action here is calculated from observation, when executing, they may cause some inaccuracy
        """
        seeds = [rnd.randint(0, 4096) for _ in range(amount)]
        self.task._static_positions = not random

        demo_pack = []
        for seed in seeds:
            np.random.seed(seed)
            pack = self.task.get_demos(1, True)[0]

            demo_traj = []
            np.random.seed(seed)
            desc, obs = self.task.reset()
            v_tar = 0.
            for o_tar in pack[1:]:
                action = []
                if self._action_config.arm == ArmActionMode.ABS_JOINT_VELOCITY:
                    action.extend((o_tar.joint_positions - obs.joint_positions) / _DT)
                elif self._action_config.arm == ArmActionMode.ABS_JOINT_POSITION:
                    action.extend(o_tar.joint_positions)
                elif self._action_config.arm == ArmActionMode.ABS_JOINT_TORQUE:
                    action.extend(o_tar.joint_forces)
                    raise TypeError("Warning, abs_joint_torque is not currently supported")
                elif self._action_config.arm == ArmActionMode.ABS_EE_POSE:
                    action.extend(o_tar.gripper_pose)
                elif self._action_config.arm == ArmActionMode.ABS_EE_VELOCITY:
                    # WARNING: This calculating method is not so accurate since rotation cannot be directed 'add' together
                    #          since the original RLBench decides to do so, we should follow it
                    action.extend((o_tar.gripper_pose - obs.gripper_pose) / _DT)
                elif self._action_config.arm == ArmActionMode.DELTA_JOINT_VELOCITY:
                    v_tar = (o_tar.joint_positions - obs.joint_positions) / _DT
                    action.extend(v_tar - obs.joint_velocities)
                    raise TypeError("Warning, delta_joint_velocity is not currently supported")
                elif self._action_config.arm == ArmActionMode.DELTA_JOINT_POSITION:
                    action.extend(o_tar.joint_positions - obs.joint_positions)
                elif self._action_config.arm == ArmActionMode.DELTA_JOINT_TORQUE:
                    action.extend(o_tar.joint_forces - obs.joint_forces)
                    raise TypeError("Warning, delta_joint_torque is not currently supported")
                elif self._action_config.arm == ArmActionMode.DELTA_EE_POSE:
                    action.extend(o_tar.gripper_pose[:3] - obs.gripper_pose[:3])
                    q = Quaternion(o_tar.gripper_pose[3:7]) * Quaternion(obs.gripper_pose[3:7]).conjugate
                    action.extend(list(q))
                elif self._action_config.arm == ArmActionMode.DELTA_EE_VELOCITY:
                    # WARNING: This calculating method is not so accurate since rotation cannot be directed 'add' together
                    #          since the original RLBench decides to do so, we should follow it
                    v_tar_new = (o_tar.gripper_pose - obs.gripper_pose) / _DT
                    action.extend(v_tar_new - v_tar)
                    v_tar = v_tar_new
                    raise TypeError("Warning, delta_ee_velocity is not currently supported")

                action.append(1.0 if o_tar.gripper_open > 0.9 else 0.0)
                action = np.asarray(action, dtype=np.float32)
                demo_traj.append({'observation': obs,
                                  'a': action,
                                  's': obs.get_low_dim_data()})
                obs, reward, done = self.task.step(action)
                demo_traj[-1]['r'] = reward

            demo_pack.append(demo_traj)
        return {"trajectory": demo_pack,
                "config": "default",
                "policy": "hand-coding",
                "env class": self.__class__.__name__,
                "env name": self._task_name,
                "env config": "default",
                "observation config": self._observation_mode,
                "robot config": self._robot_name,
                "action config": self._action_mode}


if __name__ == "__main__":
    import bz2
    import pickle
    from time import sleep

    # normal rl environment test
    e = FakeRLBenchEnv("CloseMicrowave")
    e.init(display=False)
    e.reset()
    for i in range(10):
        e.step({'a': np.random.randn(*e.info()["action dim"])})
        sleep(0.1)

    e.finalize()

    # generate demonstrations
    env = FakeRLBenchEnv("PutUmbrellaInUmbrellaStand")
    env.init(display=True)
    pack = env.live_demo(10, False)
    with bz2.BZ2File("demo_10x_PutUmbrellaInUmbrellaStand.pkl", "wb") as f:
        pickle.dump(pack, f)

    env.finalize()
