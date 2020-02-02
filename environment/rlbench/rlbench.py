import logging
import numpy as np
from os.path import join
from types import ModuleType
from inspect import getmembers, isclass
from pyquaternion import Quaternion
from pyrep import PyRep
from pyrep.backend.utils import suppress_std_out_and_err
from pyrep.errors import IKError
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from rlbench import tasks
from rlbench.backend.const import *
from rlbench.backend.robot import Robot
from rlbench.backend.scene import Scene
from rlbench.environment import DIR_PATH
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes import ActionMode, ArmActionMode
from rlbench.backend.exceptions import BoundaryError, WaypointError
from rlbench.task_environment import InvalidActionError, TaskEnvironmentError,\
                                     TORQUE_MAX_VEL, DT, MAX_RESET_ATTEMPTS
from rlbench.backend.task import Task
from core.common import StepDict, Type
from core.environment.environment import Environment

# TaskClass: child class inherited from Task
TaskClass = Type[Task]

__all__ = ["EnvironmentImpl"]


class EnvironmentImpl(Environment):
    """Each environment has a scene."""

    @staticmethod
    def all_task_names():
        return tuple(o[0] for o in getmembers(tasks) if isclass(o[1]))

    def __init__(self, task_name: str, obs_config: ObservationConfig = ObservationConfig(task_low_dim_state=True),
                 action_mode: ActionMode = ActionMode(),
                 arm_name: str = "Panda", gripper_name: str = "Panda_gripper"):
        super(EnvironmentImpl, self).__init__(task_name)

        self._arm_name = arm_name
        self._gripper_name = gripper_name
        self._action_mode = action_mode
        self._obs_config = obs_config
        # TODO: modify the task/robot/arm/gripper to support early instantiation before v-rep launched
        self._task = None
        self._pyrep = None
        self._robot = None
        self._scene = None

        self._variation_number = 0
        self._reset_called = False
        self._prev_ee_velocity = None
        self._update_info_dict()

    def init(self, display=False):
        if self._pyrep is not None:
            self.finalize()

        with suppress_std_out_and_err():
            self._pyrep = PyRep()
            # TODO: TTT_FILE should be defined by robot, but now robot depends on launched pyrep
            self._pyrep.launch(join(DIR_PATH, TTT_FILE), headless=not display)
            self._pyrep.set_simulation_timestep(0.005)

            # TODO: Load arm and gripper from name
            self._robot = Robot(Panda(), PandaGripper())
            self._scene = Scene(self._pyrep, self._robot, self._obs_config)
            self._set_arm_control_action()

            # Str comparison because sometimes class comparison doesn't work.
            if self._task is not None:
                self._task.unload()
            self._task = self._get_class_by_name(self._task_name, tasks)(self._pyrep, self._robot)
            self._scene.load(self._task)
            self._pyrep.start()

    def finalize(self):
        with suppress_std_out_and_err():
            self._pyrep.shutdown()
            self._pyrep = None

    def reset(self, random: bool = True) -> StepDict:
        logging.info('Resetting task: %s' % self._task.get_name())

        self._scene.reset()
        try:
            # TODO: let desc be constant
            desc = self._scene.init_episode(self._variation_number, max_attempts=MAX_RESET_ATTEMPTS, randomly_place=random)
        except (BoundaryError, WaypointError) as e:
            raise TaskEnvironmentError(
                'Could not place the task %s in the scene. This should not '
                'happen, please raise an issues on this task.'
                % self._task.get_name()) from e

        ctr_loop = self._robot.arm.joints[0].is_control_loop_enabled()
        locked = self._robot.arm.joints[0].is_motor_locked_at_zero_velocity()
        self._robot.arm.set_control_loop_enabled(False)
        self._robot.arm.set_motor_locked_at_zero_velocity(True)

        self._reset_called = True

        self._robot.arm.set_control_loop_enabled(ctr_loop)
        self._robot.arm.set_motor_locked_at_zero_velocity(locked)

        # Returns a list o f descriptions and the first observation
        return {'s': self._scene.get_observation().get_low_dim_data(), "opt": desc}

    def step(self, last_step: StepDict) -> (StepDict, bool):
        # returns observation, reward, done, info
        if not self._reset_called:
            raise RuntimeError("Call 'reset' before calling 'step' on a task.")
        assert 'a' in last_step, "Key 'a' for action not in last_step, maybe you passed a wrong dict ?"

        # action should contain 1 extra value for gripper open close state
        arm_action = np.array(last_step['a'][:-1])

        ee_action = last_step['a'][-1]
        current_ee = (1.0 if self._robot.gripper.get_open_amount()[0] > 0.9 else 0.0)

        if ee_action > 0.0:
            ee_action = 1.0
        elif ee_action < -0.0:
            ee_action = 0.0

        if self._action_mode.arm == ArmActionMode.ABS_JOINT_VELOCITY:
            self._assert_action_space(arm_action, (len(self._robot.arm.joints),))
            self._robot.arm.set_joint_target_velocities(arm_action)
        elif self._action_mode.arm == ArmActionMode.DELTA_JOINT_VELOCITY:
            self._assert_action_space(arm_action, (len(self._robot.arm.joints),))
            cur = np.array(self._robot.arm.get_joint_velocities())
            self._robot.arm.set_joint_target_velocities(cur + arm_action)
        elif self._action_mode.arm == ArmActionMode.ABS_JOINT_POSITION:
            self._assert_action_space(arm_action, (len(self._robot.arm.joints),))
            self._robot.arm.set_joint_target_positions(arm_action)
        elif self._action_mode.arm == ArmActionMode.DELTA_JOINT_POSITION:
            self._assert_action_space(arm_action, (len(self._robot.arm.joints),))
            cur = np.array(self._robot.arm.get_joint_positions())
            self._robot.arm.set_joint_target_positions(cur + arm_action)
        elif self._action_mode.arm == ArmActionMode.ABS_EE_POSE:
            self._assert_action_space(arm_action, (7,))
            self._ee_action(list(arm_action))
        elif self._action_mode.arm == ArmActionMode.DELTA_EE_POSE:
            self._assert_action_space(arm_action, (7,))
            a_x, a_y, a_z, a_qx, a_qy, a_qz, a_qw = arm_action
            x, y, z, qx, qy, qz, qw = self._robot.arm.get_tip().get_pose()
            new_rot = Quaternion(a_qw, a_qx, a_qy, a_qz) * Quaternion(qw, qx, qy, qz)
            qw, qx, qy, qz = list(new_rot)
            new_pose = [a_x + x, a_y + y, a_z + z] + [qx, qy, qz, qw]
            self._ee_action(list(new_pose))
        elif self._action_mode.arm == ArmActionMode.ABS_EE_VELOCITY:
            self._assert_action_space(arm_action, (7,))
            pose = self._robot.arm.get_tip().get_pose()
            new_pos = np.array(pose) + (arm_action * DT)
            self._ee_action(list(new_pos))
        elif self._action_mode.arm == ArmActionMode.DELTA_EE_VELOCITY:
            self._assert_action_space(arm_action, (7,))
            if self._prev_ee_velocity is None:
                self._prev_ee_velocity = np.zeros((7,))
            self._prev_ee_velocity += arm_action
            pose = self._robot.arm.get_tip().get_pose()
            pose = np.array(pose)
            new_pose = pose + (self._prev_ee_velocity * DT)
            self._ee_action(list(new_pose))
        elif self._action_mode.arm == ArmActionMode.ABS_JOINT_TORQUE:
            self._assert_action_space(arm_action, (len(self._robot.arm.joints),))
            self._torque_action(arm_action)
        elif self._action_mode.arm == ArmActionMode.DELTA_JOINT_TORQUE:
            cur = np.array(self._robot.arm.get_joint_forces())
            new_action = cur + arm_action
            self._torque_action(new_action)
        else:
            raise RuntimeError('Unrecognised action mode.')

        if current_ee != ee_action:
            done = False
            while not done:
                done = self._robot.gripper.actuate(ee_action, velocity=0.04)
                self._pyrep.step()
                self._task.step()
            if ee_action == 0.0:
                # If gripper close action, the check for grasp.
                for g_obj in self._task.get_graspable_objects():
                    self._robot.gripper.grasp(g_obj)
            else:
                # If gripper opem action, the check for ungrasp.
                self._robot.gripper.release()

        self._scene.step()

        success, terminate = self._task.success()
        last_step['r'] = int(success)
        next_step = {'s': self._scene.get_observation().get_low_dim_data(), "opt": None}
        return last_step, next_step, terminate

    def name(self) -> str:
        return self._task_name

    # ------------- private methods ------------- #

    def _update_info_dict(self):
        # update info dict
        self._info["action mode"] = self._action_mode
        self._info["observation mode"] = self._obs_config
        # TODO: action dim should related to robot, not action mode, here we fixed it temporally
        self._info["action dim"] = (8,)
        self._info["action low"] = np.zeros(self._info["action dim"], dtype=np.float32) - 1.
        self._info["action high"] = np.zeros(self._info["action dim"], dtype=np.float32) + 1.
        self._info["state dim"] = (73,)
        self._info["state low"] = np.zeros(self._info["state dim"], dtype=np.float32) - 100.
        self._info["state high"] = np.zeros(self._info["state dim"], dtype=np.float32) + 100.
        self._info["reward low"] = -np.inf
        self._info["reward high"] = np.inf

    def _set_arm_control_action(self):
        self._robot.arm.set_control_loop_enabled(True)
        if self._action_mode.arm in (ArmActionMode.ABS_JOINT_VELOCITY, ArmActionMode.DELTA_JOINT_VELOCITY):
            self._robot.arm.set_control_loop_enabled(False)
            self._robot.arm.set_motor_locked_at_zero_velocity(True)
        elif self._action_mode.arm in (ArmActionMode.ABS_JOINT_POSITION, ArmActionMode.DELTA_JOINT_POSITION,
                                       ArmActionMode.ABS_EE_POSE, ArmActionMode.DELTA_EE_POSE,
                                       ArmActionMode.ABS_EE_VELOCITY, ArmActionMode.DELTA_EE_VELOCITY):
            self._robot.arm.set_control_loop_enabled(True)
        elif self._action_mode.arm in (ArmActionMode.ABS_JOINT_TORQUE, ArmActionMode.DELTA_JOINT_TORQUE):
            self._robot.arm.set_control_loop_enabled(False)
        else:
            raise RuntimeError('Unrecognised action mode.')

    def sample_variation(self) -> int:
        self._variation_number = np.random.randint(0, self._task.variation_count())
        return self._variation_number

    def _assert_action_space(self, action, expected_shape):
        if np.shape(action) != expected_shape:
            raise RuntimeError(
                'Expected the action shape to be: %s, but was shape: %s' % (
                    str(expected_shape), str(np.shape(action))))

    def _torque_action(self, action):
        self._robot.arm.set_joint_target_velocities([(TORQUE_MAX_VEL if t < 0 else -TORQUE_MAX_VEL) for t in action])
        self._robot.arm.set_joint_forces(np.abs(action))

    def _ee_action(self, action):
        try:
            joint_positions = self._robot.arm.solve_ik(action[:3], quaternion=action[3:])
            self._robot.arm.set_joint_target_positions(joint_positions)
        except IKError as e:
            raise InvalidActionError("Could not find a path.") from e
        self._pyrep.step()

    @staticmethod
    def _get_class_by_name(class_name: str, model: ModuleType) -> TaskClass:
        all_class_dict = {}
        for o in getmembers(model):
            if isclass(o[1]):
                all_class_dict[o[0]] = o[1]

        if class_name not in all_class_dict:
            raise NotImplementedError(f"No class {class_name} found in {model.__name__} !")
        return all_class_dict[class_name]


if __name__ == "__main__":
    from time import sleep
    e = EnvironmentImpl("CloseMicrowave")
    e.init(True)
    e.reset()
    for i in range(100):
        e.step({'a': np.random.randn(8)})
        sleep(0.1)
