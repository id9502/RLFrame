import gym

from core.common.types import StepDict
from core.environment.environment import Environment


class FakeGymEnv(Environment):

    def __init__(self, task_name: str):
        super(FakeGymEnv, self).__init__(task_name)
        self._render = False
        self._env = gym.make(task_name)
        self._info["action dim"] = self._env.action_space.shape
        self._info["action low"] = self._env.action_space.low
        self._info["action high"] = self._env.action_space.high
        self._info["state dim"] = self._env.observation_space.shape
        self._info["state low"] = self._env.observation_space.low
        self._info["state high"] = self._env.observation_space.high
        self._info["reward low"] = self._env.reward_range[0]
        self._info["reward high"] = self._env.reward_range[1]
        self._env.close()
        self._env = None
        pass

    def init(self, display: bool = False) -> bool:
        self._render = display
        self._env = gym.make(self._task_name)
        return True

    def finalize(self) -> bool:
        if self._env is not None:
            self._env.close()
        self._env = None
        return True

    def reset(self, random: bool = True) -> StepDict:
        s = self._env.reset()
        return {'s': s}

    def step(self, last_step: StepDict) -> (StepDict, StepDict, bool):
        s, r, done, info = self._env.step(last_step['a'])
        if self._render:
            self._env.render()
        last_step['r'] = r
        next_step = {'s': s, "info": info}
        return last_step, next_step, done
