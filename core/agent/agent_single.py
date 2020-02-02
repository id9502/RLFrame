from core.common import ParamDict
# import interface class instead of its implementation
from core.environment.environment import Environment
from core.model.policy import Policy
from core.filter.filter import Filter
from core.agent.agent import Agent


class Agent_single(Agent):
    """
    An agent class will maintain multiple policy net and environments, each worker will have one environment and one policy
    useful for most of single agent RL/IL settings
    """
    def __init__(self, config: ParamDict, environment: Environment, policy: Policy, filter_op: Filter):
        super(Agent_single, self).__init__(config, environment, policy, filter_op)

        self._fixed_env = True
        self._max_step = 1000
        self._batch_sz = 0

        self._environment.init(display=False)

    def __del__(self):
        self._environment.finalize()
        print("Agent exited")

    def broadcast(self, config: ParamDict):
        policy_state, filter_state, self._max_step, self._batch_size, self._fixed_env, fixed_policy, fixed_filter = \
            config.require("policy state dict", "filter state dict", "trajectory max step",
                           "batch size", "fixed environment", "fixed policy", "fixed filter")

        self._replay_buffer = []
        policy_state["fixed policy"] = fixed_policy
        filter_state["fixed filter"] = fixed_filter

        self._filter.reset(filter_state)
        self._policy.reset(policy_state)

    def collect(self):
        for _ in range(self._batch_size):
            step_buffer = []

            current_step = self._environment.reset(random=not self._fixed_env)
            done = False

            for _ in range(self._max_step):
                policy_step = self._filter.operate_currentStep(current_step)
                current_step = self._policy.step([policy_step])[0]
                last_step, current_step, done = self._environment.step(current_step)
                record_step = self._filter.operate_recordStep(last_step)
                step_buffer.append(record_step)
                if done:
                    break

            traj = self._filter.operate_stepList(step_buffer, done=done)
            self._replay_buffer.append(traj)

        batch, info = self._filter.operate_trajectoryList(self._replay_buffer)
        return batch, info
