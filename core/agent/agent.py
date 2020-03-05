import torch
import random
import numpy as np
from time import sleep
from core.common import SampleBatch, Union, InfoDict, ParamDict, Tuple
from copy import deepcopy
from core.utilities import decide_device
# import interface class instead of its implementation
from core.model.policy import Policy
from core.filter.filter import Filter
from core.environment.environment import Environment


class Agent(object):
    """
    This is the class interface, you should not instantiation this class directly
     """
    def __init__(self, config: ParamDict, environment: Environment, policy: Policy, filter_op: Filter):
        seed, gpu = config.require("seed", "gpu")
        # replay buffer
        self._replay_buffer = []
        self._batch_size = 0

        # device and seed
        self.device = decide_device(gpu)
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # policy which will be copied to child thread before each roll-out
        self._policy = deepcopy(policy)
        # environment which will be copied to child thread and not inited in main thread
        self._environment = deepcopy(environment)
        # filter which will be copied to child thread and also be kept in main thread
        self._filter = deepcopy(filter_op)

        self._filter.to_device(self.device)
        self._filter.init()
        self._policy.to_device(self.device)
        self._policy.init()

    def __del__(self):
        self._filter.finalize()
        self._policy.finalize()
        self._replay_buffer.clear()

    def broadcast(self, *arg, **argv) -> None:
        """
        Tell samplers to start parallel sampling with new policy,
        non-block function whose results can be obtained by collect()
        TO BE CAREFUL: torch state_dict always do shallow copy, so keep the net constant during broadcasting !!
        :param arg:
        :param argv:
        :return:
        """
        raise NotImplementedError

    def collect(self) -> Union[None, Tuple[SampleBatch, InfoDict, ParamDict]]:
        """
        Return batches of samples, will return None if not ready, used after calling broadcast()
        :return:
        """
        raise NotImplementedError

    def rollout(self, *arg, **argv) -> Tuple[SampleBatch, InfoDict, ParamDict]:
        """
        Similar as calling broadcast + collect
        :param arg:
        :param argv:
        :return:
        """
        self.broadcast(*arg, **argv)
        samples = None
        while samples is None:
            # decreasing query frequency
            sleep(0.1)
            samples = self.collect()
        return samples

    def policy(self):
        """
        Return reference of policy used in this agent
        :return:
        """
        return self._policy

    def filter(self):
        """
        Return reference of filter used in this agent
        :return:
        """
        return self._filter

    # environment and policy should be inited outside this call, policy should also be reset
    def verify(self, config: ParamDict):
        policy_state, filter_state, max_step, max_iter, display, fixed_env, fixed_policy, fixed_filter =\
             config.require("policy state dict", "filter state dict", "trajectory max step", "max iter", "display",
                            "fixed environment", "fixed policy", "fixed filter")

        policy_state["fixed policy"] = fixed_policy
        filter_state["fixed filter"] = fixed_filter

        self._environment.init(display=display)
        self._policy.reset(policy_state)
        self._filter.reset(filter_state)

        r_sum = []
        for i in range(max_iter):
            current_step = self._environment.reset(random=not fixed_env)
            # sampling
            r_sum.append(0.)
            i_step = 0
            for _ in range(max_step):
                current_step = self._filter.operate_currentStep(current_step)
                last_step = self._policy.step([current_step])[0]
                last_step, current_step, done = self._environment.step(last_step)
                record_step = self._filter.operate_recordStep(last_step)
                r_sum[-1] += record_step['r']
                i_step += 1
                if done:
                    break
            print(f"Verification iter {i}: reward={r_sum[-1]}; step={i_step}")
        # finalization
        self._environment.finalize()
        r_sum = np.asarray(r_sum, dtype=np.float32)
        r_sum_mean = r_sum.mean()
        r_sum_std = r_sum.std()
        r_sum_max = r_sum.max()
        r_sum_min = r_sum.min()
        print(f"Verification done with reward sum: avg={r_sum_mean}, std={r_sum_std}, min={r_sum_min}, max={r_sum_max}")
        return r_sum_mean, r_sum_std, r_sum_min, r_sum_max


if __name__ == "__main__":
    from multiprocessing import set_start_method
    set_start_method("spawn")
