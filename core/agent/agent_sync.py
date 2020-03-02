import torch
import random
import numpy as np
from time import sleep
from copy import deepcopy
from torch.multiprocessing import Process, Pipe, Value, Lock
from core.common.config import ParamDict
from core.utilities import decide_device
# import interface class instead of its implementation
from core.agent.agent import Agent
from core.model.policy import Policy
from core.filter.filter import Filter
from core.environment.environment import Environment


class Agent_sync(Agent):
    """
    An agent class will maintain multiple policy net and environments, each worker will have one environment and one policy
    useful for most of single agent RL/IL settings
    """
    def __init__(self, config: ParamDict, environment: Environment, policy: Policy, filter_op: Filter):
        threads, gpu = config.require("threads", "gpu")
        super(Agent_sync, self).__init__(config, environment, policy, filter_op)

        # sync signal, -1: terminate, 0: normal running, >0 restart and waiting for parameter update
        self._sync_signal = Value('i', 0)

        # sampler sub-process list
        self._sampler_proc = []

        # used for synchronize commands
        self._cmd_pipe = None
        self._param_pipe = None
        self._cmd_lock = Lock()

        cmd_pipe_child, cmd_pipe_parent = Pipe(duplex=True)
        param_pipe_child, param_pipe_parent = Pipe(duplex=False)
        self._cmd_pipe = cmd_pipe_parent
        self._param_pipe = param_pipe_parent
        for i_thread in range(threads):
            child_name = f"sampler_{i_thread}"
            worker_cfg = ParamDict({"seed": self.seed + 1024 + i_thread, "gpu": gpu})
            child = Process(target=Agent_sync._sampler_worker, name=child_name,
                            args=(worker_cfg, cmd_pipe_child, param_pipe_child,
                                  self._cmd_lock, self._sync_signal, deepcopy(policy),
                                  deepcopy(environment), deepcopy(filter_op)))
            self._sampler_proc.append(child)
            child.start()

    def __del__(self):
        """
        We should terminate all child-process here
        """
        self._sync_signal.value = -1
        sleep(1)
        for _proc in self._sampler_proc:
            _proc.join(2)
            if _proc.is_alive():
                _proc.terminate()

        self._cmd_pipe.close()
        self._param_pipe.close()

    def broadcast(self, config: ParamDict):
        policy_state, filter_state, max_step, self._batch_size, fixed_env, fixed_policy, fixed_filter = \
            config.require("policy state dict", "filter state dict", "trajectory max step", "batch size",
                           "fixed environment", "fixed policy", "fixed filter")

        self._replay_buffer = []
        policy_state["fixed policy"] = fixed_policy
        filter_state["fixed filter"] = fixed_filter
        cmd = ParamDict({"trajectory max step": max_step,
                         "fixed environment": fixed_env,
                         "filter state dict": filter_state})

        assert self._sync_signal.value < 1, "Last sync event not finished due to some error, some sub-proc maybe died, abort"
        # tell sub-process to reset
        with self._sync_signal.get_lock():
            self._sync_signal.value = len(self._sampler_proc)

        # sync net parameters
        with self._cmd_lock:
            for _ in range(len(self._sampler_proc)):
                self._param_pipe.send(policy_state)

        # wait for all agents' ready feedback
        while self._sync_signal.value > 0:
            sleep(0.01)

        # sync commands
        for _ in range(self._batch_size):
            self._cmd_pipe.send(cmd)

    def collect(self):
        if self._cmd_pipe.poll(0.1):
            self._replay_buffer.append(self._cmd_pipe.recv())
        if len(self._replay_buffer) < self._batch_size:
            return None
        else:
            batch = self._filter.operate_trajectoryList(self._replay_buffer)
            return batch

    @staticmethod
    def _sampler_worker(setups: ParamDict, pipe_cmd, pipe_param, read_lock, sync_signal, policy, environment, filter_op):
        gpu, seed = setups.require("gpu", "seed")

        device = decide_device(gpu)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        environment.init(display=False)
        filter_op.to_device(torch.device("cpu"))
        filter_op.init()
        policy.to_device(device)
        policy.init()

        # -1: syncing, 0: waiting for new command, 1: sampling
        local_state = 0
        current_step = None
        step_buffer = []
        cmd = None

        def _get_piped_data(pipe):
            with read_lock:
                if pipe.poll(0.001):
                    return pipe.recv()
                else:
                    return None

        while sync_signal.value >= 0:
            # check sync counter for sync event, and waiting for new parameters
            if sync_signal.value > 0 and local_state >= 0:
                # receive sync signal, reset all workspace settings, decrease sync counter,
                # and set state machine to -1 for not init again
                while _get_piped_data(pipe_cmd) is not None:
                    pass
                step_buffer.clear()
                _policy_state = _get_piped_data(pipe_param)
                if _policy_state is not None:
                    # set new parameters
                    policy.reset(_policy_state)
                    with sync_signal.get_lock():
                        sync_signal.value -= 1
                    local_state = -1

            # if sync ends, tell state machine to recover from syncing state, and reset environment
            elif sync_signal.value == 0 and local_state == -1:
                local_state = 0

            # waiting for states (states are list of dicts)
            elif sync_signal.value == 0 and local_state == 0:
                cmd = _get_piped_data(pipe_cmd)
                if cmd is not None:
                    step_buffer.clear()
                    cmd.require("filter state dict", "fixed environment", "trajectory max step")
                    current_step = environment.reset(random=not cmd["fixed environment"])
                    filter_op.reset(cmd["filter state dict"])
                    local_state = 1

            # sampling
            elif sync_signal.value == 0 and local_state == 1:
                with torch.no_grad():
                    policy_step = filter_op.operate_currentStep(current_step)
                    last_step = policy.step([policy_step])[0]
                last_step, current_step, done = environment.step(last_step)
                record_step = filter_op.operate_recordStep(last_step)
                step_buffer.append(record_step)

                if len(step_buffer) >= cmd["trajectory max step"] or done:
                    traj = filter_op.operate_stepList(step_buffer, done=done)
                    with read_lock:
                        pipe_cmd.send(traj)
                    local_state = 0

        # finalization
        filter_op.finalize()
        policy.finalize()
        environment.finalize()
        pipe_cmd.close()
        pipe_param.close()
        print("Sampler sub-process exited")
