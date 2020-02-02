import torch
import random
import numpy as np
from time import sleep
from copy import deepcopy
from core.common import ParamDict
from core.utilities.gpu_tool import decide_device
from torch.multiprocessing import Process, Pipe, Value, Lock
# import interface class instead of its implementation
from core.agent.agent import Agent
from core.model.policy import Policy
from core.filter.filter import Filter
from core.environment.environment import Environment


class Agent_async(Agent):
    """
    An agent class will maintain multiple policy net and multiple environments, each works asynchronously
    useful for most of single agent RL/IL settings
    """
    def __init__(self, config: ParamDict, environment: Environment, policy: Policy, filter_op: Filter):
        threads, gpu = config.require("threads", "gpu")
        threads_gpu = config["gpu threads"] if "gpu threads" in config else 2
        super(Agent_async, self).__init__(config, environment, policy, filter_op)

        # sync signal, -1: terminate, 0: normal running, >0 restart and waiting for parameter update
        self._sync_signal = Value('i', 0)

        # environment sub-process list
        self._environment_proc = []
        # policy sub-process list
        self._policy_proc = []

        # used for synchronize policy parameters
        self._param_pipe = None
        self._policy_lock = Lock()
        # used for synchronize roll-out commands
        self._control_pipe = None
        self._environment_lock = Lock()

        step_pipe = []
        cmd_pipe_child, cmd_pipe_parent = Pipe(duplex=True)
        param_pipe_child, param_pipe_parent = Pipe(duplex=False)
        self._control_pipe = cmd_pipe_parent
        self._param_pipe = param_pipe_parent
        for i_envs in range(threads):
            child_name = f"environment_{i_envs}"
            step_pipe_pi, step_pipe_env = Pipe(duplex=True)
            step_lock = Lock()
            worker_cfg = ParamDict({"seed": self.seed + 1024 + i_envs, "gpu": gpu})
            child = Process(target=Agent_async._environment_worker, name=child_name,
                            args=(worker_cfg, cmd_pipe_child, step_pipe_env, self._environment_lock, step_lock,
                                  self._sync_signal, deepcopy(environment), deepcopy(filter_op)))
            self._environment_proc.append(child)
            step_pipe.append((step_pipe_pi, step_lock))
            child.start()

        for i_policies in range(threads_gpu):
            child_name = f"policy_{i_policies}"
            worker_cfg = ParamDict({"seed": self.seed + 2048 + i_policies, "gpu": gpu})
            child = Process(target=Agent_async._policy_worker, name=child_name,
                            args=(worker_cfg, param_pipe_child, step_pipe,
                                  self._policy_lock, self._sync_signal, deepcopy(policy)))
            self._policy_proc.append(child)
            child.start()
        sleep(5)

    def __del__(self):
        """
        We should terminate all child-process here
        """
        self._sync_signal.value = -1
        sleep(1)

        for _pi in self._policy_proc:
            _pi.join(2)
            if _pi.is_alive():
                _pi.terminate()

        for _env in self._environment_proc:
            _env.join(2)
            if _env.is_alive():
                _env.terminate()

        self._control_pipe.close()
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
        self._sync_signal.value = len(self._policy_proc) + len(self._environment_proc)

        # sync net parameters
        with self._policy_lock:
            for _ in range(len(self._policy_proc)):
                self._param_pipe.send(policy_state)

        # wait for all agents' ready feedback
        while self._sync_signal.value > 0:
            sleep(0.01)

        # sending commands
        with self._environment_lock:
            for _ in range(self._batch_size):
                self._control_pipe.send(cmd)

    def collect(self):
        if self._control_pipe.poll(0.1):
            self._replay_buffer.append(self._control_pipe.recv())
        if len(self._replay_buffer) < self._batch_size:
            return None
        else:
            batch = self._filter.operate_trajectoryList(self._replay_buffer)
            return batch

    @staticmethod
    def _environment_worker(setups: ParamDict, pipe_cmd, pipe_step, read_lock, step_lock, sync_signal, environment, filter_op):
        gpu, seed = setups.require("gpu", "seed")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        environment.init(display=False)
        filter_op.init()
        filter_op.to_device(torch.device("cpu"))
        # -1: syncing, 0: waiting for command, 1: waiting for action
        local_state = 0
        step_buffer = []
        cmd = None

        def _get_piped_data(pipe, lock):
            with lock:
                if pipe.poll(0.001):
                    return pipe.recv()
                else:
                    return None

        while sync_signal.value >= 0:
            # check sync counter for sync event
            if sync_signal.value > 0 and local_state >= 0:
                # receive sync signal, reset all workspace settings, decrease sync counter,
                # and set state machine to -1 for not init again
                while _get_piped_data(pipe_cmd, read_lock) is not None:
                    pass
                while _get_piped_data(pipe_step, step_lock) is not None:
                    pass
                step_buffer.clear()
                with sync_signal.get_lock():
                    sync_signal.value -= 1
                local_state = -1

            # if sync ends, tell state machine to recover from syncing state, and reset environment
            elif sync_signal.value == 0 and local_state == -1:
                local_state = 0

            # idle and waiting for new command
            elif sync_signal.value == 0 and local_state == 0:
                cmd = _get_piped_data(pipe_cmd, read_lock)
                if cmd is not None:
                    step_buffer.clear()
                    cmd.require("fixed environment", "trajectory max step")
                    current_step = environment.reset(random=not cmd["fixed environment"])
                    filter_op.reset(cmd["filter state dict"])

                    policy_step = filter_op.operate_currentStep(current_step)
                    with step_lock:
                        pipe_step.send(policy_step)
                    local_state = 1

            # waiting for action
            elif sync_signal.value == 0 and local_state == 1:
                last_step = _get_piped_data(pipe_step, step_lock)
                if last_step is not None:
                    last_step, current_step, done = environment.step(last_step)
                    record_step = filter_op.operate_recordStep(last_step)
                    step_buffer.append(record_step)
                    if len(step_buffer) >= cmd["trajectory max step"] or done:
                        traj = filter_op.operate_stepList(step_buffer, done=done)
                        with read_lock:
                            pipe_cmd.send(traj)
                        local_state = 0
                    else:
                        policy_step = filter_op.operate_currentStep(current_step)
                        with step_lock:
                            pipe_step.send(policy_step)

        # finalization
        environment.finalize()
        filter_op.finalize()
        pipe_cmd.close()
        pipe_step.close()
        print("Environment sub-process exited")

    @staticmethod
    def _policy_worker(setups: ParamDict, pipe_param, pipe_steps, read_lock, sync_signal, policy):
        gpu, seed = setups.require("gpu", "seed")
        device = decide_device(gpu)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        policy.init()
        policy.to_device(device)
        # -1: syncing, 0: waiting for state
        local_state = 0
        max_batchsz = 8

        def _get_piped_data(pipe, lock):
            with lock:
                if pipe.poll():
                    return pipe.recv()
                else:
                    return None

        while sync_signal.value >= 0:
            # check sync counter for sync event, and waiting for new parameters
            if sync_signal.value > 0:
                # receive sync signal, reset all workspace settings, decrease sync counter,
                # and set state machine to -1 for not init again
                for _pipe, _lock in pipe_steps:
                    while _get_piped_data(_pipe, _lock) is not None:
                        pass
                if local_state >= 0:
                    _policy_state = _get_piped_data(pipe_param, read_lock)
                    if _policy_state is not None:
                        # set new parameters
                        policy.reset(_policy_state)
                        with sync_signal.get_lock():
                            sync_signal.value -= 1
                        local_state = -1
                    else:
                        sleep(0.01)

            # if sync ends, tell state machine to recover from syncing state, and reset environment
            elif sync_signal.value == 0 and local_state == -1:
                local_state = 0

            # waiting for states (states are list of dicts)
            elif sync_signal.value == 0 and local_state == 0:
                idx = []
                data = []
                for i, (_pipe, _lock) in enumerate(pipe_steps):
                    if len(idx) >= max_batchsz:
                        break
                    _steps = _get_piped_data(_pipe, _lock)
                    if _steps is not None:
                        data.append(_steps)
                        idx.append(i)
                if len(idx) > 0:
                    # prepare for data batch
                    with torch.no_grad():
                        data = policy.step(data)
                    # send back actions
                    for i, d in zip(idx, data):
                        with pipe_steps[i][1]:
                            pipe_steps[i][0].send(d)
                else:
                    sleep(0.00001)

        # finalization
        policy.finalize()
        pipe_param.close()
        for _pipe, _lock in pipe_steps:
            _pipe.close()
        print("Policy sub-process exited")

