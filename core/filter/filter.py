import torch
from core.common import StepDict, StepDictList, SampleTraj, SampleBatch, List, InfoDict, ParamDict


class Filter(object):
    """
    Filter class will modify the stepDict passing:
            operate_currentStep: from environment to policy, you can do ZFiltering/compression/mapping on states and so-on
            operate_recordStep: from last_step to record_step, you can decide what to be kept into step list memory
            operate_stepList: from step list to SampleTraj, you can decide what to be used for forming a trajectory
            operate_trajectoryList: from SampleTraj to SampleBatch
    Be careful: filter will be copied to separate threads and the pipeline will be called in different places,
                do not save changes inside class, but save them into return values
    """

    def __init__(self):
        self.device = torch.device("cpu")
        pass

    def init(self):
        pass

    def finalize(self):
        pass

    def reset(self, param: ParamDict):
        pass

    def operate_currentStep(self, current_step: StepDict) -> StepDict:
        """
        decorate current stepDict before transferring to policy net
        """
        current_step['s'] = torch.as_tensor(current_step['s'], dtype=torch.float32, device=self.device)
        return current_step

    def operate_recordStep(self, last_step: StepDict) -> StepDict:
        """
        decorate last stepDict before putting to step memory
        """
        last_step['a'] = torch.as_tensor(last_step['a'], dtype=torch.float32, device=self.device)
        last_step['r'] = torch.as_tensor([last_step['r']], dtype=torch.float32, device=self.device)
        return last_step

    def operate_stepList(self, step_list: StepDictList, done: bool) -> SampleTraj:
        """
        decorate step memory of one roll-out epoch and form single trajectory dict,
                 must contain keyword "trajectory", "done", "length", "reward sum"
        """
        states = torch.stack([step['s'] for step in step_list], dim=0)
        actions = torch.stack([step['a'] for step in step_list], dim=0)
        rewards = torch.stack([step['r'] for step in step_list], dim=0)
        return {"states": states,
                "actions": actions,
                "rewards": rewards,
                "step": rewards.nelement(),
                "rsum": rewards.sum(dtype=torch.float32),
                "done": done}

    def operate_trajectoryList(self, traj_list: List[SampleTraj]) -> (SampleBatch, InfoDict, ParamDict):
        """
        decorate trajectory list before return to user
        """
        states = [b["states"] for b in traj_list]
        actions = [b["actions"] for b in traj_list]
        rewards = [b["rewards"] for b in traj_list]

        states = torch.cat(states, dim=0)
        actions = torch.cat(actions, dim=0)
        rewards = torch.cat(rewards, dim=0)

        steps = torch.as_tensor([b["step"] for b in traj_list], dtype=torch.int)
        rsums = torch.as_tensor([b["rsum"] for b in traj_list], dtype=torch.float32)

        batch = {"states": states,
                 "actions": actions,
                 "rewards": rewards}
        info = {"rsums": rsums,
                "steps": steps}
        return batch, info

    def getStateDict(self) -> ParamDict:
        return ParamDict()

    def to_device(self, device: torch.device):
        self.device = device
