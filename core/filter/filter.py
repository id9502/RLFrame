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
        return current_step

    def operate_recordStep(self, last_step: StepDict) -> StepDict:
        """
        decorate last stepDict before putting to step memory
        """
        return last_step

    def operate_stepList(self, step_list: StepDictList, done: bool) -> SampleTraj:
        """
        decorate step memory of one roll-out epoch and form single trajectory dict,
                 must contain keyword "trajectory", "done", "length", "reward sum"
        """
        states = torch.as_tensor([step['s'] for step in step_list], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([step['a'] for step in step_list], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor([[step['r']] for step in step_list], dtype=torch.float32, device=self.device)
        sample_traj = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "step": rewards.nelement(),
            "rsum": rewards.sum(dtype=torch.float32),
            "done": done
        }
        if "display reward" in step_list[0]["info"]:
            display_reward = torch.as_tensor([[step["info"]["display reward"]] for step in step_list],
                                             dtype=torch.float32, device=self.device)
            sample_traj["display rewards"] = display_reward
            sample_traj["display rsum"] = display_reward.sum(dtype=torch.float32)
        return sample_traj

    def operate_trajectoryList(self, traj_list: List[SampleTraj]) -> (SampleBatch, InfoDict):
        """
        decorate trajectory list before return to user
        """
        states = [b["states"] for b in traj_list]
        actions = [b["actions"] for b in traj_list]
        rewards = [b["rewards"] for b in traj_list]

        states = torch.cat(states, dim=0)
        actions = torch.cat(actions, dim=0)
        rewards = torch.cat(rewards, dim=0)

        steps = torch.as_tensor([b["step"] for b in traj_list], dtype=torch.int, device=self.device)
        rsums = torch.as_tensor([b["rsum"] for b in traj_list], dtype=torch.float32, device=self.device)

        batch = {"states": states,
                 "actions": actions,
                 "rewards": rewards}
        info = {"rsums": rsums,
                "steps": steps}

        if "display rewards" in traj_list[0]:
            display_rsums = torch.as_tensor([b["display rsum"] for b in traj_list], dtype=torch.float32, device=self.device)
            info["display rsums"] = display_rsums
        return batch, info

    def getStateDict(self) -> ParamDict:
        return ParamDict()

    def to_device(self, device: torch.device):
        self.device = device
