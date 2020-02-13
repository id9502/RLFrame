import torch
import numpy as np
from core.filter.filter import Filter
from core.math.advantage import advantage
from core.common import StepDictList, SampleTraj, SampleBatch, StepDict, List, ParamDict, InfoDict

# from https://github.com/joschu/modular_rl
# http://www.johndcook.com/blog/standard_deviation/


class ZFilter(Filter):
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, advantage_gamma, advantage_tau, clip=10.):
        super(ZFilter, self).__init__()
        self.clip = clip
        self.mean = None
        self.errsum = None
        self.n_step = 0
        self.is_fixed = False
        self.gamma = advantage_gamma
        self.tau = advantage_tau

    def init(self):
        super(ZFilter, self).init()
        self.mean = None
        self.errsum = None
        self.n_step = 0
        self.is_fixed = False

    def finalize(self):
        super(ZFilter, self).finalize()
        self.mean = None
        self.errsum = None
        self.n_step = 0

    def reset(self, param: ParamDict):
        super(ZFilter, self).reset(param)
        self.mean, self.errsum, self.n_step, self.is_fixed =\
            param.require("zfilter mean", "zfilter errsum", "zfilter n_step", "fixed filter")
        if self.mean is not None:
            self.mean = self.mean.cpu().numpy()
            self.errsum = self.errsum.cpu().numpy()

    def operate_currentStep(self, current_step: StepDict) -> StepDict:
        """
        :param x: input state tensor, !! x will be modified inplace !!
        :param self.update: update mean/std estimation
        :return:
        """
        current_step = super(ZFilter, self).operate_currentStep(current_step)
        x = current_step['s']

        if self.mean is None:
            self.mean = x.copy()
            self.errsum = np.zeros_like(self.mean)

        if not self.is_fixed:
            self.n_step += 1
            oldM = self.mean
            self.mean = self.mean + (x - self.mean) / self.n_step
            self.errsum = self.errsum + (x - oldM) * (x - self.mean)

        std = np.sqrt(self.errsum / (self.n_step - 1)) if self.n_step > 1 else self.mean

        x -= self.mean
        x /= std + 1e-8
        if self.clip is not None:
            x = x.clip(-self.clip, self.clip)

        current_step['s'] = x
        return current_step

    def operate_recordStep(self, last_step: StepDict) -> StepDict:
        last_step = super(ZFilter, self).operate_recordStep(last_step)
        return last_step

    def operate_stepList(self, step_list: StepDictList, done: bool) -> SampleTraj:
        traj = super(ZFilter, self).operate_stepList(step_list, done)

        advantage(step_list, self.gamma, self.tau)
        returns = torch.as_tensor([[step["return"]] for step in step_list], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor([[step["advantage"]] for step in step_list], dtype=torch.float32, device=self.device)
        traj["advantages"] = advantages
        traj["returns"] = returns

        traj["filter state dict"] = {"zfilter mean": self.mean.copy(),
                                     "zfilter errsum": self.errsum.copy(),
                                     "zfilter n_step": self.n_step,
                                     "fixed filter": self.is_fixed}
        return traj

    def operate_trajectoryList(self, traj_list: List[SampleTraj]) -> (SampleBatch, InfoDict, ParamDict):
        batch, info = super(ZFilter, self).operate_trajectoryList(traj_list)

        advantages = [b["advantages"] for b in traj_list]
        returns = [b["returns"] for b in traj_list]

        advantages = torch.cat(advantages, dim=0)
        returns = torch.cat(returns, dim=0)
        batch["advantages"] = advantages
        batch["returns"] = returns

        fs = traj_list[0].pop("filter state dict")
        self.mean = fs["zfilter mean"] / len(traj_list)
        self.errsum = fs["zfilter errsum"] / len(traj_list)
        self.n_step = fs["zfilter n_step"]
        for traj in traj_list[1:]:
            fs = traj.pop("filter state dict")
            self.mean += fs["zfilter mean"] / len(traj_list)
            self.errsum += fs["zfilter errsum"] / len(traj_list)
            self.n_step = min(fs["zfilter n_step"], self.n_step)

        return batch, info

    def getStateDict(self) -> ParamDict:
        state_dict = super(ZFilter, self).getStateDict()
        return state_dict + ParamDict({"zfilter mean": torch.as_tensor(self.mean, dtype=torch.float32, device=torch.device("cpu")) if self.mean is not None else None,
                                       "zfilter errsum": torch.as_tensor(self.errsum, dtype=torch.float32, device=torch.device("cpu")) if self.errsum is not None else None,
                                       "zfilter n_step": self.n_step,
                                       "fixed filter": self.is_fixed})

    def to_device(self, device: torch.device):
        super(ZFilter, self).to_device(device)
