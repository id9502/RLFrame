import torch
from core.common import StringList, StepDictList, InfoDict, ParamDict


class Policy(object):

    def __init__(self):
        self._info = {}
        self._description = [""]
        self.device = torch.device("cpu")

    def init(self):
        raise NotImplementedError

    def finalize(self):
        raise NotImplementedError

    def step(self, current_step_list: StepDictList) -> StepDictList:
        """
        :param current_step_list: each element should contain 's'
        :return: current step list with estimated 'a'
        """
        raise NotImplementedError

    def reset(self, param: ParamDict):
        raise NotImplementedError

    def getStateDict(self) -> ParamDict:
        raise NotImplementedError

    def info(self) -> InfoDict:
        return self._info

    def description(self) -> StringList:
        return self._description

    def to_device(self, device: torch.device):
        self.device = device

