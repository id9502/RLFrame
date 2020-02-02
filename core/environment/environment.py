from core.common.types import StepDict, InfoDict, StringList


__all__ = []


class Environment(object):

    def __init__(self, task_name: str):
        self._task_name = task_name
        self._description = [""]
        self._info = {}

    def init(self, display: bool = False) -> bool:
        raise NotImplementedError

    def finalize(self) -> bool:
        raise NotImplementedError

    def reset(self, random: bool = True) -> StepDict:
        """
        reset the environment
        :param random: is the task initialized randomly
        :return: first step
        """
        raise NotImplementedError

    def step(self, last_step: StepDict) -> (StepDict, StepDict, bool):
        """
        add reward and other method into last step, and return next step
        :param last_step: last_step dict
        :return: last_step, next_step, done
        """
        raise NotImplementedError

    def name(self) -> str:
        return self._task_name

    def description(self) -> StringList:
        return self._description

    def info(self) -> InfoDict:
        return self._info

