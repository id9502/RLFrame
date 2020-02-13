import torch
from core.common import ParamDict, InfoDict


__all__ = ["DemoDumper", "DemoLoader", "DemoPlayBack"]


class DemoLoader(object):

    def __init__(self, file_path):
        self._info = {}

    def info(self) -> InfoDict:
        return self._info


class DemoDumper(object):

    def __init__(self, file_name):
        pass


class DemoPlayBack(object):

    def __init__(self, file_path):
        pass
