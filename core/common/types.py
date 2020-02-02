from typing import Dict, List

__all__ = ["StepDict", "InfoDict", "StringList", "StepDictList", "SampleTraj", "SampleBatch"]


# StepDict: should at least have key ['s', 'a', 'r']
StepDict = Dict
# InfoDict: dict contains information names and values
InfoDict = Dict
# StringList: [str, str....]
StringList = List[str]
# StepDictList: [StepDict, StepDict...]
StepDictList = List[StepDict]
# SampleTraj: should at least have key ["trajectory", "done", "total steps", "reward sum"]
SampleTraj = Dict
# SampleTrajBatch: anything that will be used in following training step
SampleBatch = Dict
