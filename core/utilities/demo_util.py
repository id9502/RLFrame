import os
import bz2
import pickle
import numpy as np
from core.common import InfoDict

__all__ = ["DemoLoader"]


class DemoLoader(object):

    def __init__(self):
        self._file_path = ""
        self._info = {}
        self._trajectory = None

    def load_file(self, file_path: str):
        self._file_path = file_path
        if not os.path.isfile(file_path):
            print(f"Warning: no file found with path {file_path}")
            self._info = {}
            self._trajectory = None
            return

        with bz2.BZ2File(file_path, "rb") as f:
            batch = pickle.load(f)

        if not isinstance(batch, dict) or "trajectory" not in batch:
            print(f"Warning: demo loaded with unsupported format, try using 'tool/demo_converter.py' to convert it")
            self._info = {}
            self._trajectory = None
            return

        self._trajectory = batch["trajectory"]
        self._info = batch

    def __len__(self):
        if self._trajectory is not None:
            return len(self._trajectory)
        else:
            return -1

    def generate_all(self):
        return self._trajectory

    def generate_batch(self, n_traj: int = -1, random: bool = True, ignore_last: bool = False):
        if self._trajectory is None:
            yield None

        if random:
            np.random.shuffle(self._trajectory)
        if n_traj <= 0:
            yield self._trajectory
        else:
            idx = 0
            while idx + n_traj <= len(self._trajectory):
                yield self._trajectory[idx: idx+n_traj]
            if not ignore_last:
                yield self._trajectory[idx:]
        return

    def info(self) -> InfoDict:
        return self._info
