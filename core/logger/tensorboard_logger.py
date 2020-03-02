import os
import torch
import numpy as np
from tensorboardX import SummaryWriter
from core.common import ParamDict
from core.utilities import log_dir, t2str
from core.logger.logger import Logger


class TensorBoardLogger(Logger):

    def __init__(self):
        super(TensorBoardLogger, self).__init__()
        self.logger_dict["train"] = self.train_logger
        self.logger_dict["validate"] = self.validate_logger
        self.loggerx = None
        self._log_dir = None
        self._epoch_train = 0
        self._epoch_val = 0
        self.train()

    def __del__(self):
        if self.loggerx is not None:
            self.loggerx.close()

    def init(self, config: ParamDict):
        env_name, tag, short, seed = config.require("env name", "tag", "short", "seed")
        self.default_name = f"{tag}-{env_name}-{short}-{seed}"
        self._log_dir = log_dir(config)
        self.loggerx = SummaryWriter(log_dir=self._log_dir)

        self._epoch_train = 0
        self._epoch_val = 0

    def train(self, train=True):
        if train:
            self.set_type("train")
        else:
            self.set_type("validate")

    def train_logger(self, info: dict):
        if "epoch" in info:
            self._epoch_train = info["epoch"]

        for key in info:
            if key == "epoch" or key == "duration":
                continue
            elif key == "rsums":
                rsums = info["rsums"]

                self.loggerx.add_scalar("r-min", rsums.min(), self._epoch_train)
                self.loggerx.add_scalar("r-max", rsums.max(), self._epoch_train)
                self.loggerx.add_scalar("r-avg", rsums.mean(), self._epoch_train)
            elif key == "steps":
                steps = info["steps"]

                self.loggerx.add_scalar("s-max", steps.max(), self._epoch_train)
                self.loggerx.add_scalar("s-min", steps.min(), self._epoch_train)

            elif isinstance(key, str):
                # adding scalar info
                if isinstance(info[key], (int, float)) or\
                        (isinstance(info[key], (np.ndarray, torch.Tensor, list, tuple)) and len(info[key]) == 1):
                    self.loggerx.add_scalar(key, info[key], self._epoch_train)
                elif isinstance(info[key], (np.ndarray, torch.Tensor, tuple, list)):
                    self.loggerx.add_scalar(f"{key}-max", np.asarray(info[key]).max(), self._epoch_train)
                    self.loggerx.add_scalar(f"{key}-min", np.asarray(info[key]).min(), self._epoch_train)
                    self.loggerx.add_scalar(f"{key}-avg", np.asarray(info[key]).mean(), self._epoch_train)

        self._epoch_train += 1

    def validate_logger(self, info: dict):
        if "epoch" in info:
            self._epoch_val = info["epoch"]

        message = f"I {self._epoch_val:<4}-> "
        if "rsums" in info:
            rsums = info["rsums"]
            message += f"R: min={rsums.min():<8.2f} max={rsums.max():<8.2f} avg={rsums.mean():<8.2f}| "

        if "steps" in info:
            steps = info["steps"]
            message += f"S: {steps.min():<4d}~{steps.max():<4d}; "

        if "duration" in info:
            message += f"Time={t2str(info['duration'])} "

        message += self.default_name
        print(message)

        for key in info:
            if key == "epoch" or key == "duration":
                continue
            elif key == "rsums":
                rsums = info["rsums"]

                self.loggerx.add_scalar("val-r-min", rsums.min(), self._epoch_val)
                self.loggerx.add_scalar("val-r-max", rsums.max(), self._epoch_val)
                self.loggerx.add_scalar("val-r-avg", rsums.mean(), self._epoch_val)

            elif key == "steps":
                steps = info["steps"]

                self.loggerx.add_scalar("val-s-max", steps.max(), self._epoch_val)
                self.loggerx.add_scalar("val-s-min", steps.min(), self._epoch_val)

            elif isinstance(key, str):
                # adding scalar info
                if isinstance(info[key], (int, float)) or\
                        (isinstance(info[key], (np.ndarray, torch.Tensor, list, tuple)) and len(info[key]) == 1):
                    self.loggerx.add_scalar(f"val-{key}", info[key], self._epoch_val)
                elif isinstance(info[key], (np.ndarray, torch.Tensor, tuple, list)):
                    self.loggerx.add_scalar(f"val-{key}-max", np.asarray(info[key]).max(), self._epoch_val)
                    self.loggerx.add_scalar(f"val-{key}-min", np.asarray(info[key]).min(), self._epoch_val)
                    self.loggerx.add_scalar(f"val-{key}-avg", np.asarray(info[key]).mean(), self._epoch_val)

        with open(os.path.join(self._log_dir, f"{self.default_name}.log"), 'a') as f:
            f.write(message)
            f.write('\n')
        self._epoch_val += 1
