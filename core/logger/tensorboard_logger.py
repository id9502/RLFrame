import os
from tensorboardX import SummaryWriter
from core.common import ParamDict
from core.utilities.path_tool import log_dir
from core.utilities.timeit_tool import t2str
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

        if "rsums" in info:
            rsums = info["rsums"]

            self.loggerx.add_scalar("r-min", rsums.min(), self._epoch_train)
            self.loggerx.add_scalar("r-max", rsums.max(), self._epoch_train)
            self.loggerx.add_scalar("r-avg", rsums.mean(), self._epoch_train)

        if "steps" in info:
            steps = info["steps"]

            self.loggerx.add_scalar("s-max", steps.max(), self._epoch_train)
            self.loggerx.add_scalar("s-min", steps.min(), self._epoch_train)

        self._epoch_train += 1

    def validate_logger(self, info: dict):
        if "epoch" in info:
            self._epoch_val = info["epoch"]

        message = f"I {self._epoch_val:<4}-> "
        if "rsums" in info:
            rsums = info["rsums"]
            message += f"R: min={rsums.min():<8.2f} max={rsums.max():<8.2f} avg={rsums.mean():<8.2f}| "

            self.loggerx.add_scalar("val-r-min", rsums.min(), self._epoch_val)
            self.loggerx.add_scalar("val-r-max", rsums.max(), self._epoch_val)
            self.loggerx.add_scalar("val-r-avg", rsums.mean(), self._epoch_val)

        if "steps" in info:
            steps = info["steps"]
            message += f"S: {steps.min():<4d}~{steps.max():<4d}; "

            self.loggerx.add_scalar("val-s-max", steps.max(), self._epoch_val)
            self.loggerx.add_scalar("val-s-min", steps.min(), self._epoch_val)

        if "duration" in info:
            message += f"Time={t2str(info['duration'])} "

        message += self.default_name
        print(message)
        with open(os.path.join(self._log_dir, f"{self.default_name}.log"), 'a') as f:
            f.write(message)
            f.write('\n')
        self._epoch_val += 1
