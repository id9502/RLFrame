from core.common import ParamDict


class Logger(object):

    def __init__(self):
        self.logger_dict = {"default": self.default_logger}
        self.logger_type = "default"
        self.default_name = None

    def init(self, config: ParamDict):
        self.default_name = "default"

    def __call__(self, *args, **kwargs):
        assert self.default_name is not None, "Error: you have not inited logger, abort"
        self.logger_dict[self.logger_type](*args, **kwargs)

    def set_type(self, typename: str):
        if typename not in self.logger_dict:
            raise KeyError(f"Logger '{typename}' has not been defined, {self.logger_dict.keys()} are available")
        self.logger_type = typename

    def default_logger(self, *args, **kwargs):
        print(args, kwargs)
