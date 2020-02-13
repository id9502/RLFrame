from math import modf
from functools import wraps
from timeit import default_timer as timer


__all__ = ["TimeThis", "time_this", "running_time"]

_running_start_time = timer()


def running_time(fmt=True) -> str:
    return t2str(timer() - _running_start_time, fmt)


def t2str(t, fmt=True) -> str:
    if fmt:
        fp, i = modf(t)
        if t > 60.:
            return "{:d}'{:0>2d}.{:0>3d}".format(int(i // 60), int(i % 60), int(fp * 1000))
        else:
            return "{:d}.{:0>3d}".format(int(i), int(fp * 1000))
    else:
        return "{:.4f}".format(t)


# time a function when anytime it is called
# usage:
#   @time_this(format=True)
#   def your_function(any_arg):
#       ......
#       return any_thing
# or
#   @time_this
#   def your_function(any_arg):
#       ......
#       return any_thing
def time_this(fmt=True):
    if isinstance(fmt, bool):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                __t_start__ = timer()
                ret_val = func(*args, **kwargs)
                print("{}: {}".format(func.__name__, t2str(timer() - __t_start__, fmt)))
                return ret_val
            return wrapper
        return decorator
    else:
        @wraps(fmt)
        def wrapper(*args, **kwargs):
            __t_start__ = timer()
            ret_val = fmt(*args, **kwargs)
            print("{}: {}".format(fmt.__name__, t2str(timer() - __t_start__, True)))
            return ret_val
        return wrapper


# time a context, or time between this class created and freed, or any checkpoint
# usage:
#   with TimeThis(format=True):
#       do_something()
# or
#   with TimeThis(format=True) as t:
#       do_something()
#       t.elapse()
# or
#   t = TimeThis()
#   t.check_point("Point 1")
#   t.check_point("Point 2")
#   t.check_point("Point 1")
#   t.check_point("Point 2")
class TimeThis(object):
    def __init__(self, fmt=True, name=None, show_when_del=False):
        self.__life_time = timer()
        self.__context_time = None
        self.__check_point = {}
        self.name = name
        self.fmt = fmt
        self.show_when_del = show_when_del

    def __enter__(self):
        self.__context_time = timer()
        self.name = "Context" if self.name is None else self.name
        return self

    def __exit__(self, type_, value, traceback):
        self.__context_time = None
        print("{}: {}".format(self.name, t2str(timer() - self.__life_time, self.fmt)))

    def __del__(self):
        if self.show_when_del:
            print(t2str(timer() - self.__life_time, self.fmt))

    def elapse(self, t=None, show=True):
        if t is not None:
            prefix = "Duration"
            dt = timer() - t
        elif self.__context_time is not None:
            prefix = "Context"
            dt = timer() - self.__context_time
        else:
            prefix = self.name if self.name is not None else "Duration"
            dt = timer() - self.__life_time
        if show:
            print("{}: {}".format(prefix, t2str(dt, self.fmt)))
        return dt

    def check_point(self, name="", update=True, show=True):
        t = timer()
        dt = -1.
        if name in self.__check_point:
            dt = t - self.__check_point[name]
        if update:
            self.__check_point[name] = t
        if dt > 0. and show:
            if name != "":
                print("{}: {}".format(name, t2str(dt, self.fmt)))
            else:
                print("Duration: {}".format(t2str(dt, self.fmt)))
        return dt


if __name__ == "__main__":
    import time
    t = TimeThis(show_when_del=True)
    time.sleep(2)
