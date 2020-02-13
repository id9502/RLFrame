import io
import os
import sys
from contextlib import contextmanager


def _is_in_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        pass
    return False


@contextmanager
def suppress_stdout():
    """Used for suppressing std out/err.

    This is needed because the OMPL plugin outputs logging info even when
    logging is turned off.
    """
    try:
        # If we are using an IDE, then this will fail
        original_stdout_fd = sys.stdout.fileno()
        original_stderr_fd = sys.stderr.fileno()
    except io.UnsupportedOperation:
        # Nothing we can do about this, just don't suppress
        yield
        return

    if _is_in_ipython():
        yield
        return

    with open(os.devnull, "w") as devnull:

        devnull_fd = devnull.fileno()

        def _redirect_stdout(to_fd):
            sys.stdout.close()
            os.dup2(to_fd, original_stdout_fd)

            sys.stdout = os.fdopen(original_stdout_fd, 'w')

        def _redirect_stderr(to_fd):
            sys.stderr.close()
            os.dup2(to_fd, original_stderr_fd)

            sys.stderr = os.fdopen(original_stderr_fd, 'w')

        saved_stdout_fd = os.dup(original_stdout_fd)
        saved_stderr_fd = os.dup(original_stderr_fd)

        try:
            _redirect_stdout(devnull_fd)
            _redirect_stderr(devnull_fd)
            yield
            _redirect_stdout(saved_stdout_fd)
            _redirect_stderr(saved_stderr_fd)
        finally:
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)
