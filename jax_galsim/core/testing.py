from contextlib import contextmanager
from time import perf_counter_ns


class TimingResult:
    def __init__(self):
        self.dt = None

    def __str__(self):
        if self.dt is None:
            return "- ms"
        else:
            if self.dt > 10000:
                return f"{self.dt/1000} s"
            else:
                return f"{self.dt} ms"


@contextmanager
def time_code_block(msg=None, quiet=False):
    tr = TimingResult()
    t0 = perf_counter_ns()
    yield tr
    t1 = perf_counter_ns()
    tr.dt = (t1 - t0) / 1e6
    if not quiet:
        if msg is not None:
            msg = msg + " "
        else:
            msg = ""
        print(f"{msg}time: {tr.dt} ms")
