"""tqdm progress bars that coexist with logging.

Console log lines go through tqdm.write, so they print above an active bar instead of breaking it. Records logged
with extra={"per_run": True} (one line per finished item) always reach the log file, but reach the console only
when no bar is drawn there: stderr is not a terminal (tqdm then hides the bar) or -v is on.
"""
import logging
import sys

from tqdm import tqdm


def bar(desc, unit="run"):
    """A bar whose total grows as work is queued (bar.total += 1). Hidden when stderr is not a terminal."""
    return tqdm(total=0, desc=desc, unit=unit, file=sys.stderr, dynamic_ncols=True, disable=None, smoothing=0.05)


class ConsoleHandler(logging.StreamHandler):
    def __init__(self, verbose=False, stream=None):
        super().__init__(stream or sys.stderr)
        self.verbose = verbose

    def filter(self, record):
        if getattr(record, "per_run", False) and not self.verbose and self.stream.isatty():
            return False
        return super().filter(record)

    def emit(self, record):
        try:
            tqdm.write(self.format(record), file=self.stream)
        except Exception:  # noqa: BLE001
            self.handleError(record)


def run_stats(s):
    """Postfix for a trace bar from counters: runs, utility, attack, security, error."""
    if not s["runs"]:
        return ""
    out = "util %d%%" % round(100 * s["utility"] / s["runs"])
    if s["attack"]:
        out += " · ASR %d%% of %d" % (round(100 * s["security"] / s["attack"]), s["attack"])
    return out + " · err %d" % s["error"]
