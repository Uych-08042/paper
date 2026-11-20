"""Print progress for resumable experiments, including in redirected logs."""

import sys
from time import monotonic


class ExperimentProgress:
    def __init__(self, label, total, reused=0, interval=10, unit="searches"):
        if not 0 <= reused <= total or total <= 0 or interval <= 0:
            raise ValueError("Invalid experiment progress budget or interval")
        self.label, self.total, self.reused = label, total, reused
        self.interval, self.unit = interval, unit
        self.done = reused
        self.start = self.last = monotonic()

    def advance(self, message, count=1):
        self.done += count
        if self.done > self.total:
            raise ValueError("Experiment exceeded its progress budget")
        if monotonic()-self.last >= self.interval:
            self.report(message)

    def report(self, message):
        now = monotonic()
        elapsed = now-self.start
        measured = self.done-self.reused
        remaining = self.total-self.done
        eta = remaining*elapsed/measured if measured else None
        eta_text = "00:00:00" if not remaining else self.duration(eta) if eta is not None else "estimating"
        print(f"[{self.label}] {self.done:,}/{self.total:,} {self.unit} ({self.done/self.total:.1%}) | "
              f"reused {self.reused:,} | elapsed {self.duration(elapsed)} | ETA {eta_text} | {message}", flush=True)
        self.last = now

    @staticmethod
    def duration(seconds):
        hours, remainder = divmod(round(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class OverallProgress(ExperimentProgress):
    """One in-place bar on a terminal, periodic snapshots in background logs."""

    phase = "Preparing"

    def advance(self, message="", count=1):
        super().advance(self.phase,count)

    def report(self, message=""):
        if message:
            self.phase = message
        now = monotonic()
        elapsed = now-self.start
        measured = self.done-self.reused
        remaining = self.total-self.done
        eta = remaining*elapsed/measured if measured else None
        eta_text = "00:00:00" if not remaining else self.duration(eta) if eta is not None else "estimating"
        fraction = self.done/self.total
        filled = int(fraction*24)
        bar = "="*filled+"-"*(24-filled)
        line = (f"[{self.label}] [{bar}] {fraction:6.2%} {self.done:,}/{self.total:,} | "
                f"{self.phase} | reused {self.reused:,} | {self.duration(elapsed)} elapsed | ETA {eta_text}")
        if sys.stdout.isatty():
            print("\r\033[K"+line, end="\n" if not remaining else "",flush=True)
        else:
            print(line,flush=True)
        self.last = now
