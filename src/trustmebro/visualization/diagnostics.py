"""Opt-in, process-owned timing logs; no shared writers or measurement changes."""

from __future__ import annotations

import csv
import os
import time
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import cast


class TimingLog:
    """Buffered spans distinguish wall, process CPU, and calling-thread CPU time.

    CPU includes executor management threads in the coordinator. Nested spans
    overlap, so totals must be compared by category, not summed indiscriminately.
    Flush at theorem boundaries to retain completed work after forced shutdown.
    """

    def __init__(self, dir: Path, role: str):
        dir.mkdir(parents=True, exist_ok=True)
        self.stream = (dir / f"{role}-{os.getpid()}.csv").open("x", newline="", buffering=65536)
        self.writer = csv.writer(self.stream)
        self.writer.writerow(("label", "phase", "start_ns", "wall_ns", "cpu_ns", "thread_ns", "ok"))
        self.flush()

    def flush(self) -> None:
        self.stream.flush()

    def close(self) -> None:
        self.stream.close()


@contextmanager
def checkpoint(log: TimingLog | None, label: str, phase: str) -> Generator[None]:
    """Time a complete operation, never a generator suspension across consumer work."""
    if log is None:
        yield
        return
    start, cpu, thread = time.perf_counter_ns(), time.process_time_ns(), time.thread_time_ns()
    ok = False
    try:
        yield
        ok = True
    finally:
        elapsed, used = time.perf_counter_ns() - start, time.process_time_ns() - cpu
        thread_used = time.thread_time_ns() - thread
        log.writer.writerow((label, phase, start, elapsed, used, thread_used, int(ok)))


def timed[**P, T](
    log: TimingLog | None, label: str, phase: str, fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs
) -> T:
    """Disabled diagnostics perform no clock reads, encoding, or file I/O."""
    if log is None:
        return fn(*args, **kwargs)
    with checkpoint(log, label, phase):
        return fn(*args, **kwargs)


def timed_batches[T](log: TimingLog | None, label: str, phase: str, batches: Iterable[T]) -> Generator[T]:
    """Measure producing each batch; serialization/consumer time belongs to other spans."""
    if log is None:
        yield from batches
        return
    iterator = iter(batches)
    sentinel = object()
    while (batch := timed(log, label, phase, next, iterator, sentinel)) is not sentinel:
        yield cast(T, batch)
