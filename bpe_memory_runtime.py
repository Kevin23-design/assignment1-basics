"""Public full-text dispatch: Q3 with clean workers, or the Python compatibility path."""
from __future__ import annotations
from contextlib import contextmanager
import os
import signal
import threading
from parallel_bpe_encoding import affinity

def _partition_ranges(tokenizer, text, count):
    """Only cut after non-overlapping actual special matches, using BPE's priority."""
    if count < 1:
        raise ValueError('Partition count must be positive')
    if not text:
        return None
    if count == 1:
        return [(0, len(text))]
    if tokenizer._special_re is None:
        return None
    targets = [len(text) * index // count for index in range(1, count)]
    cuts, index = [0], 0
    for match in tokenizer._special_re.finditer(text):
        while index < len(targets) and match.start() >= targets[index]:
            if match.end() < len(text) and cuts[-1] != match.end():
                cuts.append(match.end())
            index += 1
        if index == len(targets):
            break
    cuts.append(len(text))
    return list(zip(cuts, cuts[1:])) if len(cuts) == count + 1 else None


def encode_if_parallel(tokenizer, text, workers, multiplier):
    """None means an inapplicable input/platform; worker or resource failures propagate."""
    if not hasattr(os, 'sched_getaffinity') or not hasattr(os, 'sched_setaffinity'):
        return None
    # Preserve the general integer-ID interface for vocabularies outside uint32.
    if not tokenizer.vocab or min(tokenizer.vocab) < 0 or max(tokenizer.vocab) >= 1 << 32:
        return None
    if tokenizer._use_native and (any(type(key) is not int for key in tokenizer.vocab)
            or min(tokenizer.vocab) != 0 or max(tokenizer.vocab) != len(tokenizer.vocab) - 1):
        return None  # Integer pooling requires dense IDs; the serial core is general.
    ranges = _partition_ranges(tokenizer, text, workers * multiplier)
    if ranges is None:
        return None
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < workers:
        raise RuntimeError(f'Full-text BPE requests {workers} logical CPUs; only {len(allowed)} allowed. '
                           'No automatic resource downgrade.')
    cpus = allowed[:workers]
    affinity(cpus, include_descendants=True)
    return _launch(tokenizer, text, ranges, cpus, workers)


@contextmanager
def _interruptible():
    """Allow SIGTERM to unwind worker cleanup without changing the caller's handler permanently."""
    installed = threading.current_thread() is threading.main_thread() and signal.getsignal(signal.SIGTERM) == signal.SIG_DFL
    def interrupted(signum, frame):
        raise KeyboardInterrupt('BPE encoding interrupted')
    if installed:
        signal.signal(signal.SIGTERM, interrupted)
    try:
        yield
    finally:
        if installed:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)


def _launch(tokenizer, text, ranges, cpus, workers):
    if not tokenizer._use_native:
        from bpe_python_runtime import _launch as python_launch
        return python_launch(tokenizer, text, ranges, cpus, workers)
    from task1_pipeline_runtime import launch_all_first
    stats = {}
    result = launch_all_first(tokenizer, text, ranges, cpus, workers, False, stats)
    tokenizer._task1_worker_metrics = stats.get('worker_metadata', [])
    return result
