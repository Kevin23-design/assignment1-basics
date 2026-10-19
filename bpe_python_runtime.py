"""Private full-text BPE process infrastructure; no W&B, NumPy or model imports.

Independent lightweight workers avoid re-executing the caller's training entry.
The caller dispatches directly and assembles completed partitions in input order.

All startup, tokenizer reconstruction, IPC, joining and list assembly happen inside
BPE.encode(). No text or token files are created and no warm process pool is kept.
"""
from __future__ import annotations

from array import array
from contextlib import contextmanager
import multiprocessing as mp
from multiprocessing.connection import Connection, wait
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from bpe import BPE
from parallel_bpe_encoding import THREAD_VARS, affinity


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


class _Collector:
    def __init__(self, count):
        self.count, self.next = count, 0
        self.pending, self.seen, self.ids = {}, set(), []

    def accept(self, index, values):
        if index in self.seen or not 0 <= index < self.count or not isinstance(values, array) or values.typecode != 'I':
            raise RuntimeError('Duplicate, invalid or out-of-range partition')
        self.seen.add(index)
        self.pending[index] = values
        while self.next in self.pending:
            self.ids.extend(self.pending.pop(self.next))
            self.next += 1

    def finish(self):
        if self.next != self.count:
            raise RuntimeError('Incomplete BPE output')
        return self.ids


def _stop_processes(processes):
    deadline = time.monotonic() + 5
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for process in processes:
        try:
            process.wait(timeout=max(0., deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def _receive(receivers, processes):
    ready = wait(receivers, timeout=.2)
    if not ready and (any(p.poll() not in (None, 0) for p in processes)
                      or not any(p.poll() is None for p in processes)):
        raise RuntimeError('BPE workers exited before completing')
    for connection in ready:
        try:
            message = connection.recv()
        except EOFError as error:
            raise RuntimeError('BPE worker closed its pipe before completing') from error
        if message[0] == 'error':
            raise RuntimeError(f'BPE worker {message[1]} failed: {message[2]}')
        yield connection, message


def _launch(tokenizer, text, ranges, cpus, workers):
    environment = os.environ.copy()
    environment.update({key: '1' for key in THREAD_VARS})
    environment['CUDA_VISIBLE_DEVICES'] = ''
    context = mp.get_context('spawn')
    connections, processes = [], []
    collector = _Collector(len(ranges))
    with _interruptible():
        try:
            for index in range(workers):
                parent, child = context.Pipe(duplex=True)
                connections.extend([parent, child])
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                            str(child.fileno()), str(index)], pass_fds=(child.fileno(),),
                                           env=environment, start_new_session=True)
                processes.append(process)
                child.close()
                # Snapshot the live tokenizer once per worker; no files or full-text forwarding.
                parent.send({'state': (tokenizer.vocab, tokenizer.merges, tokenizer.special_tokens),
                             'cpu': cpus[index] if cpus else None})
            peers = connections[::2]
            starting = set(peers)
            while starting:
                for connection, (kind, index, payload) in _receive(list(starting), processes):
                    if kind != 'ready' or connection not in starting:
                        raise RuntimeError('Unexpected startup message')
                    starting.remove(connection)
            tasks, inflight = iter(enumerate(ranges)), {}

            def dispatch(connection):
                task = next(tasks, None)
                if task is None:
                    inflight[connection] = None
                    connection.send(('stop', 0, None))
                else:
                    index, (lo, hi) = task
                    inflight[connection] = index
                    connection.send(('job', index, text[lo:hi]))

            for connection in peers:
                dispatch(connection)
            active = list(peers)
            while active:
                for connection, (kind, index, payload) in _receive(active, processes):
                    if kind == 'part' and inflight[connection] == index:
                        collector.accept(index, payload)
                        dispatch(connection)
                    elif kind == 'done' and inflight[connection] is None:
                        active.remove(connection)
                    else:
                        raise RuntimeError('Unexpected pipe-worker result')
            for process in processes:
                if process.wait() != 0:
                    raise RuntimeError('BPE worker failed during exit')
            return collector.finish()
        except EOFError as error:
            raise RuntimeError('BPE worker closed before returning complete output') from error
        finally:
            for connection in connections:
                connection.close()
            _stop_processes(processes)


def _worker(index, request, output):
    try:
        cpu = request['cpu']
        affinity([] if cpu is None else [cpu])
        tokenizer = BPE(*request['state'])
        output.send(('ready', index, None))
        while True:
            kind, part, text = output.recv()
            if kind == 'stop':
                break
            if kind != 'job':
                raise RuntimeError('Unexpected worker command')
            packed = array('I', tokenizer._encode_serial(text))
            if packed.itemsize != 4:
                raise RuntimeError('Requires 32-bit unsigned array storage')
            output.send(('part', part, packed))
        output.send(('done', index, None))
    except BaseException as error:
        try:
            output.send(('error', index, repr(error)))
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        output.close()


def main():
    # A fresh interpreter receives its thread limits and GPU hiding before imports.
    def interrupted(signum, frame):
        raise KeyboardInterrupt('BPE worker interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    with Connection(int(sys.argv[1])) as output:
        try:
            request = output.recv()
            _worker(int(sys.argv[2]), request, output)
        except BaseException as error:
            try:
                output.send(('error', 0, repr(error)))
            except (BrokenPipeError, EOFError, OSError):
                pass
            raise


if __name__ == '__main__':
    main()
