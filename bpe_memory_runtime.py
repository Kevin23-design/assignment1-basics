"""Private full-text BPE process infrastructure; no W&B, NumPy or model imports.

A clean coordinator prevents spawn from re-executing the caller's training entry.
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


def _launch(tokenizer, text, ranges, cpus, workers):
    environment = os.environ.copy()
    environment.update({key: '1' for key in THREAD_VARS})
    environment['CUDA_VISIBLE_DEVICES'] = ''
    context = mp.get_context('spawn')
    parent, child = context.Pipe(duplex=True)
    process = None
    with _interruptible():
        try:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(child.fileno())],
                                       pass_fds=(child.fileno(),), env=environment, start_new_session=True)
            child.close()
            # Snapshot the live tokenizer, including from_files()/constructor instances.
            # Workers never reopen possibly changed or deleted serialization files.
            parent.send((text, tokenizer.vocab, tokenizer.merges, tokenizer.special_tokens, ranges, cpus, workers))
            ids, expected = [], 0
            while True:
                kind, index, payload = parent.recv()
                if kind == 'error':
                    raise RuntimeError(f'BPE coordinator failed: {payload}')
                if kind == 'done':
                    if index != len(ranges) or expected != len(ranges):
                        raise RuntimeError('Incomplete BPE output')
                    break
                if kind != 'part' or index != expected or not isinstance(payload, array) or payload.typecode != 'I':
                    raise RuntimeError('Invalid or unordered BPE output')
                ids.extend(payload)
                expected += 1
            if process.wait() != 0:
                raise RuntimeError('BPE coordinator exited unsuccessfully')
            return ids
        except EOFError as error:
            raise RuntimeError('BPE coordinator closed before returning the complete encoding') from error
        finally:
            parent.close()
            child.close()
            if process is not None:
                # Also covers partial startup, a user interrupt and a failed result transfer.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def _worker(index, cpu, state, jobs, output):
    # Only the clean coordinator starts these workers; their entry imports no logging/model code.
    try:
        affinity([] if cpu is None else [cpu])
        tokenizer = BPE(*state)
        output.send(('ready', index, None))
        while True:
            job = jobs.get()
            if job is None:
                break
            part, text = job
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


def _receive(receivers, processes):
    ready = wait(receivers, timeout=.2)
    if not ready and (any(p.exitcode not in (None, 0) for p in processes)
                      or not any(p.is_alive() for p in processes)):
        raise RuntimeError('BPE workers exited before completing')
    for connection in ready:
        try:
            message = connection.recv()
        except EOFError as error:
            raise RuntimeError('BPE worker closed its pipe before completing') from error
        if message[0] == 'error':
            raise RuntimeError(f'BPE worker {message[1]} failed: {message[2]}')
        yield connection, message


def _coordinate(output, text, vocab, merges, specials, ranges, cpus, workers):
    context = mp.get_context('spawn')
    jobs = context.Queue()  # Unbounded dispatch avoids a jobs/results backpressure cycle.
    processes, receivers, senders = [], [], []
    parts = {}
    try:
        for index in range(workers):
            receiver, sender = context.Pipe(duplex=False)
            receivers.append(receiver)
            senders.append(sender)
            process = context.Process(target=_worker, args=(index, cpus[index] if cpus else None,
                                                           (vocab, merges, specials), jobs, sender))
            process.start()
            processes.append(process)
            sender.close()
        ready = set()
        while len(ready) < workers:
            for _, (kind, index, payload) in _receive(receivers, processes):
                if kind != 'ready' or index in ready:
                    raise RuntimeError('Unexpected startup message')
                ready.add(index)
        for index, (start, end) in enumerate(ranges):
            jobs.put((index, text[start:end]))
        for _ in processes:
            jobs.put(None)
        active = list(receivers)
        while active:
            for connection, (kind, index, payload) in _receive(active, processes):
                if kind == 'part' and 0 <= index < len(ranges) and index not in parts:
                    parts[index] = payload
                elif kind == 'done':
                    active.remove(connection)
                else:
                    raise RuntimeError('Unexpected worker result')
        for process in processes:
            process.join()
            if process.exitcode:
                raise RuntimeError('BPE worker failed during exit')
        if len(parts) != len(ranges):
            raise RuntimeError('Missing BPE partitions')
        for index in range(len(ranges)):
            output.send(('part', index, parts.pop(index)))
        output.send(('done', len(ranges), None))
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()
        for connection in receivers + senders:
            connection.close()
        jobs.cancel_join_thread()
        jobs.close()


def main():
    # The coordinator is a fresh interpreter with thread limits/GPU hiding set before any import.
    def interrupted(signum, frame):
        raise KeyboardInterrupt('BPE coordinator interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    with Connection(int(sys.argv[1])) as output:
        try:
            request = output.recv()
            cpus = request[-2]
            affinity(cpus)
            _coordinate(output, *request)
        except BaseException as error:
            try:
                output.send(('error', 0, repr(error)))
            except (BrokenPipeError, EOFError, OSError):
                pass
            raise


if __name__ == '__main__':
    main()
