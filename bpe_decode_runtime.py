"""Private bytes-only decoding infrastructure. Worker entry imports only the standard library."""
from __future__ import annotations

from array import array
from contextlib import contextmanager
import multiprocessing as mp
from multiprocessing.connection import Connection, wait
from operator import itemgetter
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import threading
import time

METHODS = ('R0', 'N128', 'N1M', 'P4', 'P8', 'P16')
BLOCK = 131072
TASK = 1048576
THREAD_VARS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')


def tick(stats):
    return time.perf_counter() if stats is not None else 0.


def elapsed(stats, name, start):
    if stats is not None:
        stats[name] = stats.get(name, 0.) + time.perf_counter() - start


def raw_chunks(vocab, ids):
    for start in range(0, len(ids), BLOCK):
        block = ids[start:start + BLOCK]
        values = itemgetter(*block)(vocab)
        yield b''.join((values,) if len(block) == 1 else values)


def serial(vocab, ids):
    return b''.join(raw_chunks(vocab, ids)).decode('utf-8', errors='replace')


def dense(vocab):
    return bool(vocab) and min(vocab) == 0 and max(vocab) == len(vocab) - 1


def numpy_decode(vocab, ids, block_size, stats=None):
    if not dense(vocab):
        return serial(vocab, ids)
    start = tick(stats)
    import numpy as np
    elapsed(stats, 'numpy_import_seconds', start)
    start = tick(stats)
    table = np.empty(len(vocab), dtype=object)
    table[:] = [vocab[i] for i in range(len(vocab))]
    elapsed(stats, 'table_seconds', start)
    blocks = []
    for start_index in range(0, len(ids), block_size):
        start = tick(stats)
        try:
            indices = np.asarray(ids[start_index:start_index + block_size])
        except (OverflowError, ValueError, TypeError):
            return serial(vocab, ids)
        # Do not truncate floats, wrap negative IDs or overflow unsigned IDs into intp.
        if (indices.dtype.kind not in ('i', 'u') or indices.min() < 0 or indices.max() >= len(vocab)):
            return serial(vocab, ids)
        indices = indices.astype(np.intp, copy=False)
        elapsed(stats, 'index_seconds', start)
        start = tick(stats)
        values = np.take(table, indices, mode='raise').tolist()
        elapsed(stats, 'lookup_seconds', start)
        start = tick(stats)
        blocks.append(b''.join(values))
        elapsed(stats, 'bytes_join_seconds', start)
    start = tick(stats)
    raw = b''.join(blocks)
    elapsed(stats, 'bytes_join_seconds', start)
    start = tick(stats)
    result = raw.decode('utf-8', errors='replace')
    elapsed(stats, 'utf8_seconds', start)
    return result


@contextmanager
def interruptible():
    installed = threading.current_thread() is threading.main_thread() and signal.getsignal(signal.SIGTERM) == signal.SIG_DFL
    def interrupted(signum, frame):
        raise KeyboardInterrupt('BPE decode interrupted')
    if installed:
        signal.signal(signal.SIGTERM, interrupted)
    try:
        yield
    finally:
        if installed:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)


def stop(processes):
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


def receive(connections, processes):
    ready = wait(connections, timeout=.2)
    if not ready and (any(p.poll() not in (None, 0) for p in processes) or all(p.poll() is not None for p in processes)):
        raise RuntimeError('Decode workers exited before completion')
    for connection in ready:
        try:
            message = connection.recv()
        except EOFError as error:
            raise RuntimeError('Decode worker closed before completion') from error
        if message[0] == 'error':
            raise RuntimeError(f'Decode worker {message[1]} failed: {message[2]}')
        yield connection, message


class InvalidIDs(Exception):
    """Return to the serial path to preserve its input-error semantics after worker cleanup."""


def parallel_decode(vocab, ids, workers, cpus, stats=None):
    connections, processes = [], []
    env = {**os.environ, **dict.fromkeys(THREAD_VARS, '1'), 'CUDA_VISIBLE_DEVICES': ''}
    began = tick(stats)
    with interruptible():
        try:
            for index in range(workers):
                parent, child = mp.get_context('spawn').Pipe(duplex=True)
                connections.extend((parent, child))
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker',
                                            str(child.fileno()), str(index)], pass_fds=(child.fileno(),),
                                           env=env, start_new_session=True)
                processes.append(process)
                child.close()
            peers = connections[::2]
            for index, parent in enumerate(peers):
                parent.send((vocab, cpus[index] if cpus else None))
            pending = set(peers)
            startup = []
            while pending:
                for connection, (kind, index, payload) in receive(list(pending), processes):
                    if kind != 'ready' or connection not in pending:
                        raise RuntimeError('Invalid decoder startup')
                    pending.remove(connection)
                    startup.append(payload)
            elapsed(stats, 'startup_seconds', began)
            if stats is not None:
                stats['worker_startup'] = startup
            tasks = iter(enumerate(range(0, len(ids), TASK)))
            inflight, parts, seen, blocks = {}, {}, set(), []
            next_part = 0
            def dispatch(connection):
                task = next(tasks, None)
                if task is None:
                    inflight[connection] = None
                    connection.send(('stop', 0, None))
                    return
                index, offset = task
                start = tick(stats)
                values = ids[offset:offset + TASK]
                try:
                    if min(values) < 0 or max(values) >= len(vocab):
                        raise InvalidIDs()
                    packed = array('I', values)
                    if packed.itemsize != 4:
                        raise RuntimeError('Requires uint32 array storage')
                except (TypeError, OverflowError):
                    raise InvalidIDs() from None
                elapsed(stats, 'pack_seconds', start)
                inflight[connection] = index
                start = tick(stats)
                connection.send(('job', index, packed))
                elapsed(stats, 'send_seconds', start)
            for connection in peers:
                dispatch(connection)
            active, metadata = list(peers), []
            while active:
                iterator = iter(receive(active, processes))
                while True:
                    start = tick(stats)
                    try:
                        connection, (kind, index, payload) = next(iterator)
                    except StopIteration:
                        elapsed(stats, 'receive_wait_seconds', start)
                        break
                    elapsed(stats, 'receive_wait_seconds', start)
                    if kind == 'part' and inflight[connection] == index and index not in seen and isinstance(payload, bytes):
                        start = tick(stats)
                        seen.add(index)
                        parts[index] = payload
                        while next_part in parts:
                            blocks.append(parts.pop(next_part))
                            next_part += 1
                        elapsed(stats, 'assembly_seconds', start)
                        dispatch(connection)
                    elif kind == 'done' and inflight[connection] is None:
                        metadata.append(payload)
                        active.remove(connection)
                    else:
                        raise RuntimeError('Invalid decoder result')
            if next_part != (len(ids) + TASK - 1) // TASK:
                raise RuntimeError('Incomplete decoder result')
            start = tick(stats)
            for process in processes:
                if process.wait():
                    raise RuntimeError('Decoder exited unsuccessfully')
            elapsed(stats, 'join_seconds', start)
            if stats is not None:
                stats['worker_metadata'] = metadata
            start = tick(stats)
            raw = b''.join(blocks)
            elapsed(stats, 'bytes_join_seconds', start)
            start = tick(stats)
            decoded = raw.decode('utf-8', errors='replace')
            elapsed(stats, 'utf8_seconds', start)
            return decoded
        finally:
            start = tick(stats)
            for connection in connections:
                connection.close()
            stop(processes)
            elapsed(stats, 'cleanup_seconds', start)


def decode_ids(vocab, ids, method, stats=None, max_workers=16, development=False):
    if method not in METHODS:
        raise ValueError('Unknown decoder')
    threshold = 1048576 if method.startswith('P') else BLOCK
    if not ids or method == 'R0' or (not development and len(ids) < threshold) or not dense(vocab):
        return serial(vocab, ids)
    if method.startswith('N'):
        return numpy_decode(vocab, ids, BLOCK if method == 'N128' else TASK, stats)
    workers = min(int(method[1:]), max_workers)
    if workers < 1 or max_workers > 16:
        raise ValueError('Decoder worker allocation must be 1..16')
    if hasattr(os, 'sched_getaffinity'):
        allowed = sorted(os.sched_getaffinity(0))
        if len(allowed) < workers:
            raise RuntimeError('Insufficient allowed CPUs for decoder; no downgrade')
        cpus = allowed[:workers]
    elif development:
        cpus = []
    else:
        return serial(vocab, ids)
    if stats is not None:
        stats.update(requested_workers=int(method[1:]), effective_workers=workers)
    try:
        return parallel_decode(vocab, ids, workers, cpus, stats)
    except InvalidIDs:
        return serial(vocab, ids)


def worker(fd, index):
    # This leaf owns no child processes, files or shared-memory blocks. The controller
    # owns cleanup, so default termination also avoids Python-shutdown signal races.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    with Connection(fd) as output:
        try:
            vocab, cpu = output.recv()
            if cpu is not None:
                os.sched_setaffinity(0, {cpu})
            output.send(('ready', index, {
                'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else [],
                'gpu_visibility': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'threads': {key: os.environ.get(key) for key in THREAD_VARS},
                'forbidden_imports': sorted({'numpy', 'wandb', 'torch', 'bpe'} & sys.modules.keys())}))
            tasks, tokens, encode_seconds = 0, 0, 0.
            while True:
                kind, part, values = output.recv()
                if kind == 'stop':
                    break
                if kind != 'job' or not isinstance(values, array) or values.typecode != 'I':
                    raise RuntimeError('Invalid decoder job')
                start = time.perf_counter()
                raw = b''.join(raw_chunks(vocab, values))
                encode_seconds += time.perf_counter() - start
                tasks += 1
                tokens += len(values)
                output.send(('part', part, raw))
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            output.send(('done', index, {'tasks': tasks, 'tokens': tokens, 'lookup_join_seconds': encode_seconds,
                                        'peak_rss_bytes': int(peak if sys.platform == 'darwin' else peak * 1024)}))
        except (EOFError, BrokenPipeError, KeyboardInterrupt):
            return  # Controller cancelled; it still checks completion and reaps this process.
        except BaseException as error:
            try:
                output.send(('error', index, repr(error)))
            except (OSError, EOFError):
                pass
            raise


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] != '--worker':
        raise SystemExit('Private decoder worker entry')
    worker(int(sys.argv[2]), int(sys.argv[3]))
