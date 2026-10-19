"""Private direct worker transport. No coordinator and no calling-entry imports."""
from __future__ import annotations
from array import array
import multiprocessing as mp  # noqa: F401 -- launcher uses this module's spawn context
from multiprocessing.connection import Connection, wait
from multiprocessing import shared_memory
import os
import signal
import subprocess  # noqa: F401 -- launcher creates and owns the clean worker processes
import sys
import time
from bpe import BPE
from parallel_bpe_encoding import THREAD_VARS, affinity, rss_bytes

def stamp(enabled):
    return time.perf_counter() if enabled else 0.


def elapsed(stats, key, started, enabled):
    if enabled:
        stats[key] = stats.get(key, 0.) + time.perf_counter() - started


def environment():
    env = os.environ.copy()
    env.update({key: '1' for key in THREAD_VARS})
    env['CUDA_VISIBLE_DEVICES'] = ''
    return env


def exitcode(process):
    return process.poll() if hasattr(process, 'poll') else process.exitcode


def messages(connections, processes):
    ready = wait(connections, timeout=.2)
    if not ready and (any(exitcode(p) not in (None, 0) for p in processes)
                      or not any(exitcode(p) is None for p in processes)):
        raise RuntimeError('Transport workers exited before completing')
    for connection in ready:
        try:
            value = connection.recv()
        except EOFError as error:
            raise RuntimeError('Worker closed before completing') from error
        if value[0] == 'error':
            raise RuntimeError(f'Worker {value[1]} failed: {value[2]}')
        yield connection, value


def timed_messages(connections, processes, stats, diagnostic):
    iterator = iter(messages(connections, processes))
    while True:
        start = stamp(diagnostic)
        try:
            value = next(iterator)
        except StopIteration:
            elapsed(stats, 'wait_receive_seconds', start, diagnostic)
            return
        elapsed(stats, 'wait_receive_seconds', start, diagnostic)
        yield value


def stop_processes(processes, groups=False):
    deadline = time.monotonic() + 5
    for process in processes:
        if exitcode(process) is None:
            if groups:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                process.terminate()
    for process in processes:
        remaining = max(0., deadline - time.monotonic())
        if groups:
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        else:
            process.join(timeout=remaining)
            if process.is_alive():
                process.kill()
                process.join()


def worker_ready(index, diagnostic):
    if not diagnostic:
        return None
    return {'worker_id': index, 'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else [],
            'gpu_visibility': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'threads': {key: os.environ.get(key) for key in THREAD_VARS},
            'forbidden_imports': sorted({'wandb', 'torch', 'numpy'} & sys.modules.keys())}


def encode_piece(tokenizer, text, stats, diagnostic):
    start = stamp(diagnostic)
    ids = tokenizer._encode_serial(text)
    elapsed(stats, 'encode_seconds', start, diagnostic)
    start = stamp(diagnostic)
    values = array('I', ids)
    if values.itemsize != 4:
        raise RuntimeError('Requires 32-bit unsigned arrays')
    elapsed(stats, 'pack_seconds', start, diagnostic)
    stats['tasks'] += 1
    stats['tokens'] += len(values)
    stats['characters'] += len(text)
    return values


def worker_stats(index):
    return {'worker_id': index, 'tasks': 0, 'tokens': 0, 'characters': 0}


def pipe_worker(index, cpu, request, output):
    diagnostic, stats = request['diagnostic'], worker_stats(index)
    attached = []
    try:
        affinity([cpu] if cpu is not None else [])
        rebuilding = stamp(diagnostic)
        tokenizer = BPE(*request['state'])
        elapsed(stats, 'tokenizer_rebuild_seconds', rebuilding, diagnostic)
        shared = request.get('shared')
        if shared:
            attached = [shared_memory.SharedMemory(name=shared[key], track=False) for key in ('input', 'output')]
        ready = worker_ready(index, diagnostic)
        if diagnostic:
            ready.update(ready_clock=time.perf_counter(),
                         init_receive_seconds=request.get('_receive_seconds', 0.),
                         tokenizer_rebuild_seconds=stats['tokenizer_rebuild_seconds'])
        output.send(('ready', index, ready))
        while True:
            start = stamp(diagnostic)
            kind, part, data = output.recv()
            elapsed(stats, 'job_wait_seconds', start, diagnostic)
            if kind == 'stop':
                break
            if kind != 'job':
                raise RuntimeError('Unexpected worker command')
            if shared:
                start = stamp(diagnostic)
                lo, hi = data
                data = bytes(attached[0].buf[lo:hi]).decode('utf-8', errors='strict')
                elapsed(stats, 'input_decode_seconds', start, diagnostic)
            values = encode_piece(tokenizer, data, stats, diagnostic)
            if shared:
                start = stamp(diagnostic)
                if len(values) * 4 > shared['slot_bytes']:
                    raise RuntimeError('Shared output exceeds byte-derived slot bound')
                offset = index * shared['slot_bytes']
                attached[1].buf[offset:offset + len(values) * 4] = values.tobytes()
                result = {'slot': index, 'count': len(values)}
                elapsed(stats, 'shared_write_seconds', start, diagnostic)
            else:
                result = values
            start = stamp(diagnostic)
            output.send(('part', part, result))
            elapsed(stats, 'send_seconds', start, diagnostic)
        stats.update(peak_rss_bytes=rss_bytes(), cache_entries=len(tokenizer._cache), native_encoding=getattr(tokenizer, '_native_encode_metrics', None))
        output.send(('done', index, stats))
    except BaseException as error:
        try:
            output.send(('error', index, repr(error)))
        except (OSError, EOFError):
            pass
        raise
    finally:
        for block in attached:
            block.close()
        output.close()


def await_ready(connections, processes, stats, diagnostic):
    active = set(connections)
    ready = []
    while active:
        for connection, (kind, index, payload) in messages(list(active), processes):
            if kind != 'ready' or connection not in active:
                raise RuntimeError('Unexpected startup message')
            active.remove(connection)
            if diagnostic:
                ready.append(payload)
    if diagnostic:
        stats['worker_startup'] = ready


def drive_pipes(connections, processes, request, on_result, stats):
    diagnostic = request['diagnostic']
    tasks = iter(enumerate(request['ranges']))
    inflight = {}
    def dispatch(connection):
        start = stamp(diagnostic)
        next_task = next(tasks, None)
        if next_task is None:
            inflight[connection] = None
            connection.send(('stop', 0, None))
        else:
            index, (lo, hi) = next_task
            inflight[connection] = index
            data = (lo, hi) if request.get('shared') else request['text'][lo:hi]
            connection.send(('job', index, data))
        elapsed(stats, 'dispatch_seconds', start, diagnostic)
    for connection in connections:
        dispatch(connection)
    active, metadata, completed = list(connections), [], []
    began = stamp(diagnostic)
    while active:
        for connection, (kind, index, payload) in timed_messages(active, processes, stats, diagnostic):
            if kind == 'part' and inflight[connection] == index:
                dispatch(connection)
                on_result(index, payload)
            elif kind == 'done' and inflight[connection] is None:
                active.remove(connection)
                metadata.append(payload)
                if diagnostic:
                    completed.append(time.perf_counter() - began)
            else:
                raise RuntimeError('Unexpected pipe-worker result')
    stats['worker_metadata'] = metadata
    if diagnostic:
        stats['worker_done_spread_seconds'] = max(completed) - min(completed)


def join_workers(processes, stats, diagnostic):
    start = stamp(diagnostic)
    for process in processes:
        if hasattr(process, 'poll'):
            code = process.wait()
        else:
            process.join()
            code = process.exitcode
        if code:
            raise RuntimeError('Worker exited unsuccessfully')
    elapsed(stats, 'join_seconds', start, diagnostic)


def main():
    if len(sys.argv) != 4 or sys.argv[1] != 'worker':
        raise ValueError('Only the independent direct-worker entry is supported')
    def interrupted(signum, frame):
        raise KeyboardInterrupt('BPE worker interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    with Connection(int(sys.argv[2])) as output:
        try:
            began = time.perf_counter()
            request = output.recv()
            if request['diagnostic']:
                request['_receive_seconds'] = time.perf_counter() - began
            pipe_worker(int(sys.argv[3]), request['cpu'], request, output)
        except BaseException as error:
            try:
                output.send(('error', 0, repr(error)))
            except (OSError, EOFError):
                pass
            raise


if __name__ == '__main__':
    main()
