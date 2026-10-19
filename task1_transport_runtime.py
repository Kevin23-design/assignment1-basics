"""Experiment-only transports behind the public BPE interface; clean worker entry."""
from __future__ import annotations

from array import array
from contextlib import contextmanager
import multiprocessing as mp
from multiprocessing.connection import Connection, wait
from multiprocessing import shared_memory
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import bpe as implementation
from bpe import BPE
import bpe_memory_runtime as baseline
from parallel_bpe_encoding import THREAD_VARS, affinity, rss_bytes

CASES = ('B', 'R', 'S', 'D')
SHM_RESERVE = 64 << 20


class Unavailable(RuntimeError):
    pass


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


def release_blocks(blocks):
    for block in reversed(blocks):
        try:
            block.close()
        finally:
            try:
                block.unlink()
            except FileNotFoundError:
                pass


def prepare_shared(text, ranges, workers, owners, stats):
    if sys.version_info < (3, 13):
        raise Unavailable('S requires Python 3.13+ SharedMemory(track=False)')
    chunks = [text[start:end].encode('utf-8') for start, end in ranges]
    lengths = [len(chunk) for chunk in chunks]
    total, slot_bytes = sum(lengths), 4 * max(lengths)
    required = total + workers * slot_bytes + SHM_RESERVE
    if sys.platform.startswith('linux'):
        try:
            available = os.statvfs('/dev/shm')
        except OSError as error:
            raise Unavailable(f'Cannot inspect /dev/shm capacity: {error}') from error
        available_bytes = available.f_bavail * available.f_frsize
        stats.update(shm_available_bytes=available_bytes, shm_required_bytes=required)
        if available_bytes < required:
            raise Unavailable(f'/dev/shm needs {required} bytes including reserve; available {available_bytes}')
    try:
        owners.append(shared_memory.SharedMemory(create=True, size=total))
        owners.append(shared_memory.SharedMemory(create=True, size=workers * slot_bytes))
    except OSError as error:
        raise Unavailable(f'Shared-memory allocation failed: {error}') from error
    offsets, offset = [], 0
    for chunk in chunks:
        owners[0].buf[offset:offset + len(chunk)] = chunk
        offsets.append((offset, offset + len(chunk)))
        offset += len(chunk)
    stats.update(shm_input_bytes=total, shm_output_capacity_bytes=workers * slot_bytes,
                 shm_names=[block.name for block in owners])
    return {'input': owners[0].name, 'output': owners[1].name, 'slot_bytes': slot_bytes}, offsets


def copy_slot(block, descriptor, slot_bytes):
    slot, count = descriptor['slot'], descriptor['count']
    if slot < 0 or count < 0 or count * 4 > slot_bytes or (slot + 1) * slot_bytes > block.size:
        raise RuntimeError('Invalid shared-memory result descriptor')
    view = block.buf[slot * slot_bytes:slot * slot_bytes + count * 4]
    try:
        values = array('I')
        values.frombytes(view)
        return values
    finally:
        view.release()


class Collector:
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
            raise RuntimeError('Incomplete transport output')
        return self.ids


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


def queue_worker(index, cpu, request, jobs, output):
    diagnostic, stats = request['diagnostic'], worker_stats(index)
    try:
        affinity([cpu] if cpu is not None else [])
        tokenizer = BPE(*request['state'])
        output.send(('ready', index, worker_ready(index, diagnostic)))
        while True:
            start = stamp(diagnostic)
            job = jobs.get()
            elapsed(stats, 'job_wait_seconds', start, diagnostic)
            if job is None:
                break
            part, text = job
            values = encode_piece(tokenizer, text, stats, diagnostic)
            start = stamp(diagnostic)
            output.send(('part', part, values))
            elapsed(stats, 'send_seconds', start, diagnostic)
        stats.update(peak_rss_bytes=rss_bytes(), cache_entries=len(tokenizer._cache))
        output.send(('done', index, stats))
    except BaseException as error:
        try:
            output.send(('error', index, repr(error)))
        except (OSError, EOFError):
            pass
        raise
    finally:
        output.close()


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
        stats.update(peak_rss_bytes=rss_bytes(), cache_entries=len(tokenizer._cache))
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
                on_result(index, payload)
                dispatch(connection)
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


def coordinate(output, request):
    context = mp.get_context('spawn')
    connections, senders, processes = [], [], []
    jobs = None
    diagnostic = request['diagnostic']
    stats = {'request_receive_seconds': request.get('_receive_seconds', 0.)} if diagnostic else {}
    try:
        start = stamp(diagnostic)
        if request['case'] == 'S':
            for index in range(request['workers']):
                receiver, sender = context.Pipe(duplex=True)
                connections.append(receiver)
                senders.append(sender)
                cpu = request['cpus'][index] if request['cpus'] else None
                worker_request = {key: request[key] for key in ('diagnostic', 'state', 'shared')}
                process = context.Process(target=pipe_worker, args=(index, cpu, worker_request, sender))
                process.start()
                processes.append(process)
                sender.close()
            await_ready(connections, processes, stats, diagnostic)
            elapsed(stats, 'startup_seconds', start, diagnostic)
            def forward(index, payload):
                start = stamp(diagnostic)
                output.send(('part', index, payload))
                if output.recv() != ('ack', index, None):
                    raise RuntimeError('Missing shared-slot acknowledgement')
                elapsed(stats, 'forward_ack_seconds', start, diagnostic)
            drive_pipes(connections, processes, request, forward, stats)
        else:
            jobs = context.Queue()
            for index in range(request['workers']):
                receiver, sender = context.Pipe(duplex=False)
                connections.append(receiver)
                senders.append(sender)
                cpu = request['cpus'][index] if request['cpus'] else None
                worker_request = {key: request[key] for key in ('diagnostic', 'state')}
                process = context.Process(target=queue_worker, args=(index, cpu, worker_request, jobs, sender))
                process.start()
                processes.append(process)
                sender.close()
            await_ready(connections, processes, stats, diagnostic)
            elapsed(stats, 'startup_seconds', start, diagnostic)
            start = stamp(diagnostic)
            for index, (lo, hi) in enumerate(request['ranges']):
                jobs.put((index, request['text'][lo:hi]))
            for _ in processes:
                jobs.put(None)
            elapsed(stats, 'dispatch_seconds', start, diagnostic)
            active, parts, metadata, done_at = list(connections), {}, [], []
            began = stamp(diagnostic)
            while active:
                for connection, (kind, index, payload) in timed_messages(active, processes, stats, diagnostic):
                    if kind == 'part' and 0 <= index < len(request['ranges']) and index not in parts:
                        parts[index] = payload if request['case'] == 'B' else None
                        if request['case'] == 'R':
                            start = stamp(diagnostic)
                            output.send(('part', index, payload))
                            elapsed(stats, 'forward_seconds', start, diagnostic)
                    elif kind == 'done':
                        metadata.append(payload)
                        active.remove(connection)
                        if diagnostic:
                            done_at.append(time.perf_counter() - began)
                    else:
                        raise RuntimeError('Unexpected queue-worker result')
            if len(parts) != len(request['ranges']):
                raise RuntimeError('Missing queue-worker partitions')
            stats['worker_metadata'] = metadata
            if diagnostic:
                stats['worker_done_spread_seconds'] = max(done_at) - min(done_at)
            join_workers(processes, stats, diagnostic)
            if request['case'] == 'B':
                start = stamp(diagnostic)
                for index in range(len(parts)):
                    output.send(('part', index, parts.pop(index)))
                elapsed(stats, 'forward_seconds', start, diagnostic)
        if request['case'] == 'S':
            join_workers(processes, stats, diagnostic)
        output.send(('done', len(request['ranges']), stats))
    finally:
        stop_processes(processes)
        for connection in connections + senders:
            connection.close()
        if jobs is not None:
            jobs.cancel_join_thread()
            jobs.close()


def launch(tokenizer, text, ranges, cpus, workers, case, diagnostic, stats):
    owners, connections, processes = [], [], []
    request = {'case': case, 'diagnostic': diagnostic, 'state': (tokenizer.vocab, tokenizer.merges, tokenizer.special_tokens),
               'ranges': ranges, 'cpus': cpus, 'workers': workers, 'text': text}
    began = stamp(diagnostic)
    collector = Collector(len(ranges))
    first_result = False
    def collect(index, values):
        nonlocal first_result
        if diagnostic and not first_result:
            stats['first_result_after_launch_seconds'] = time.perf_counter() - began
            first_result = True
        start = stamp(diagnostic)
        collector.accept(index, values)
        elapsed(stats, 'assembly_seconds', start, diagnostic)
    with baseline._interruptible():
        try:
            if case == 'S':
                start = stamp(diagnostic)
                request['shared'], request['ranges'] = prepare_shared(text, ranges, workers, owners, stats)
                del request['text']
                elapsed(stats, 'shared_prepare_seconds', start, diagnostic)
            context = mp.get_context('spawn')
            start = stamp(diagnostic)
            if case == 'D':
                for index in range(workers):
                    parent, child = context.Pipe(duplex=True)
                    connections.extend([parent, child])
                    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'worker',
                                                str(child.fileno()), str(index)], pass_fds=(child.fileno(),),
                                               env=environment(), start_new_session=True)
                    processes.append(process)
                    child.close()
                    # Send the small tokenizer state once, not the entire text/range table to every worker.
                    init = {key: request[key] for key in ('diagnostic', 'state')}
                    init['cpu'] = cpus[index] if cpus else None
                    parent.send(init)
                peers = connections[::2]
                await_ready(peers, processes, stats, diagnostic)
                elapsed(stats, 'startup_seconds', start, diagnostic)
                drive_pipes(peers, processes, request, collect, stats)
                join_workers(processes, stats, diagnostic)
            else:
                parent, child = context.Pipe(duplex=True)
                connections.extend([parent, child])
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'coordinator',
                                            str(child.fileno())], pass_fds=(child.fileno(),), env=environment(),
                                           start_new_session=True)
                processes.append(process)
                child.close()
                elapsed(stats, 'coordinator_spawn_seconds', start, diagnostic)
                start = stamp(diagnostic)
                parent.send(request)
                elapsed(stats, 'request_send_seconds', start, diagnostic)
                while True:
                    start = stamp(diagnostic)
                    kind, index, payload = parent.recv()
                    elapsed(stats, 'wait_receive_seconds', start, diagnostic)
                    if kind == 'error':
                        raise RuntimeError(f'Coordinator failed: {payload}')
                    if kind == 'done':
                        if index != len(ranges):
                            raise RuntimeError('Invalid completion count')
                        stats['coordinator'] = payload
                        break
                    if kind != 'part':
                        raise RuntimeError('Unexpected coordinator message')
                    if case == 'S':
                        start = stamp(diagnostic)
                        values = copy_slot(owners[1], payload, request['shared']['slot_bytes'])
                        elapsed(stats, 'shared_copy_seconds', start, diagnostic)
                        parent.send(('ack', index, None))
                    else:
                        values = payload
                    collect(index, values)
                join_workers(processes, stats, diagnostic)
            return collector.finish()
        except EOFError as error:
            raise RuntimeError('Transport closed before returning complete output') from error
        finally:
            start = stamp(diagnostic)
            for connection in connections:
                connection.close()
            stop_processes(processes, groups=True)
            elapsed(stats, 'process_cleanup_seconds', start, diagnostic)
            start = stamp(diagnostic)
            release_blocks(owners)
            elapsed(stats, 'shared_release_seconds', start, diagnostic)


@contextmanager
def selected_backend(case, diagnostic=False, development_workers=None):
    if case not in CASES:
        raise ValueError(f'Unknown backend: {case}')
    old_launch, old_ranges, old_dispatch = baseline._launch, baseline._partition_ranges, baseline.encode_if_parallel
    old_workers, old_threshold = implementation._ENCODE_WORKERS, implementation._PARALLEL_MIN_CHARACTERS
    stats, calls = {}, []
    def ranges(*args):
        start = stamp(diagnostic)
        value = old_ranges(*args)
        elapsed(stats, 'split_seconds', start, diagnostic)
        return value
    def dispatch(tokenizer, text, workers, multiplier):
        if development_workers is None:
            return old_dispatch(tokenizer, text, workers, multiplier)
        parts = ranges(tokenizer, text, workers * multiplier)
        if parts is None:
            raise RuntimeError('Development fixture has insufficient safe partitions')
        return baseline._launch(tokenizer, text, parts, [], workers)
    def run(tokenizer, text, parts, cpus, workers):
        calls.append(case)
        if case == 'B' and not diagnostic:
            stats['baseline_unmodified'] = True
            return old_launch(tokenizer, text, parts, cpus, workers)
        return launch(tokenizer, text, parts, cpus, workers, case, diagnostic, stats)
    try:
        baseline._launch, baseline._partition_ranges, baseline.encode_if_parallel = run, ranges, dispatch
        if development_workers is not None:
            implementation._ENCODE_WORKERS, implementation._PARALLEL_MIN_CHARACTERS = development_workers, 1
        yield stats, calls
    finally:
        baseline._launch, baseline._partition_ranges, baseline.encode_if_parallel = old_launch, old_ranges, old_dispatch
        implementation._ENCODE_WORKERS, implementation._PARALLEL_MIN_CHARACTERS = old_workers, old_threshold


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Transport interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    with Connection(int(sys.argv[2])) as output:
        try:
            receive_started = time.perf_counter()
            request = output.recv()
            if request['diagnostic']:
                request['_receive_seconds'] = time.perf_counter() - receive_started
            if sys.argv[1] == 'worker':
                pipe_worker(int(sys.argv[3]), request['cpu'], request, output)
            else:
                affinity(request['cpus'])
                coordinate(output, request)
        except BaseException as error:
            try:
                output.send(('error', 0, repr(error)))
            except (OSError, EOFError):
                pass
            raise


if __name__ == '__main__':
    main()
