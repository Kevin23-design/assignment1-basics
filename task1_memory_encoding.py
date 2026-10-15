"""Experiment-only in-memory parallel BPE. No production tokenizer changes."""
from __future__ import annotations

from array import array
from contextlib import contextmanager
import cProfile
import multiprocessing as mp
from multiprocessing.connection import wait
from pathlib import Path
import os
import time

from bpe import BPE
from parallel_bpe_encoding import THREAD_VARS, DiagnosticCache, affinity, restrict_environment, rss_bytes, validate_tokenizer


class InsufficientBoundaries(ValueError):
    pass


@contextmanager
def spawn_environment():
    previous = {key: os.environ.get(key) for key in (*THREAD_VARS, 'CUDA_VISIBLE_DEVICES')}
    restrict_environment()
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def split_text(text, tokenizer, count, separator):
    """Return character ranges, only after actual tokenizer special matches."""
    if separator not in tokenizer.special_tokens:
        raise ValueError('Split token must be in the tokenizer special-token configuration')
    if not text:
        raise ValueError('Cannot split empty text')
    if count < 1:
        raise ValueError('Partition count must be positive')
    if count == 1:
        return [(0, len(text))]
    targets = [len(text) * index // count for index in range(1, count)]
    cuts, index = [0], 0
    for match in tokenizer._special_re.finditer(text):
        if match.group() != separator:
            continue
        while index < len(targets) and match.start() >= targets[index]:
            if match.end() < len(text) and match.end() != cuts[-1]:
                cuts.append(match.end())
            index += 1
        if index == len(targets):
            break
    cuts.append(len(text))
    ranges = list(zip(cuts, cuts[1:]))
    if len(ranges) != count:
        raise InsufficientBoundaries(f'Requested {count} nonempty parts; only {len(ranges)} safe parts available')
    return ranges


def _worker(worker_id, cpu, prefix, jobs, output, diagnostic, profile_dir):
    restrict_environment()
    affinity([] if cpu is None else [cpu])
    profiler = cProfile.Profile() if profile_dir else None
    try:
        tokenizer = BPE()
        tokenizer.load(prefix)
        validate_tokenizer(tokenizer)
        if tokenizer._cache:
            raise RuntimeError('Worker cache must start empty')
        if diagnostic:
            tokenizer._cache = DiagnosticCache()
        output.send(('ready', worker_id, None))
        tasks = 0
        diagnostic_tasks = []
        active_started = time.perf_counter() if diagnostic else None
        while True:
            waiting_at = time.perf_counter() if diagnostic else None
            job = jobs.get()
            if job is None:
                break
            index, text = job
            encoding_at = time.perf_counter() if diagnostic else None
            if profiler:
                profiler.enable()
            ids = tokenizer._encode_serial(text)
            packing_at = time.perf_counter() if diagnostic else None
            token_count = len(ids)
            packed = array('I', ids)
            if packed.itemsize != 4:
                raise RuntimeError('Requires 32-bit unsigned array storage')
            del ids
            if profiler:
                profiler.disable()
            sending_at = time.perf_counter() if diagnostic else None
            output.send(('part', index, packed))
            if diagnostic:
                sent_at = time.perf_counter()
                diagnostic_tasks.append({'part': index, 'characters': len(text), 'tokens': token_count,
                                         'job_wait_seconds': encoding_at - waiting_at,
                                         'encode_seconds': packing_at - encoding_at,
                                         'pack_seconds': sending_at - packing_at,
                                         'send_seconds': sent_at - sending_at})
            tasks += 1
        metadata = {'worker_id': worker_id, 'tasks': tasks, 'peak_rss_bytes': rss_bytes(),
                    'cache_entries': len(tokenizer._cache), 'cpu': cpu}
        if diagnostic:
            metadata['cache'] = dict(tokenizer._cache.stats)
            metadata['task_diagnostics'] = diagnostic_tasks
            metadata['ready_to_done_seconds'] = time.perf_counter() - active_started
            for key in ('characters', 'tokens', 'job_wait_seconds', 'encode_seconds', 'pack_seconds', 'send_seconds'):
                metadata[key] = sum(task[key] for task in diagnostic_tasks)
        if profiler:
            profiler.dump_stats(str(Path(profile_dir) / f'worker-{worker_id:02d}.prof'))
        output.send(('done', worker_id, metadata))
    except BaseException as error:
        try:
            output.send(('error', worker_id, repr(error)))
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        output.close()


def _receive(receivers, processes):
    ready = wait(receivers, timeout=.2)
    if not ready:
        failed = [p.exitcode for p in processes if p.exitcode not in (None, 0)]
        if failed or not any(p.is_alive() for p in processes):
            raise RuntimeError(f'Encoding workers exited before completing their results: {failed}')
        return []
    messages = []
    for connection in ready:
        try:
            message = connection.recv()
        except EOFError as error:
            raise RuntimeError('Encoding worker closed its pipe without completing') from error
        if message[0] == 'error':
            raise RuntimeError(f'Encoding worker {message[1]} failed: {message[2]}')
        messages.append((connection, message))
    return messages


def encode_parallel_memory(text, tokenizer, prefix, workers, partitions_per_worker,
                           separator, cpus, diagnostic=False, profile_dir=None):
    """Return (list[int], metrics); all setup, IPC, joining and assembly are inside this call."""
    if not 1 <= workers <= 16 or partitions_per_worker < 1:
        raise ValueError('Workers must be 1..16 and partitions_per_worker positive')
    if cpus and len(cpus) != workers:
        raise ValueError('Exactly one logical CPU must be assigned per worker')
    split_started = time.perf_counter()
    ranges = split_text(text, tokenizer, workers * partitions_per_worker, separator)
    split_seconds = time.perf_counter() - split_started
    context = mp.get_context('spawn')
    # Unbounded dispatch avoids a cycle between full task queues and result pipes.
    jobs = context.Queue()
    processes, receivers, senders = [], [], []
    parts, metadata = {}, []
    try:
        startup_started = time.perf_counter()
        with spawn_environment():
            for index in range(workers):
                receiver, sender = context.Pipe(duplex=False)
                receivers.append(receiver)
                senders.append(sender)
                process = context.Process(target=_worker, args=(
                    index, cpus[index] if cpus else None, str(prefix), jobs, sender, diagnostic, profile_dir))
                process.start()
                processes.append(process)
                sender.close()
        ready_workers = set()
        while len(ready_workers) < workers:
            for _, (kind, index, payload) in _receive(receivers, processes):
                if kind != 'ready':
                    raise RuntimeError('Unexpected message during startup')
                ready_workers.add(index)
        startup_seconds = time.perf_counter() - startup_started
        encoding_started = time.perf_counter()
        for index, (start, end) in enumerate(ranges):
            jobs.put((index, text[start:end]))
        for _ in processes:
            jobs.put(None)
        active = list(receivers)
        receive_seconds = 0.
        completed_at = []
        while active:
            receiving_at = time.perf_counter() if diagnostic else None
            messages = _receive(active, processes)
            if diagnostic:
                receive_seconds += time.perf_counter() - receiving_at
            for connection, (kind, index, payload) in messages:
                if kind == 'part':
                    if index in parts:
                        raise RuntimeError('Duplicate output partition')
                    parts[index] = payload
                elif kind == 'done':
                    metadata.append(payload)
                    active.remove(connection)
                    if diagnostic:
                        completed_at.append(time.perf_counter() - encoding_started)
                else:
                    raise RuntimeError('Unexpected encoding worker message')
        joining_at = time.perf_counter() if diagnostic else None
        for process in processes:
            process.join()
            if process.exitcode:
                raise RuntimeError(f'Encoding worker failed: {process.exitcode}')
        encoding_seconds = time.perf_counter() - encoding_started
        if set(parts) != set(range(len(ranges))):
            raise RuntimeError('Missing output partitions')
        assembly_started = time.perf_counter()
        ids = []
        for index in range(len(ranges)):
            ids.extend(parts.pop(index))
        assembly_seconds = time.perf_counter() - assembly_started
        metrics = {'split_seconds': split_seconds, 'startup_seconds': startup_seconds,
                   'worker_encode_ipc_join_seconds': encoding_seconds, 'assembly_seconds': assembly_seconds,
                   'parts': len(ranges), 'worker_peak_rss_max': max(m['peak_rss_bytes'] for m in metadata),
                   'worker_peak_rss_sum': sum(m['peak_rss_bytes'] for m in metadata),
                   'worker_peak_rss_sum_scope': 'sum of independent process peaks, not concurrent RSS',
                   'worker_metadata': sorted(metadata, key=lambda m: m['worker_id'])}
        if diagnostic:
            stats = {key: sum(m['cache'][key] for m in metadata) for key in metadata[0]['cache']}
            stats['hit_rate'] = stats['hits'] / stats['queries'] if stats['queries'] else 0.
            metrics['cache_diagnostics'] = stats
            metrics['pipeline_diagnostics'] = {
                'parent_wait_receive_seconds': receive_seconds,
                'join_seconds': encoding_seconds - (joining_at - encoding_started),
                'worker_done_spread_seconds': max(completed_at) - min(completed_at),
                'scope': 'Diagnostic wall times overlap across processes; receive includes wait and unpickle; '
                         'send includes serialization and pipe blocking. These are not pure IPC costs.'}
        return ids, metrics
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


def decode_blocked(tokenizer, ids, block_size, diagnostics=None):
    """Concatenate raw bytes blocks, then decode UTF-8 exactly once."""
    if block_size <= 0:
        raise ValueError('Decode block size must be positive')
    lookup = tokenizer.vocab.__getitem__
    if diagnostics is not None:
        # Diagnostic-only materialization isolates lookup, block join, final join and UTF-8 conversion.
        # It deliberately changes allocations, so these timings never enter normal rankings.
        blocks = []
        lookup_seconds = join_seconds = 0.
        for start in range(0, len(ids), block_size):
            selected = ids[start:start + block_size]
            began = time.perf_counter()
            pieces = list(map(lookup, selected))
            looked_up = time.perf_counter()
            blocks.append(b''.join(pieces))
            joined = time.perf_counter()
            lookup_seconds += looked_up - began
            join_seconds += joined - looked_up
            del pieces, selected
        began = time.perf_counter()
        raw = b''.join(blocks)
        joined = time.perf_counter()
        decoded = raw.decode('utf-8', errors='replace')
        ended = time.perf_counter()
        diagnostics.update({'lookup_seconds': lookup_seconds, 'block_join_seconds': join_seconds,
                            'final_join_seconds': joined - began, 'utf8_seconds': ended - joined,
                            'scope': 'Diagnostic decomposition materializes byte references per block; '
                                     'allocation pattern differs from normal decode. Not a speed score.'})
        return decoded
    blocks = (b''.join(map(lookup, ids[start:start + block_size]))
              for start in range(0, len(ids), block_size))
    return b''.join(blocks).decode('utf-8', errors='replace')
