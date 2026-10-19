"""Direct, fresh lightweight workers for the T0--T3 transport experiment.

Only standard-library imports at module entry: worker processes never import the
caller's W&B entry. All allocations and cleanup belong to the encode call.
"""
from __future__ import annotations

from contextlib import contextmanager
from multiprocessing import shared_memory
from multiprocessing.connection import Connection
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

METHODS = ('T0', 'T1', 'T2', 'T3')
RESERVE = 64 << 20


def prepare_shared(method, text, ranges, workers, owners, stats):
    from task1_transport_runtime import Unavailable
    if sys.version_info < (3, 13):
        raise Unavailable('T2/T3 require Python 3.13+ SharedMemory(track=False)')
    # Bound output by raw input bytes, never by reference token counts.
    lengths = [len(text[a:b].encode('utf-8')) for a, b in ranges]
    slot_bytes = 4 * max(lengths)
    input_bytes = sum(lengths) if method == 'T3' else 0
    required = input_bytes + workers * slot_bytes + RESERVE
    if sys.platform.startswith('linux'):
        try:
            capacity = os.statvfs('/dev/shm')
        except OSError as error:
            raise Unavailable(f'Cannot inspect /dev/shm capacity: {error}') from error
        available = capacity.f_bavail * capacity.f_frsize
        stats.update(shm_available_bytes=available, shm_required_bytes=required)
        if available < required:
            raise Unavailable(f'/dev/shm requires {required} bytes including reserve; available {available}')
    spec = {'slot_bytes': slot_bytes}
    try:
        output = shared_memory.SharedMemory(create=True, size=workers * slot_bytes)
        owners.append(output)
        spec['output'] = output.name
        offsets = ranges
        if input_bytes:
            block = shared_memory.SharedMemory(create=True, size=input_bytes)
            owners.append(block)
            spec['input'] = block.name
            offsets, position = [], 0
            for (a, b), length in zip(ranges, lengths, strict=True):
                block.buf[position:position + length] = text[a:b].encode('utf-8')
                offsets.append((position, position + length))
                position += length
    except OSError as error:
        raise Unavailable(f'Shared-memory allocation failed: {error}') from error
    stats.update(shm_names=[b.name for b in owners], shm_input_bytes=input_bytes,
                 shm_output_capacity_bytes=workers * slot_bytes)
    return spec, offsets


def worker(fd, index):
    from bpe import BPE
    import task1_transport_runtime as transport
    from parallel_bpe_encoding import affinity, rss_bytes
    connection = Connection(fd)
    attached = {}
    stats = transport.worker_stats(index)
    try:
        header = connection.recv()
        diagnostic = header['diagnostic']
        affinity([header['cpu']] if header['cpu'] is not None else [])
        started = transport.stamp(diagnostic)
        state = pickle.loads(connection.recv_bytes())
        transport.elapsed(stats, 'state_receive_deserialize_seconds', started, diagnostic)
        started = transport.stamp(diagnostic)
        tokenizer = BPE(*state)
        transport.elapsed(stats, 'tokenizer_rebuild_seconds', started, diagnostic)
        spec = header['shared']
        for kind in ('input', 'output'):
            if kind in spec:
                attached[kind] = shared_memory.SharedMemory(name=spec[kind], track=False)
        ready = transport.worker_ready(index, diagnostic)
        if diagnostic:
            ready.update(ready_clock=time.perf_counter(),
                         tokenizer_rebuild_seconds=stats['tokenizer_rebuild_seconds'])
        connection.send(('ready', index, ready))
        while True:
            started = transport.stamp(diagnostic)
            kind, part, data = connection.recv()
            transport.elapsed(stats, 'job_wait_seconds', started, diagnostic)
            # A next job/stop acknowledges that the previous slot was copied.
            if kind == 'stop':
                break
            if kind != 'job':
                raise RuntimeError('Invalid worker command')
            if 'input' in attached:
                started = transport.stamp(diagnostic)
                a, b = data
                if not 0 <= a < b <= attached['input'].size:
                    raise RuntimeError('Invalid shared input range')
                data = bytes(attached['input'].buf[a:b]).decode('utf-8', errors='strict')
                transport.elapsed(stats, 'input_decode_seconds', started, diagnostic)
            values = transport.encode_piece(tokenizer, data, stats, diagnostic)
            if 'output' in attached:
                started = transport.stamp(diagnostic)
                size = len(values) * 4
                if size > spec['slot_bytes']:
                    raise RuntimeError('Output exceeds byte-derived shared slot bound')
                offset = index * spec['slot_bytes']
                attached['output'].buf[offset:offset + size] = values.tobytes()
                payload = {'slot': index, 'count': len(values)}
                transport.elapsed(stats, 'shared_write_seconds', started, diagnostic)
            else:
                payload = values
            started = transport.stamp(diagnostic)
            connection.send(('part', part, payload))
            transport.elapsed(stats, 'send_seconds', started, diagnostic)
        stats.update(peak_rss_bytes=rss_bytes(), cache_entries=len(tokenizer._cache))
        connection.send(('done', index, stats))
    except BaseException as error:
        try:
            connection.send(('error', index, repr(error)))
        except (EOFError, OSError):
            pass
        raise
    finally:
        for block in attached.values():
            block.close()
        connection.close()


def launch(tokenizer, text, ranges, cpus, workers, method, diagnostic=False, stats=None):
    import bpe_memory_runtime as baseline
    import task1_transport_runtime as transport
    from task1_pipeline_runtime import launch_all_first
    if method not in METHODS:
        raise ValueError('Unknown encoding transport')
    if method == 'T0':
        return launch_all_first(tokenizer, text, ranges, cpus, workers, diagnostic, stats if stats is not None else {})
    stats = {} if stats is None else stats
    owners, processes, connections = [], [], []
    collector = transport.Collector(len(ranges))
    began = transport.stamp(diagnostic)
    with baseline._interruptible():
        try:
            spec, offsets = {}, ranges
            if method in ('T2', 'T3'):
                started = transport.stamp(diagnostic)
                spec, offsets = prepare_shared(method, text, ranges, workers, owners, stats)
                transport.elapsed(stats, 'shared_prepare_seconds', started, diagnostic)
            started = transport.stamp(diagnostic)
            state = pickle.dumps((tokenizer.vocab, tokenizer.merges, tokenizer.special_tokens),
                                 protocol=pickle.HIGHEST_PROTOCOL)
            transport.elapsed(stats, 'state_serialization_seconds', started, diagnostic)
            stats['state_serializations'] = 1
            stats['state_payload_bytes'] = len(state)
            # Launch every interpreter before sending any state.
            for index in range(workers):
                parent, child = transport.mp.get_context('spawn').Pipe(duplex=True)
                connections.extend([parent, child])
                started = transport.stamp(diagnostic)
                process = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), str(child.fileno()), str(index),
                     str(Path(baseline.__file__).resolve().parent)],
                    pass_fds=(child.fileno(),), env=transport.environment(), start_new_session=True)
                processes.append(process)
                child.close()
                transport.elapsed(stats, 'process_creation_seconds', started, diagnostic)
            peers = connections[::2]
            for index, parent in enumerate(peers):
                started = transport.stamp(diagnostic)
                parent.send({'cpu': cpus[index] if cpus else None, 'diagnostic': diagnostic, 'shared': spec})
                parent.send_bytes(state)
                transport.elapsed(stats, 'initialization_send_seconds', started, diagnostic)
            transport.await_ready(peers, processes, stats, diagnostic)
            transport.elapsed(stats, 'startup_seconds', began, diagnostic)
            first, tasks = True, iter(enumerate(offsets))
            inflight = {}
            def dispatch(peer):
                started = transport.stamp(diagnostic)
                job = next(tasks, None)
                if job is None:
                    inflight[peer] = None
                    peer.send(('stop', 0, None))
                else:
                    index, (a, b) = job
                    inflight[peer] = index
                    data = (a, b) if 'input' in spec else text[a:b]
                    peer.send(('job', index, data))
                transport.elapsed(stats, 'dispatch_seconds', started, diagnostic)
            for peer in peers:
                dispatch(peer)
            active, completed = list(peers), []
            stats['worker_metadata'] = []
            while active:
                for peer, (kind, index, payload) in transport.timed_messages(active, processes, stats, diagnostic):
                    if kind == 'part' and inflight[peer] == index:
                        if diagnostic and first:
                            stats['first_result_after_launch_seconds'] = time.perf_counter() - began
                            first = False
                        if spec:
                            if payload['slot'] != peers.index(peer):
                                raise RuntimeError('Worker returned another worker slot')
                            started = transport.stamp(diagnostic)
                            payload = transport.copy_slot(owners[0], payload, spec['slot_bytes'])
                            transport.elapsed(stats, 'shared_copy_seconds', started, diagnostic)
                        started = transport.stamp(diagnostic)
                        collector.accept(index, payload)
                        transport.elapsed(stats, 'assembly_seconds', started, diagnostic)
                        # collector owns an independent array, including out-of-order results.
                        dispatch(peer)
                    elif kind == 'done' and inflight[peer] is None:
                        active.remove(peer)
                        stats['worker_metadata'].append(payload)
                        if diagnostic:
                            completed.append(time.perf_counter() - began)
                    else:
                        raise RuntimeError('Unexpected worker response')
            if diagnostic:
                stats['worker_done_spread_seconds'] = max(completed) - min(completed)
            transport.join_workers(processes, stats, diagnostic)
            return collector.finish()
        finally:
            started = transport.stamp(diagnostic)
            for connection in connections:
                connection.close()
            transport.stop_processes(processes, groups=True)
            transport.elapsed(stats, 'process_cleanup_seconds', started, diagnostic)
            started = transport.stamp(diagnostic)
            transport.release_blocks(owners)
            transport.elapsed(stats, 'shared_release_seconds', started, diagnostic)


@contextmanager
def selected_method(method, diagnostic=False):
    import bpe_memory_runtime as baseline
    original, original_split, stats, calls = baseline._launch, baseline._partition_ranges, {}, []
    def split(*args):
        import task1_transport_runtime as transport
        started = transport.stamp(diagnostic)
        result = original_split(*args)
        transport.elapsed(stats, 'split_seconds', started, diagnostic)
        return result
    def selected(tokenizer, text, ranges, cpus, workers):
        calls.append(method)
        return launch(tokenizer, text, ranges, cpus, workers, method, diagnostic, stats)
    baseline._launch = selected
    if diagnostic:
        baseline._partition_ranges = split
    try:
        yield stats, calls
    finally:
        baseline._launch = original
        baseline._partition_ranges = original_split


if __name__ == '__main__':
    # Select the verified source snapshot before importing BPE or worker helpers.
    sys.path.insert(0, sys.argv[3])
    worker(int(sys.argv[1]), int(sys.argv[2]))
