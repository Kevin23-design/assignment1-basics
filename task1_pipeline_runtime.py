"""Private Q3 caller: concurrent clean-worker startup and observed-length assembly."""
from __future__ import annotations
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
import sys
import time
import bpe_memory_runtime as baseline
import task1_transport_runtime as transport

def launch_all_first(tokenizer, text, ranges, cpus, workers, diagnostic, stats, all_first=True):
    """Identical D task/result path, but start every interpreter before sending state."""
    context = transport.mp.get_context('spawn')
    connections, processes = [], []
    from task1_parent_pipeline_runtime import make_collector
    collector = make_collector(tokenizer.vocab, len(ranges), 'Q3')
    request = {'diagnostic': diagnostic, 'ranges': ranges, 'text': text}
    began = transport.stamp(diagnostic)
    first = True
    def initialize(index, parent, init):
        if diagnostic:
            tick = time.perf_counter()
            packed = ForkingPickler.dumps(init)
            serialization_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            parent.send_bytes(packed)
            stats.setdefault('initialization_send', []).append({
                'worker_id': index, 'serialization_seconds': serialization_seconds,
                'send_seconds': time.perf_counter() - tick, 'bytes': len(packed)})
        else:
            parent.send(init)
    def collect(index, values):
        nonlocal first
        if diagnostic and first:
            stats['first_result_after_launch_seconds'] = time.perf_counter() - began
            first = False
        tick = transport.stamp(diagnostic)
        collector.accept(index, values)
        transport.elapsed(stats, 'assembly_seconds', tick, diagnostic)
    with baseline._interruptible():
        try:
            initializations = []
            for index in range(workers):
                parent, child = context.Pipe(duplex=True)
                connections.extend([parent, child])
                tick = transport.stamp(diagnostic)
                process = transport.subprocess.Popen(
                    [sys.executable, str(Path(transport.__file__).resolve()), 'worker', str(child.fileno()), str(index)],
                    pass_fds=(child.fileno(),), env=transport.environment(), start_new_session=True)
                processes.append(process)
                child.close()
                init = {'diagnostic': diagnostic, 'state': (tokenizer.vocab, tokenizer.merges, tokenizer.special_tokens),
                        'cpu': cpus[index] if cpus else None}
                initializations.append((parent, init))
                if diagnostic:
                    stats.setdefault('process_creation', []).append({
                        'worker_id': index, 'seconds': time.perf_counter() - tick,
                        'created_after_launch_seconds': time.perf_counter() - began})
                if not all_first:
                    initialize(index, parent, init)
            if all_first:
                for index, (parent, init) in enumerate(initializations):
                    initialize(index, parent, init)
            peers = connections[::2]
            transport.await_ready(peers, processes, stats, diagnostic)
            transport.elapsed(stats, 'startup_seconds', began, diagnostic)
            if diagnostic:
                for ready in stats['worker_startup']:
                    ready['ready_after_launch_seconds'] = ready['ready_clock'] - began
            transport.drive_pipes(peers, processes, request, collect, stats)
            transport.join_workers(processes, stats, diagnostic)
            result = collector.finish()
            tokenizer._task1_assembly_metrics = getattr(collector, 'metrics', {
                'method': 'M0', 'converted_ids': len(result), 'pool_entries': 0, 'extension_path': None})
            return result
        except EOFError as error:
            raise RuntimeError('Pipeline worker closed before returning complete output') from error
        finally:
            tick = transport.stamp(diagnostic)
            for connection in connections:
                connection.close()
            transport.stop_processes(processes, groups=True)
            transport.elapsed(stats, 'process_cleanup_seconds', tick, diagnostic)
