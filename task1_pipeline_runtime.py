"""Experiment-only startup/decode variants; no tokenizer algorithm or training changes."""
from __future__ import annotations

from contextlib import contextmanager
from multiprocessing.reduction import ForkingPickler
from operator import itemgetter
from pathlib import Path
import sys
import time

from bpe import BPE
import bpe_memory_runtime as baseline
import task1_transport_runtime as transport

DECODERS = ('C0', 'C1', 'C2', 'C3')
CASES = ('D0', 'D1', 'D2', 'D3')
BLOCK_SIZE = 131072
_ORIGINAL_DECODE = BPE.decode


def decode(tokenizer, ids, method, diagnostic=None, block_size=BLOCK_SIZE):
    if method not in DECODERS or block_size < 1:
        raise ValueError('Invalid decoder or block size')
    if method == 'C0' and diagnostic is None:
        return _ORIGINAL_DECODE(tokenizer, ids)
    began = time.perf_counter() if diagnostic is not None else 0.
    vocab, table = tokenizer.vocab, tokenizer.vocab
    if method in ('C1', 'C3') and vocab and min(vocab) == 0 and max(vocab) == len(vocab) - 1:
        table = tuple(vocab[index] for index in range(len(vocab)))
    if diagnostic is not None:
        diagnostic.update(table_build_seconds=time.perf_counter() - began, lookup_seconds=0.,
                          bytes_join_seconds=0., bounds_seconds=0., index_prepare_seconds=0.,
                          note='Diagnostic materializes lookup results to separate lookup/join; not a speed score')
    blocks = []
    for start in range(0, len(ids), block_size):
        tick = time.perf_counter() if diagnostic is not None else 0.
        block = ids[start:start + block_size]
        if diagnostic is not None:
            diagnostic['index_prepare_seconds'] += time.perf_counter() - tick
        tick = time.perf_counter() if diagnostic is not None else 0.
        chosen = table
        if table is not vocab:
            try:
                if min(block) < 0 or max(block) >= len(table):
                    chosen = vocab  # Preserve dictionary KeyError rather than negative indexing.
            except TypeError:
                chosen = vocab
        if diagnostic is not None:
            diagnostic['bounds_seconds'] += time.perf_counter() - tick
        tick = time.perf_counter() if diagnostic is not None else 0.
        try:
            if method in ('C2', 'C3'):
                values = itemgetter(*block)(chosen)
                if len(block) == 1:
                    values = (values,)
            else:
                values = map(chosen.__getitem__, block)
            if diagnostic is not None:
                values = tuple(values)
                diagnostic['lookup_seconds'] += time.perf_counter() - tick
                tick = time.perf_counter()
            raw = b''.join(values)
        except (IndexError, TypeError):
            if chosen is vocab:
                raise
            raw = b''.join(map(vocab.__getitem__, block))
        if diagnostic is not None:
            diagnostic['bytes_join_seconds'] += time.perf_counter() - tick
        blocks.append(raw)
    tick = time.perf_counter() if diagnostic is not None else 0.
    raw = b''.join(blocks)
    if diagnostic is not None:
        diagnostic['bytes_join_seconds'] += time.perf_counter() - tick
        tick = time.perf_counter()
    result = raw.decode('utf-8', errors='replace')
    if diagnostic is not None:
        diagnostic['utf8_seconds'] = time.perf_counter() - tick
    return result


@contextmanager
def selected_decoder(method, diagnostic=None):
    original = BPE.decode
    try:
        if method != 'C0' or diagnostic is not None:
            BPE.decode = lambda tokenizer, ids: decode(tokenizer, ids, method, diagnostic)
        yield
    finally:
        BPE.decode = original


def launch_all_first(tokenizer, text, ranges, cpus, workers, diagnostic, stats, all_first=True):
    """Identical D task/result path, but start every interpreter before sending state."""
    context = transport.mp.get_context('spawn')
    connections, processes = [], []
    collector = transport.Collector(len(ranges))
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
            return collector.finish()
        except EOFError as error:
            raise RuntimeError('Pipeline worker closed before returning complete output') from error
        finally:
            tick = transport.stamp(diagnostic)
            for connection in connections:
                connection.close()
            transport.stop_processes(processes, groups=True)
            transport.elapsed(stats, 'process_cleanup_seconds', tick, diagnostic)


@contextmanager
def selected_pipeline(case, decoder, diagnostic=False, development_workers=None):
    if case not in CASES or decoder not in DECODERS:
        raise ValueError('Unknown pipeline configuration')
    # Keep old D's normal path unchanged; reuse its safe splitting/public dispatch and worker core.
    with transport.selected_backend('D', diagnostic, development_workers) as (stats, calls):
        original = baseline._launch
        try:
            if case in ('D1', 'D3') or diagnostic:
                def launch(tokenizer, text, ranges, cpus, workers):
                    calls.append('D')
                    return launch_all_first(tokenizer, text, ranges, cpus, workers, diagnostic, stats,
                                            all_first=case in ('D1', 'D3'))
                baseline._launch = launch
            decoder_metrics = stats.setdefault('decoder', {}) if diagnostic else None
            with selected_decoder(decoder, decoder_metrics):
                yield stats, calls
        finally:
            baseline._launch = original
