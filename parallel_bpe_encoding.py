"""Streaming BPE process infrastructure; the default BPE interface stays unchanged."""
from __future__ import annotations

from array import array
import io
from itertools import islice
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import resource
import shutil
import sys
import time

import regex
from bpe import BPE

THREAD_VARS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')
TOKEN_BLOCK = 262144
TEXT_BLOCK = 1 << 20


def check_deadline(deadline):
    if time.perf_counter() >= deadline:
        raise TimeoutError('BPE encoding reached the formal run deadline')


def restrict_environment():
    for name in THREAD_VARS:
        os.environ[name] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''


def affinity(cpus, include_descendants=False):
    if cpus:
        # Linux affinity is per thread. Also bind logging-service descendants
        # when called by the top-level orchestrator after W&B has started.
        pids = {os.getpid()}
        if include_descendants:
            parents = {}
            for entry in Path('/proc').iterdir():
                if not entry.name.isdigit():
                    continue
                try:
                    status = (entry / 'status').read_text()
                    parents[int(entry.name)] = int(next(line.split()[1] for line in status.splitlines()
                                                       if line.startswith('PPid:')))
                except (FileNotFoundError, ProcessLookupError, PermissionError, StopIteration):
                    continue
            while True:
                children = {pid for pid, parent in parents.items() if parent in pids}
                if children <= pids:
                    break
                pids.update(children)
        for pid in pids:
            try:
                tids = [int(entry.name) for entry in (Path('/proc') / str(pid) / 'task').iterdir()]
            except FileNotFoundError:
                continue
            for tid in tids:
                try:
                    os.sched_setaffinity(tid, set(cpus))
                except ProcessLookupError:
                    continue
        if os.sched_getaffinity(0) != set(cpus):
            raise RuntimeError('CPU affinity was not enforced')


def rss_bytes():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == 'darwin' else rss * 1024


class DiagnosticCache(dict):
    def __init__(self):
        super().__init__()
        self.stats = dict.fromkeys(('queries', 'hits', 'misses', 'insertions', 'clears'), 0)

    def get(self, key, default=None):
        value = super().get(key, default)
        self.stats['queries'] += 1
        self.stats['hits' if value is not None else 'misses'] += 1
        return value

    def clear(self):
        self.stats['clears'] += 1
        super().clear()

    def __setitem__(self, key, value):
        self.stats['insertions'] += 1
        super().__setitem__(key, value)


class StreamingExperimentBPE(BPE):
    def configure(self, capacity, diagnostic=False):
        if capacity <= 0:
            raise ValueError('Cache capacity must be positive')
        self.capacity = capacity
        self._cache = DiagnosticCache() if diagnostic else {}

    def _encode_piece(self, piece):
        cached = self._cache.get(piece)
        if cached is not None:
            return cached
        return BPE._encode_uncached(self, piece, self.capacity)

    def encode_iterable(self, iterable):
        # Same production streaming boundaries, with an experiment-only cache limit.
        for special, piece in self._stream_pieces(iterable):
            if special:
                yield self._special_ids[piece]
            else:
                yield from self._encode_piece(piece)


def validate_tokenizer(bpe):
    if not 256 <= len(bpe.vocab) <= 32000:
        raise ValueError('Vocabulary must have between 256 and 32,000 entries')
    if set(bpe.vocab) != set(range(len(bpe.vocab))):
        raise ValueError('Vocabulary IDs must be contiguous from zero')
    if any(bytes([i]) not in bpe._ids for i in range(256)):
        raise ValueError('Vocabulary must contain all 256 bytes')


def special_matches(source, specials, block_size=TEXT_BLOCK, deadline=math.inf):
    """Match the exact escaped-special alternation with bounded binary lookahead.

    Defer starts within max_special_bytes-1 bytes of a read boundary so that a
    longer special can still win. Consume accepted matches non-overlappingly,
    just like production regex.finditer over the entire text.
    """
    ordered = sorted(dict.fromkeys(specials), key=len, reverse=True)
    if not ordered:
        return
    pattern = regex.compile(b'|'.join(regex.escape(s.encode('utf-8')) for s in ordered))
    longest = max(len(s.encode('utf-8')) for s in ordered)
    buffer, offset = b'', 0
    with open(source, 'rb') as handle:
        while True:
            check_deadline(deadline)
            chunk = handle.read(block_size)
            buffer += chunk
            eof = not chunk
            safe_start = len(buffer) if eof else len(buffer) - longest
            consumed = 0
            for match in pattern.finditer(buffer):
                if match.start() > safe_start:
                    break
                yield offset + match.start(), offset + match.end(), match.group()
                consumed = match.end()
            discard = len(buffer) if eof else max(consumed, max(0, safe_start + 1))
            buffer = buffer[discard:]
            offset += discard
            if eof:
                break


def split_ranges(source, bpe, workers, separator, block_size=TEXT_BLOCK, deadline=math.inf):
    check_deadline(deadline)
    if separator not in bpe.special_tokens:
        raise ValueError('Split token must be in the loaded special-token configuration')
    size = Path(source).stat().st_size
    if size == 0:
        raise ValueError('Input is empty')
    if workers == 1:
        return [(0, size)]
    targets = [size * i // workers for i in range(1, workers)]
    cuts, index = [0], 0
    raw_separator = separator.encode('utf-8')
    for start, end, raw in special_matches(source, bpe.special_tokens, block_size, deadline):
        if raw != raw_separator:
            continue
        while index < len(targets) and start >= targets[index]:
            if end < size and end != cuts[-1]:
                cuts.append(end)
            index += 1
        if index == len(targets):
            break
    cuts.append(size)
    return list(zip(cuts, cuts[1:]))


class RangeReader(io.RawIOBase):
    def __init__(self, source, start, end):
        super().__init__()
        self.handle = open(source, 'rb')
        self.handle.seek(start)
        self.remaining = end - start

    def readable(self):
        return True

    def readinto(self, buffer):
        count = self.handle.readinto(memoryview(buffer)[:min(len(buffer), self.remaining)])
        self.remaining -= count
        return count

    def close(self):
        self.handle.close()
        super().close()


def text_chunks(source, start, end, block_size=TEXT_BLOCK, deadline=math.inf):
    with io.TextIOWrapper(io.BufferedReader(RangeReader(source, start, end)),
                          encoding='utf-8', errors='strict', newline='') as handle:
        while text := handle.read(block_size):
            check_deadline(deadline)
            yield text


def write_uint32(handle, ids):
    values = array('I', ids)
    if values.itemsize != 4:
        raise RuntimeError('Requires 32-bit unsigned array storage')
    if sys.byteorder != 'little':
        values.byteswap()
    values.tofile(handle)


def read_uint32(handle):
    while raw := handle.read(TOKEN_BLOCK * 4):
        if len(raw) % 4:
            raise ValueError('Malformed uint32 token file')
        values = array('I')
        values.frombytes(raw)
        if sys.byteorder != 'little':
            values.byteswap()
        yield values


def worker_entry(spec):
    restrict_environment()
    affinity(spec['cpus'])
    part = Path(spec['part'])
    partial = Path(str(part) + '.partial')
    metadata = Path(spec['metadata'])
    try:
        deadline = spec.get('deadline', math.inf)
        check_deadline(deadline)
        bpe = StreamingExperimentBPE()
        bpe.load(spec['tokenizer_prefix'])
        validate_tokenizer(bpe)
        bpe.configure(spec['capacity'], spec['diagnostic'])
        count = 0
        pieces = iter(bpe.encode_iterable(text_chunks(spec['source'], spec['start'], spec['end'], deadline=deadline)))
        with partial.open('wb') as handle:
            while block := list(islice(pieces, TOKEN_BLOCK)):
                check_deadline(deadline)
                write_uint32(handle, block)
                count += len(block)
        partial.replace(part)
        result = {'token_count': count, 'peak_rss_bytes': rss_bytes(), 'cache_entries': len(bpe._cache),
                  'affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None}
        if spec['diagnostic']:
            result['diagnostics'] = bpe._cache.stats
        metadata.write_text(json.dumps(result), encoding='utf-8')
    except BaseException as error:
        metadata.write_text(json.dumps({'error': repr(error)}), encoding='utf-8')
        partial.unlink(missing_ok=True)
        raise


def encode_parallel(source, tokenizer_prefix, destination, workers, capacity,
                    separator, cpus, diagnostic=False, deadline=math.inf):
    """Coordinator: no W&B, no comparison; all file creation is timed by caller."""
    if not 1 <= workers <= 16 or capacity <= 0:
        raise ValueError('Workers must be 1..16 and cache capacity positive')
    if cpus and len(cpus) != workers:
        raise ValueError('Exactly one CPU must be assigned per worker')
    restrict_environment()
    affinity(cpus)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    work = destination.parent / (destination.name + '.parts')
    work.mkdir(exist_ok=False)
    partial = Path(str(destination) + '.partial')
    processes = []
    try:
        check_deadline(deadline)
        started = time.perf_counter()
        bpe = BPE()
        bpe.load(tokenizer_prefix)
        validate_tokenizer(bpe)
        ranges = split_ranges(source, bpe, workers, separator, deadline=deadline)
        split_seconds = time.perf_counter() - started
        if len(ranges) != workers:
            return {'status': 'unavailable', 'reason': 'Insufficient distinct special-token cut points',
                    'requested_workers': workers, 'actual_parts': len(ranges),
                    'split_seconds': split_seconds}
        del bpe
        specs = []
        encoding_started = time.perf_counter()
        context = mp.get_context('spawn')
        for i, (start, end) in enumerate(ranges):
            spec = {'source': str(source), 'tokenizer_prefix': str(tokenizer_prefix),
                    'start': start, 'end': end, 'cpus': cpus[i:i + 1],
                    'capacity': capacity, 'diagnostic': diagnostic,
                    'deadline': deadline,
                    'part': str(work / f'part-{i:03d}.tokens'),
                    'metadata': str(work / f'part-{i:03d}.json')}
            specs.append(spec)
            process = context.Process(target=worker_entry, args=(spec,))
            process.start()
            processes.append(process)
        while any(p.is_alive() for p in processes):
            check_deadline(deadline)
            if any(p.exitcode not in (None, 0) for p in processes):
                raise RuntimeError('Encoding worker failed; see worker traceback in coordinator log')
            for process in processes:
                process.join(timeout=0.05)
        for process in processes:
            process.join()
            if process.exitcode:
                raise RuntimeError('Encoding worker failed; see worker traceback in coordinator log')
        encode_seconds = time.perf_counter() - encoding_started
        metadata = [json.loads(Path(s['metadata']).read_text()) for s in specs]
        concat_started = time.perf_counter()
        check_deadline(deadline)
        if workers == 1:
            Path(specs[0]['part']).replace(partial)
        else:
            with partial.open('wb') as output:
                for spec in specs:
                    with open(spec['part'], 'rb') as part:
                        while block := part.read(TEXT_BLOCK):
                            check_deadline(deadline)
                            output.write(block)
                    Path(spec['part']).unlink()
        check_deadline(deadline)
        partial.replace(destination)
        concatenate_seconds = time.perf_counter() - concat_started
        result = {'status': 'ok', 'workers': workers, 'cache_capacity': capacity,
                  'split_seconds': split_seconds, 'encode_seconds': encode_seconds,
                  'concatenate_seconds': concatenate_seconds,
                  'input_bytes': Path(source).stat().st_size,
                  'token_count': sum(m['token_count'] for m in metadata),
                  'parent_peak_rss': rss_bytes(),
                  'worker_peak_rss_max': max(m['peak_rss_bytes'] for m in metadata),
                  'worker_peak_rss_sum': sum(m['peak_rss_bytes'] for m in metadata),
                  'worker_peak_rss_sum_scope': 'sum of separate process peaks, not concurrent RSS',
                  'cpu_affinity': cpus, 'worker_affinities': [m['affinity'] for m in metadata],
                  'ranges': ranges}
        if diagnostic:
            stats = {key: sum(m['diagnostics'][key] for m in metadata) for key in metadata[0]['diagnostics']}
            stats['hit_rate'] = stats['hits'] / stats['queries'] if stats['queries'] else 0
            result['diagnostics'] = stats
        return result
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()
        partial.unlink(missing_ok=True)
        shutil.rmtree(work, ignore_errors=True)


def compare_tokens(candidate, reference, expected_count):
    expected_size = expected_count * 4
    if Path(candidate).stat().st_size != expected_size or Path(reference).stat().st_size != expected_size:
        raise ValueError('Token file size/count does not match expected reference')
    with open(candidate, 'rb') as a, open(reference, 'rb') as b:
        offset = 0
        while block := a.read(TEXT_BLOCK):
            if block != b.read(len(block)):
                raise ValueError(f'Token IDs differ at byte block {offset}')
            offset += len(block)
        if b.read(1):
            raise ValueError('Reference has extra bytes')


def compare_roundtrip(tokens, source, bpe):
    with open(tokens, 'rb') as encoded, open(source, 'rb') as original:
        for ids in read_uint32(encoded):
            decoded = b''.join(bpe.vocab[token_id] for token_id in ids)
            if decoded != original.read(len(decoded)):
                raise ValueError('Decoded token bytes differ from original input')
        if original.read(1):
            raise ValueError('Decoded bytes are shorter than original input')


def main():
    # Spawn reimports this small module, never the Transformer/GPU entrypoint.
    import argparse
    import signal

    def interrupted(signum, frame):
        raise InterruptedError('BPE coordinator interrupted; terminating workers')

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    parser.add_argument('--result', required=True)
    args = parser.parse_args()
    restrict_environment()
    job = json.loads(Path(args.job).read_text(encoding='utf-8'))
    result = encode_parallel(**job)
    Path(args.result).write_text(json.dumps(result), encoding='utf-8')


if __name__ == '__main__':
    main()
