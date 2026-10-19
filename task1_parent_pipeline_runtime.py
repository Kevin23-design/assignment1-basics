"""Private parent collectors. No source text, reference-token sizes or retained workers."""
from __future__ import annotations

from array import array
import importlib.util
import json
from pathlib import Path
import platform
import sys
import sysconfig
import time

from task1_native_runtime import sha

_DIRECTORY = Path(__file__).resolve().parent
_loaded = None


def verify_build(directory, full=True):
    if sys.implementation.name != 'cpython' or sys.version_info[:2] != (3, 13) or sysconfig.get_config_var('Py_GIL_DISABLED'):
        raise RuntimeError('Parent assembly requires GIL-enabled CPython 3.13')
    directory = Path(directory).resolve()
    path = directory / ('parent_assembly_build.json' if (directory / 'parent_assembly_build.json').exists() else 'build.json')
    if not path.is_file():
        raise RuntimeError('Parent assembly extension missing; explicitly run build_task1_parent_assembly.py')
    metadata = json.loads(path.read_text())
    for key, value in {'implementation': sys.implementation.name, 'cache_tag': sys.implementation.cache_tag,
                       'soabi': sysconfig.get_config_var('SOABI'), 'machine': platform.machine(), 'gil_disabled': False}.items():
        if metadata.get(key) != value:
            raise RuntimeError(f'Parent assembly ABI mismatch: {key}')
    if not metadata['binary'].startswith('_task1_parent_assembly.'):
        raise RuntimeError('Not a parent assembly build')
    files = metadata['files'] if full else {metadata['binary']: metadata['files'][metadata['binary']]}
    for name, digest in files.items():
        target = (directory / name).resolve()
        if not target.is_relative_to(directory) or not target.is_file() or sha(target) != digest:
            raise RuntimeError(f'Parent assembly artifact missing or changed: {name}')
    return metadata


def load_parent():
    global _loaded
    if _loaded is not None:
        return _loaded
    build = verify_build(_DIRECTORY, full=False)
    expected = (_DIRECTORY / build['binary']).resolve()
    spec = importlib.util.spec_from_file_location('_task1_parent_assembly', expected)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load parent assembly extension: {expected}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != expected:
        raise RuntimeError('Parent assembly extension loaded from another path')
    _loaded = module
    return module


class Collector:
    def __init__(self, vocabulary, count, case):
        if case not in ('Q2', 'Q3'):
            raise ValueError('Unknown parent assembly strategy')
        if type(count) is not int or count < 1:
            raise ValueError('Partition count must be positive')
        if (not vocabulary or any(type(key) is not int for key in vocabulary)
                or min(vocabulary) != 0 or max(vocabulary) != len(vocabulary) - 1):
            raise ValueError('Requires dense nonnegative vocabulary')
        self.count, self.case, self.next = count, case, 0
        self.seen, self.pending, self.ids = set(), {}, []
        self.chunks = [None] * count if case == 'Q3' else None
        self.tokens = 0
        self.pending_tokens = 0
        began = time.perf_counter()
        native = load_parent()
        loaded = time.perf_counter()
        self.pool = native.Pool(len(vocabulary))
        prepared = time.perf_counter()
        self.metrics = {'method': case, 'converted_ids': 0, 'fragments': 0, 'pool_entries': len(vocabulary),
                        'extension_path': str(native.__file__), 'temporary_id_lists': 0,
                        'allocation_basis': 'observed_worker_lengths' if case == 'Q3' else 'normal_list_growth',
                        'retained_raw_bytes_peak': 0, 'import_seconds': loaded - began,
                        'pool_prepare_seconds': prepared - loaded}

    def accept(self, index, values):
        if (type(index) is not int or index in self.seen or not 0 <= index < self.count
                or type(values) is not array or values.typecode != 'I' or values.itemsize != 4):
            raise RuntimeError('Duplicate, malformed or out-of-range partition')
        self.seen.add(index)
        if self.case == 'Q3':
            if self.chunks is None:
                raise RuntimeError('Cannot accept after final assembly')
            self.chunks[index] = values
            self.tokens += len(values)
            self.metrics['retained_raw_bytes_peak'] = self.tokens * 4
            self.metrics['fragments'] += 1
        else:
            self.pending[index] = values
            self.pending_tokens += len(values)
            self.metrics['retained_raw_bytes_peak'] = max(self.metrics['retained_raw_bytes_peak'], self.pending_tokens * 4)
            while self.next in self.pending:
                block = self.pending.pop(self.next)
                self.metrics['converted_ids'] += self.pool.append(self.ids, block)
                self.pending_tokens -= len(block)
                self.metrics['fragments'] += 1
                self.next += 1

    def finish(self):
        if len(self.seen) != self.count:
            raise RuntimeError('Incomplete transport output')
        if self.case == 'Q2':
            if self.next != self.count:
                raise RuntimeError('Incomplete native assembly')
            return self.ids
        if self.chunks is None:
            raise RuntimeError('Final list already assembled')
        try:
            result = self.pool.assemble_owned(self.chunks)
            self.metrics['converted_ids'] = len(result)
            return result
        finally:
            self.chunks = None


def make_collector(vocabulary, count, case):
    return Collector(vocabulary, count, case)


def check_result(snapshot, result):
    verify_build(snapshot['directory'])
    metrics = result.get('assembly') or {}
    if (metrics.get('method') != snapshot['parent_pipeline_case'] or metrics.get('converted_ids') != result['token_count']
            or metrics.get('fragments') != snapshot['parts']):
        raise ValueError('Candidate bypassed parent assembly strategy')
    expected = Path(snapshot['directory']) / snapshot['parent_assembly_build']['binary']
    if Path(metrics['extension_path']).resolve() != expected.resolve():
        raise ValueError('Parent assembly extension loaded from another snapshot')
