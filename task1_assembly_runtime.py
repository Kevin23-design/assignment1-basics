"""Parent-only auxiliary collector. No automatic compilation or text cache."""
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
    directory = Path(directory).resolve()
    path = directory / ('assembly_build.json' if (directory / 'assembly_build.json').exists() else 'build.json')
    if not path.is_file():
        raise RuntimeError('Assembly extension missing; explicitly run build_task1_assembly.py')
    metadata = json.loads(path.read_text())
    for key, value in {'implementation': sys.implementation.name, 'cache_tag': sys.implementation.cache_tag,
                       'soabi': sysconfig.get_config_var('SOABI'), 'machine': platform.machine()}.items():
        if metadata.get(key) != value:
            raise RuntimeError(f'Assembly ABI mismatch: {key}; rebuild with the experiment Python')
    if not metadata['binary'].startswith('_task1_assembly.'):
        raise RuntimeError('Not an assembly extension build')
    files = metadata['files'] if full else {metadata['binary']: metadata['files'][metadata['binary']]}
    for name, digest in files.items():
        target = (directory / name).resolve()
        if not target.is_relative_to(directory) or not target.is_file() or sha(target) != digest:
            raise RuntimeError(f'Assembly build file missing or changed: {name}')
    return metadata


def load_assembly():
    global _loaded
    if _loaded is not None:
        return _loaded
    metadata = verify_build(_DIRECTORY, full=False)
    expected = (_DIRECTORY / metadata['binary']).resolve()
    spec = importlib.util.spec_from_file_location('_task1_assembly', expected)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load assembly extension: {expected}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != expected:
        raise RuntimeError('Assembly extension loaded from another path')
    _loaded = module
    return module


class NativeCollector:
    def __init__(self, count, vocabulary, method):
        if method not in ('M1', 'M2'):
            raise ValueError('Unknown native assembly method')
        # Pool indexing is valid only for the canonical dense vocab used here.
        if (not vocabulary or any(type(key) is not int for key in vocabulary)
                or min(vocabulary) != 0 or max(vocabulary) != len(vocabulary) - 1):
            raise ValueError('Native assembly requires a dense nonnegative vocabulary')
        self.count, self.next = count, 0
        self.pending, self.seen, self.ids = {}, set(), []
        began = time.perf_counter()
        native = load_assembly()
        imported = time.perf_counter()
        self.boxer = native.Boxer(len(vocabulary), method == 'M2')
        prepared = time.perf_counter()
        self.metrics = {'method': method, 'converted_ids': 0, 'fragments': 0,
                        'pool_entries': len(vocabulary) if method == 'M2' else 0,
                        'extension_path': str(native.__file__), 'import_seconds': imported - began,
                        'pool_prepare_seconds': prepared - imported}

    def accept(self, index, values):
        if (index in self.seen or not 0 <= index < self.count or not isinstance(values, array)
                or values.typecode != 'I' or values.itemsize != 4):
            raise RuntimeError('Duplicate, invalid or out-of-range partition')
        self.seen.add(index)
        self.pending[index] = values
        while self.next in self.pending:
            converted = self.boxer.convert(self.pending.pop(self.next))
            self.ids.extend(converted)
            self.metrics['converted_ids'] += len(converted)
            self.metrics['fragments'] += 1
            self.next += 1

    def finish(self):
        if self.next != self.count:
            raise RuntimeError('Incomplete transport output')
        return self.ids


def make_collector(vocabulary, count, method):
    if method == 'M0':
        from task1_transport_runtime import Collector
        return Collector(count)
    return NativeCollector(count, vocabulary, method)


def profile_decode(tokenizer, ids):
    module = load_assembly()
    text, metrics = module.profile_decode(tokenizer.vocab, ids)
    metrics['extension_path'] = str(module.__file__)
    tokenizer._native_decode_metrics = metrics
    return text
