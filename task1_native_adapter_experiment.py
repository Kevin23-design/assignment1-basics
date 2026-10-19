"""Frozen loader-only candidates; reuses already compiled native artifacts."""
from __future__ import annotations

import ast
from pathlib import Path
import shutil

from benchmark_task1_efficiency import save_json
from task1_decode_experiment import replace_function, source_hashes, verify_snapshot
from task1_native_runtime import verify_build

CASES = ('A0', 'A1', 'OLD-CY-ED')

FAST_LOADER = '''def load_native():
    global _loaded
    if _loaded is not None:
        return _loaded
    metadata = verify_build(_DIRECTORY, full=False)
    expected = (_DIRECTORY / metadata['binary']).resolve()
    spec = importlib.util.spec_from_file_location('_task1_native', expected)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load native extension: {expected}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != expected:
        raise RuntimeError('Native extension loaded from the wrong snapshot')
    # Do not publish a failed or only partially validated load.
    _loaded = module
    return module
'''

# Appended ONLY to separate diagnostic snapshots. Normal candidates have no
# per-call diagnostic flag, counters, timers or monkey patches.
DIAGNOSTIC_SETUP = '''

_adapter_loader_metrics = {'calls': 0, 'validation_calls': 0, 'path_resolve_calls': 0,
                           'load_seconds': 0., 'validation_seconds': 0., 'path_resolve_seconds': 0.}
_AdapterBasePath = type(Path())
class _AdapterPath(_AdapterBasePath):
    def resolve(self, *args, **kwargs):
        started = time.perf_counter()
        _adapter_loader_metrics['path_resolve_calls'] += 1
        try:
            return super().resolve(*args, **kwargs)
        finally:
            _adapter_loader_metrics['path_resolve_seconds'] += time.perf_counter() - started
Path = _AdapterPath
_DIRECTORY = Path(str(_DIRECTORY))
_adapter_original_verify = verify_build
_adapter_original_load = load_native
_adapter_original_decode = decode

def _adapter_verify(*args, **kwargs):
    started = time.perf_counter()
    _adapter_loader_metrics['validation_calls'] += 1
    try:
        return _adapter_original_verify(*args, **kwargs)
    finally:
        _adapter_loader_metrics['validation_seconds'] += time.perf_counter() - started

def _adapter_load():
    started = time.perf_counter()
    _adapter_loader_metrics['calls'] += 1
    try:
        return _adapter_original_load()
    finally:
        _adapter_loader_metrics['load_seconds'] += time.perf_counter() - started

def _adapter_decode(tokenizer, ids):
    result = _adapter_original_decode(tokenizer, ids)
    tokenizer._native_decode_metrics['loader'] = dict(_adapter_loader_metrics)
    return result

verify_build = _adapter_verify
load_native = _adapter_load
decode = _adapter_decode
'''


def native_identity(snapshot):
    """Compare both artifact roles and hashes, ignoring original directories."""
    verify_snapshot(snapshot)
    build = verify_build(snapshot['directory'])
    groups = {kind: sorted(digest for name, digest in build['files'].items() if name.endswith(suffixes))
              for kind, suffixes in [('pyx', ('.pyx', '.pxd')), ('c', ('.c',)), ('binary', ('.so', '.pyd'))]}
    if any(not values for values in groups.values()):
        raise ValueError('Native snapshot must retain pyx, generated C and binary fingerprints')
    return {**groups, 'soabi': build['soabi'], 'cache_tag': build['cache_tag'],
            'machine': build['machine'], 'compile_flags': build['compile_flags']}


def assert_identical_native(snapshots):
    identities = [native_identity(snapshot) for snapshot in snapshots]
    if not identities or any(identity != identities[0] for identity in identities[1:]):
        raise ValueError('Candidates must use identical native source, generated C and binary')
    return identities[0]


def make_snapshot(directory, origin, case, diagnostic=False):
    if case not in CASES or origin['method'] != ('CY-D' if case == 'A0' else 'CY-ED'):
        raise ValueError('Adapter candidate does not match its native origin')
    verify_snapshot(origin)
    verify_build(origin['directory'])
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    for name in origin['source_sha256']:
        if Path(name).name != name:
            raise ValueError('Expected flat immutable native snapshot')
        shutil.copyfile(Path(origin['directory']) / name, directory / name)
    build = {**origin['native_build'], 'directory': str(directory)}
    save_json(directory / 'native_build.json', build)
    helper = directory / 'task1_native_runtime.py'
    before = helper.read_text()
    updated = replace_function(before, 'load_native', FAST_LOADER) if case == 'A1' else before
    if case == 'A1':
        def strip_loader(src):
            return [ast.dump(node) for node in ast.parse(src).body
                    if getattr(node, 'name', None) != 'load_native']
        if strip_loader(before) != strip_loader(updated):
            raise ValueError('A1 must change only load_native')
    if diagnostic:
        updated += DIAGNOSTIC_SETUP
        worker = directory / 'task1_transport_runtime.py'
        source = worker.read_text()
        line = "stats.update(peak_rss_bytes=rss_bytes(), cache_entries=len(tokenizer._cache)"
        if line not in source:
            raise ValueError('Cannot attach diagnostic worker metadata to this snapshot')
        source = source.replace(line, line + ", native_loading=getattr(sys.modules.get('task1_native_runtime'), "
                                "'_adapter_loader_metrics', None)")
        worker.write_text(source)
    helper.write_text(updated)
    snapshot = {**origin, 'directory': str(directory), 'adapter_case': case,
                'diagnostic_instrumentation': diagnostic, 'native_build': build,
                'source_sha256': source_hashes(directory)}
    save_json(directory / 'snapshot.json', snapshot)
    verify_snapshot(snapshot)
    assert_identical_native([origin, snapshot])
    return snapshot
