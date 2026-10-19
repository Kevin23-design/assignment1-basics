"""Immutable native candidate packaging; core production BPE remains unchanged."""
from __future__ import annotations

import ast
from pathlib import Path
import shutil

import benchmark_task1_efficiency as common
from task1_decode_experiment import replace_function, source_hashes, verify_snapshot
from task1_native_runtime import verify_build

ROOT = Path(__file__).resolve().parent
METHODS = ('B0', 'CY-D', 'CY-ED')


def make_snapshot(directory, baseline, method, build_dir):
    if method not in METHODS:
        raise ValueError('Unknown native candidate')
    verify_snapshot(baseline)
    build_dir = Path(build_dir).resolve()
    build = verify_build(build_dir)
    directory = Path(directory).resolve()
    # Old W&B runs may have outputs inside a plain-source directory; do not clone them.
    directory.mkdir(parents=True, exist_ok=False)
    for name in baseline['source_sha256']:
        shutil.copyfile(Path(baseline['directory']) / name, directory / name)
    # Standalone verification and recording must understand native source/binaries.
    for name in ('verify_task1_plain.py', 'wandb_record.py', 'task1_native_runtime.py', 'build_task1_native.py'):
        shutil.copyfile(ROOT / name, directory / name)
    copied = {}
    for original, digest in build['files'].items():
        target = build['binary'] if original == build['binary'] else (
            'task1_native.pyx' if original.endswith('.pyx') else 'task1_native_generated.c')
        if target in copied:
            raise ValueError('Ambiguous native build inputs')
        shutil.copyfile(build_dir / original, directory / target)
        copied[target] = digest
    native_build = {**build, 'directory': str(directory), 'files': copied, 'origin_directory': str(build_dir)}
    common.save_json(directory / 'native_build.json', native_build)
    path = directory / 'bpe.py'
    source = path.read_text()
    if method != 'B0':
        source = replace_function(source, 'decode', '''    def decode(self, ids: list[int]) -> str:
        from task1_native_runtime import decode
        return decode(self, ids)
''', 'BPE')
    if method == 'CY-ED':
        tree = ast.parse(source)
        scope = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BPE')
        node = next(n for n in scope.body if getattr(n, 'name', None) == '_encode_uncached')
        original = ''.join(source.splitlines(keepends=True)[node.lineno - 1:node.end_lineno])
        original = original.replace('def _encode_uncached(', 'def _python_encode_uncached(', 1)
        source = replace_function(source, '_encode_uncached', original + '''
    def _encode_uncached(self, piece, cache_capacity=CACHE_CAPACITY):
        from task1_native_runtime import encode_uncached
        return encode_uncached(self, piece, cache_capacity)
''', 'BPE')
        tree = ast.parse(source)
        scope = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BPE')
        node = next(n for n in scope.body if getattr(n, 'name', None) == '_rebuild')
        original = ''.join(source.splitlines(keepends=True)[node.lineno - 1:node.end_lineno])
        source = replace_function(source, '_rebuild', original.rstrip() + '''
        self._native_encoder_state = None
        self._native_encode_metrics = None
''', 'BPE')
        path_worker = directory / 'task1_transport_runtime.py'
        worker_source = path_worker.read_text().replace(
            'stats.update(peak_rss_bytes=rss_bytes(), cache_entries=len(tokenizer._cache))',
            "stats.update(peak_rss_bytes=rss_bytes(), cache_entries=len(tokenizer._cache), "
            "native_encoding=getattr(tokenizer, '_native_encode_metrics', None))")
        path_worker.write_text(worker_source)
    path.write_text(source)
    memory = directory / 'bpe_memory_runtime.py'
    memory.write_text(replace_function(memory.read_text(), '_launch', '''def _launch(tokenizer, text, ranges, cpus, workers):
    from task1_pipeline_runtime import launch_all_first
    stats = {}
    result = launch_all_first(tokenizer, text, ranges, cpus, workers, False, stats)
    tokenizer._task1_worker_metrics = stats.get('worker_metadata', [])
    return result
'''))
    # Training, splitting, streaming, serialization, and encode's public dispatch stay intact.
    before = next(n for n in ast.parse((Path(baseline['directory']) / 'bpe.py').read_text()).body
                  if isinstance(n, ast.ClassDef) and n.name == 'BPE')
    after = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'BPE')
    changed = {'decode', '_rebuild', '_encode_uncached', '_python_encode_uncached'}
    assert [ast.dump(n) for n in before.body if getattr(n, 'name', None) not in changed] == [
        ast.dump(n) for n in after.body if getattr(n, 'name', None) not in changed]
    snapshot = {**baseline, 'directory': str(directory), 'method': method, 'native_build': native_build,
                'source_sha256': source_hashes(directory)}
    common.save_json(directory / 'snapshot.json', snapshot)
    verify_snapshot(snapshot)
    return snapshot
