"""Frozen assembly candidates, retaining the original native decoder binary."""
from __future__ import annotations

from pathlib import Path
import shutil

import benchmark_task1_efficiency as common
from task1_decode_experiment import replace_function, source_hashes, verify_snapshot
from task1_assembly_runtime import verify_build as verify_assembly_build
from task1_native_adapter_experiment import assert_identical_native
from task1_native_runtime import verify_build

ROOT = Path(__file__).resolve().parent
CASES = ('M0', 'M1', 'M2', 'M1-A1', 'M2-A1')


def make_snapshot(directory, origin, case, build_dir, diagnostic=False):
    if case not in CASES or origin['method'] != ('CY-ED' if case.endswith('-A1') else 'CY-D'):
        raise ValueError('Assembly candidate has the wrong encoder origin')
    if origin.get('diagnostic_instrumentation'):
        raise ValueError('Cannot use an instrumented origin as a normal candidate')
    verify_snapshot(origin)
    verify_build(origin['directory'])
    build_dir = Path(build_dir).resolve()
    build = verify_assembly_build(build_dir)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    for name in origin['source_sha256']:
        if Path(name).name != name:
            raise ValueError('Expected a flat immutable origin snapshot')
        shutil.copyfile(Path(origin['directory']) / name, directory / name)
    native_build = {**origin['native_build'], 'directory': str(directory)}
    common.save_json(directory / 'native_build.json', native_build)
    files = {}
    for name, digest in build['files'].items():
        target = build['binary'] if name == build['binary'] else (
            'task1_assembly.pyx' if name.endswith('.pyx') else 'task1_assembly_generated.c')
        if target in files:
            raise ValueError('Ambiguous assembly build artifacts')
        shutil.copyfile(build_dir / name, directory / target)
        files[target] = digest
    assembly_build = {**build, 'directory': str(directory), 'files': files, 'origin_directory': str(build_dir)}
    common.save_json(directory / 'assembly_build.json', assembly_build)
    for name in ('task1_assembly_runtime.py', 'build_task1_assembly.py', 'verify_task1_plain.py'):
        shutil.copyfile(ROOT / name, directory / name)
    method = case.split('-')[0]
    pipeline = directory / 'task1_pipeline_runtime.py'
    source = pipeline.read_text()
    target = 'collector = transport.Collector(len(ranges))'
    if source.count(target) != 1 or source.count('return collector.finish()') != 1:
        raise ValueError('Unsupported pipeline collector integration')
    source = source.replace(target, f"from task1_assembly_runtime import make_collector\n    collector = make_collector(tokenizer.vocab, len(ranges), {method!r})")
    source = source.replace('return collector.finish()', '''result = collector.finish()
            tokenizer._task1_assembly_metrics = getattr(collector, 'metrics', {
                'method': 'M0', 'converted_ids': len(result), 'pool_entries': 0, 'extension_path': None})
            return result''')
    pipeline.write_text(source)
    if diagnostic:
        bpe = directory / 'bpe.py'
        bpe.write_text(replace_function(bpe.read_text(), 'decode', '''    def decode(self, ids: list[int]) -> str:
        from task1_assembly_runtime import profile_decode
        return profile_decode(self, ids)
''', 'BPE'))
    snapshot = {**origin, 'directory': str(directory), 'assembly_case': case, 'assembly_method': method,
                'assembly_diagnostic': diagnostic, 'native_build': native_build, 'assembly_build': assembly_build,
                'source_sha256': source_hashes(directory)}
    common.save_json(directory / 'snapshot.json', snapshot)
    verify_snapshot(snapshot)
    verify_assembly_build(directory)
    assert_identical_native([origin, snapshot])
    return snapshot
