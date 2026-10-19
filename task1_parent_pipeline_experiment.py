"""Frozen Q0-Q3 parent-only changes; preserve all accepted encoder/decoder binaries."""
from __future__ import annotations

from pathlib import Path
import shutil

from benchmark_task1_efficiency import save_json
from task1_decode_experiment import source_hashes, verify_snapshot
from task1_native_adapter_experiment import assert_identical_native
from task1_assembly_runtime import verify_build as verify_assembly
from task1_decode_access_runtime import verify_build as verify_decode
from task1_parent_pipeline_runtime import verify_build

ROOT = Path(__file__).resolve().parent
CASES = ('Q0', 'Q1', 'Q2', 'Q3')


def make_snapshot(directory, origin, case, build_dir, diagnostic=False):
    if (case not in CASES or origin.get('decode_access_case') != 'N2' or origin.get('parts') != 256
            or origin.get('decode_access_diagnostic') or origin.get('assembly_diagnostic')
            or origin.get('diagnostic_instrumentation') or origin.get('parent_pipeline_case')
            or origin.get('assembly_method') != 'M2' or origin.get('adapter_case') != 'A1'):
        raise ValueError('Requires the accepted normal N2/256 origin')
    verify_snapshot(origin)
    verify_assembly(origin['directory'])
    verify_decode(origin['directory'])
    build_dir = Path(build_dir).resolve()
    build = verify_build(build_dir)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    for name in origin['source_sha256']:
        if Path(name).name != name:
            raise ValueError('Expected a flat immutable snapshot')
        shutil.copyfile(Path(origin['directory']) / name, directory / name)
    updates = {}
    for key in ('native_build', 'assembly_build', 'decode_access_build'):
        updates[key] = {**origin[key], 'directory': str(directory)}
        save_json(directory / (key + '.json'), updates[key])
    shutil.copyfile(ROOT / 'verify_task1_plain.py', directory / 'verify_task1_plain.py')
    if case != 'Q0':
        path = directory / 'task1_transport_runtime.py'
        source = path.read_text()
        before = '                on_result(index, payload)\n                dispatch(connection)'
        if source.count(before) != 1:
            raise ValueError('Unsupported dispatch/assembly sequence')
        path.write_text(source.replace(before, '                dispatch(connection)\n                on_result(index, payload)'))
    if case in ('Q2', 'Q3'):
        files = {}
        for name, digest in build['files'].items():
            target = build['binary'] if name == build['binary'] else (
                'task1_parent_assembly.pyx' if name.endswith('.pyx') else 'task1_parent_assembly_generated.c')
            if target in files:
                raise ValueError('Ambiguous parent assembly artifacts')
            shutil.copyfile(build_dir / name, directory / target)
            files[target] = digest
        updates['parent_assembly_build'] = {**build, 'directory': str(directory), 'files': files,
                                            'origin_directory': str(build_dir)}
        save_json(directory / 'parent_assembly_build.json', updates['parent_assembly_build'])
        for name in ('task1_parent_pipeline_runtime.py', 'build_task1_parent_assembly.py'):
            shutil.copyfile(ROOT / name, directory / name)
        path = directory / 'task1_pipeline_runtime.py'
        source = path.read_text()
        old = "from task1_assembly_runtime import make_collector\n    collector = make_collector(tokenizer.vocab, len(ranges), 'M2')"
        if source.count(old) != 1:
            raise ValueError('Unsupported parent collector integration')
        path.write_text(source.replace(old, "from task1_parent_pipeline_runtime import make_collector\n    "
                                       f"collector = make_collector(tokenizer.vocab, len(ranges), {case!r})"))
    if diagnostic:
        path = directory / 'task1_pipeline_runtime.py'
        source = path.read_text()
        old = '            result = collector.finish()'
        if source.count(old) != 1:
            raise ValueError('Unsupported final assembly diagnostic')
        path.write_text(source.replace(old, "            tick = transport.stamp(diagnostic)\n" + old +
                                       "\n            transport.elapsed(stats, 'finish_assembly_seconds', tick, diagnostic)"))
    snapshot = {**origin, **updates, 'directory': str(directory), 'parent_pipeline_case': case,
                'parent_pipeline_diagnostic': diagnostic,
                'assembly_method': case if case in ('Q2', 'Q3') else 'M2', 'source_sha256': source_hashes(directory)}
    save_json(directory / 'snapshot.json', snapshot)
    verify_snapshot(snapshot)
    assert_identical_native([origin, snapshot])
    # Every candidate retains identical original assembly and N2 artifacts.
    for key in ('assembly_build', 'decode_access_build'):
        if snapshot[key]['files'] != origin[key]['files']:
            raise ValueError('Original assembly or decoder artifacts changed')
    if case in ('Q2', 'Q3'):
        verify_build(directory)
    return snapshot
