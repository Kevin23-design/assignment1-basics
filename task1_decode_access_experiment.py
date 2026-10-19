"""Immutable decode/split candidates derived solely from the accepted M2-A1 snapshot."""
from __future__ import annotations

from pathlib import Path
import re
import shutil

from benchmark_task1_efficiency import save_json
from task1_assembly_runtime import verify_build as verify_assembly
from task1_decode_access_runtime import verify_build
from task1_decode_experiment import replace_function, source_hashes, verify_snapshot
from task1_native_adapter_experiment import assert_identical_native

ROOT = Path(__file__).resolve().parent
CASES = ('N0', 'N1', 'N2')
PARTS = (64, 128, 256, 512)


def make_snapshot(directory, origin, case, parts, build_dir, diagnostic=False):
    if case not in CASES or parts < origin['workers'] or parts % origin['workers']:
        raise ValueError('Invalid decoder/partition configuration')
    if (origin.get('assembly_case') != 'M2-A1' or origin['method'] != 'CY-ED'
            or origin.get('adapter_case') != 'A1' or origin.get('assembly_diagnostic')
            or origin.get('diagnostic_instrumentation') or origin.get('decode_access_case')):
        raise ValueError('Requires the normal accepted M2-A1 origin')
    verify_snapshot(origin)
    verify_assembly(origin['directory'])
    build_dir = Path(build_dir).resolve()
    build = verify_build(build_dir)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    for name in origin['source_sha256']:
        if Path(name).name != name:
            raise ValueError('Expected flat immutable source snapshot')
        shutil.copyfile(Path(origin['directory']) / name, directory / name)
    updates = {}
    for key, filename in [('native_build', 'native_build.json'), ('assembly_build', 'assembly_build.json')]:
        updates[key] = {**origin[key], 'directory': str(directory)}
        save_json(directory / filename, updates[key])
    # N0 has no dependency on, or timed import of, the new extension.
    if case != 'N0':
        files = {}
        for name, digest in build['files'].items():
            target = build['binary'] if name == build['binary'] else (
                'task1_decode_access.pyx' if name.endswith('.pyx') else 'task1_decode_access_generated.c')
            if target in files:
                raise ValueError('Ambiguous decode access artifacts')
            shutil.copyfile(build_dir / name, directory / target)
            files[target] = digest
        updates['decode_access_build'] = {**build, 'directory': str(directory), 'files': files,
                                          'origin_directory': str(build_dir)}
        save_json(directory / 'decode_access_build.json', updates['decode_access_build'])
        for name in ('task1_decode_access_runtime.py', 'build_task1_decode_access.py'):
            shutil.copyfile(ROOT / name, directory / name)
    shutil.copyfile(ROOT / 'verify_task1_plain.py', directory / 'verify_task1_plain.py')
    bpe = directory / 'bpe.py'
    source, changed = re.subn(r'^_PARTITIONS_PER_WORKER = \d+$',
                              f'_PARTITIONS_PER_WORKER = {parts // origin["workers"]}', bpe.read_text(), flags=re.M)
    if changed != 1:
        raise ValueError('Unknown partition constant integration')
    if case != 'N0':
        source = replace_function(source, 'decode', f'''    def decode(self, ids: list[int]) -> str:
        from task1_decode_access_runtime import decode
        return decode(self, ids, {case!r}, diagnostic={diagnostic!r})
''', 'BPE')
    elif diagnostic:
        source = replace_function(source, 'decode', '''    def decode(self, ids: list[int]) -> str:
        from task1_assembly_runtime import profile_decode
        return profile_decode(self, ids)
''', 'BPE')
    bpe.write_text(source)
    # Fail explicitly instead of falling back to serial when a requested split
    # cannot be produced. Preflight separately marks these configurations absent.
    path = directory / 'bpe_memory_runtime.py'
    source = path.read_text()
    old = 'if ranges is None:\n        return None'
    if source.count(old) != 1:
        raise ValueError('Unknown safe partition guard')
    path.write_text(source.replace(old, "if ranges is None:\n        raise RuntimeError('Requested safe nonempty partitions unavailable')"))
    snapshot = {**origin, **updates, 'directory': str(directory), 'parts': parts,
                'decode_access_case': case, 'decode_access_diagnostic': diagnostic,
                'source_sha256': source_hashes(directory)}
    save_json(directory / 'snapshot.json', snapshot)
    verify_snapshot(snapshot)
    verify_assembly(directory)
    assert_identical_native([origin, snapshot])
    if case != 'N0':
        verify_build(directory)
    return snapshot
