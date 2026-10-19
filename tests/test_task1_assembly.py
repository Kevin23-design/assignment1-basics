"""Tiny native assembly correctness, ownership, provenance and experiment tests."""
from array import array
from contextlib import contextmanager
import gc
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from bpe_decode_runtime import serial
import benchmark_task1_assembly as benchmark
import build_task1_assembly as builder
import build_task1_native as native_builder
from task1_decode_experiment import verify_snapshot
import task1_assembly_experiment as packaging
import task1_assembly_runtime as runtime
from task1_native_adapter_experiment import make_snapshot as adapter_snapshot
from tests.test_task1_native_adapter import adapter_history
from tests.test_task1_encode_transport import sample


@pytest.fixture(scope='session')
def builds(tmp_path_factory):
    pytest.importorskip('Cython')
    root = tmp_path_factory.mktemp('assembly-builds')
    native_builder.build(root / 'native')
    builder.build(root / 'assembly')
    return root / 'native', root / 'assembly'


@pytest.fixture(scope='session')
def extension(builds):
    directory = builds[1]
    metadata = runtime.verify_build(directory)
    spec = importlib.util.spec_from_file_location('_task1_assembly', directory / metadata['binary'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_history(tmp_path, builds):
    args, prior = adapter_history(tmp_path, builds[0])
    origins = {case: adapter_snapshot(tmp_path / ('prior-' + case),
                                     prior['candidate_snapshots']['CY-D' if case == 'A0' else 'CY-ED'], case)
               for case in ('A0', 'A1')}
    history = {**prior, 'recommended': 'A0', 'snapshot': origins['A0'], 'candidate_snapshots': origins}
    report = tmp_path / 'adapter-selection.json'
    report.write_text(json.dumps(history))
    confirmation = json.loads(Path(args.plain_confirmation).read_text())
    confirmation.update(adapter_case='A0', bpe_source=str(Path(origins['A0']['directory']) / 'bpe.py'))
    confirmation['native_decode']['extension_path'] = str(Path(origins['A0']['directory']) / origins['A0']['native_build']['binary'])
    confirmation_path = tmp_path / 'adapter-confirmation.json'
    confirmation_path.write_text(json.dumps(confirmation))
    args = benchmark.parse_args(['--previous-selection', str(report), '--plain-confirmation', str(confirmation_path),
                                 '--build-dir', str(builds[1]), '--development', '--no-wandb', '--repeats', '1',
                                 '--output-dir', str(tmp_path / 'out')])
    return args, history


@pytest.mark.parametrize('reuse', [False, True])
def test_boxing_values_empty_bounds_and_partial_failure(extension, reuse):
    boxer = extension.Boxer(32000, reuse)
    assert boxer.convert(array('I')) == []
    assert boxer.convert(array('I', [0, 256, 10000, 31999])) == [0, 256, 10000, 31999]
    with pytest.raises(ValueError, match='vocabulary'):
        boxer.convert(array('I', [10000, 32000]))
    for buffer in (None, b'abcd', array('i', [1, 2]), array('Q', [1, 2]), memoryview(array('I', [1, 2, 3]))[::2]):
        with pytest.raises((ValueError, TypeError, BufferError)):
            boxer.convert(buffer)
    unaligned = memoryview(bytearray(13))[1:].cast('I')
    with pytest.raises(BufferError, match='Unaligned'):
        boxer.convert(unaligned)
    data = array('I', [1])
    assert boxer.convert(data) == [1]
    data.append(2)  # No exported view remains after convert returns.


def test_pool_identity_lifetime_and_reference_ownership(extension):
    normal, pooled = extension.Boxer(32000, False), extension.Boxer(32000, True)
    first, second = pooled.convert(array('I', [10000, 10000])), pooled.convert(array('I', [10000]))
    assert first[0] is first[1] is second[0]
    ordinary = normal.convert(array('I', [10000, 10000]))
    assert ordinary[0] == ordinary[1] and ordinary[0] is not ordinary[1]
    value = first[0]
    before = sys.getrefcount(value)
    for _ in range(100):
        with pytest.raises(ValueError):
            pooled.convert(array('I', [10000, 32000]))
    assert sys.getrefcount(value) == before
    del pooled
    gc.collect()
    assert first == [10000, 10000] and second == [10000]
    assert first[0] is second[0]
    assert sys.getrefcount(value) == before - 1


def test_overflow_checks(extension):
    assert extension._checked_list_size(0) == 0
    for size in (-1, sys.maxsize, sys.maxsize + 1):
        with pytest.raises(OverflowError):
            extension._checked_list_size(size)
    with pytest.raises(OverflowError):
        extension.Boxer(-1)


def test_profile_matches_unicode_replace_and_phases(extension):
    vocab = {i: bytes([i]) for i in range(256)}
    for ids in ([], list('中🙂\r\n\0'.encode()), [0xe4], [0xf0, 0x9f, 0x99, 0x82, 0]):
        text, metrics = extension.profile_decode(vocab, ids)
        assert text == serial(vocab, ids)
        assert metrics['diagnostic_implementation'] == 'assembly-profile'
        assert all(value >= 0 for key, value in metrics.items() if key.endswith('_seconds'))
    for ids in ([-1], [256], [2**100]):
        with pytest.raises(KeyError):
            extension.profile_decode(vocab, ids)


def test_collectors_out_of_order_empty_duplicate_missing_and_pool(builds, monkeypatch):
    monkeypatch.setattr(runtime, '_DIRECTORY', builds[1])
    monkeypatch.setattr(runtime, '_loaded', None)
    vocabulary = {i: b'a' for i in range(32000)}
    for method in ('M0', 'M1', 'M2'):
        collector = runtime.make_collector(vocabulary, 3, method)
        collector.accept(2, array('I', [10000]))
        collector.accept(0, array('I', [10000]))
        with pytest.raises(RuntimeError, match='Incomplete'):
            collector.finish()
        with pytest.raises(RuntimeError):
            collector.accept(0, array('I', [1]))
        collector.accept(1, array('I'))
        ids = collector.finish()
        assert type(ids) is list and ids == [10000, 10000]
        if method == 'M2':
            assert ids[0] is ids[1]
        del collector
        assert ids == [10000, 10000]
    with pytest.raises(ValueError, match='dense'):
        runtime.NativeCollector(1, {0: b'a', 100: b'b'}, 'M2')
    collector = runtime.NativeCollector(1, vocabulary, 'M2')
    with pytest.raises(ValueError):
        collector.accept(0, array('I', [32000]))
    with pytest.raises(RuntimeError):
        collector.finish()


@pytest.mark.parametrize('case,diagnostic', [('M0', False), ('M1', False), ('M2', False), ('M1-A1', False),
                                           ('M2-A1', False), ('M0', True), ('M1', True), ('M2', True), ('M2-A1', True)])
def test_public_interface_original_decoder_and_ordinary_snapshot(tmp_path, builds, case, diagnostic):
    args, history = make_history(tmp_path, builds)
    origin = history['candidate_snapshots']['A1' if case.endswith('-A1') else 'A0']
    snapshot = packaging.make_snapshot(tmp_path / 'candidate', origin, case, builds[1], diagnostic)
    job = {k: history[k] for k in ('tokenizer_prefix', 'valid_file', 'reference', 'expected_tokens')}
    job.update(snapshot=snapshot, method=snapshot['method'], diagnostic=diagnostic, cpus=[], workers=2,
               development=True, plain=False, active_wandb=False, input_sha256=history['input_sha256'])
    result = benchmark.native.launch_job(job, tmp_path / 'job.json', tmp_path / 'result.json', tmp_path / 'trial.log')
    assert result['status'] == 'ok' and result['roundtrip_passed'] and result['ids_match_reference']
    assert result['assembly']['method'] == case.split('-')[0]
    assert result['assembly']['converted_ids'] == history['expected_tokens']
    metrics = result['native_decode']
    if diagnostic:
        assert metrics['diagnostic_implementation'] == 'assembly-profile'
        assert 'copy_first_touch_seconds' in metrics
    else:
        assert 'diagnostic_implementation' not in metrics
        assert Path(metrics['extension_path']).name == snapshot['native_build']['binary']
        assert (Path(snapshot['directory']) / 'bpe.py').read_bytes() == (Path(origin['directory']) / 'bpe.py').read_bytes()
        plain = benchmark.native.launch_job({**job, 'plain': True}, tmp_path / 'plain-job.json',
                                            tmp_path / 'plain.json', tmp_path / 'plain.log')
        assert plain['status'] == 'ok' and plain['patching'] is False
        assert plain['assembly']['method'] == case.split('-')[0]
    assert (Path(snapshot['directory']) / snapshot['native_build']['binary']).read_bytes() == (
        Path(origin['directory']) / origin['native_build']['binary']).read_bytes()


@pytest.mark.parametrize('failure', ['diagnostic_origin', 'binary', 'abi', 'confirmation', 'cpu'])
def test_provenance_and_normal_origin_guards(tmp_path, builds, failure):
    args, history = make_history(tmp_path, builds)
    if failure == 'diagnostic_origin':
        origin = history['candidate_snapshots']['A1']
        origin['diagnostic_instrumentation'] = True
        (Path(origin['directory']) / 'snapshot.json').write_text(json.dumps(origin))
        Path(args.previous_selection).write_text(json.dumps(history))
    elif failure in ('binary', 'abi'):
        origin = history['candidate_snapshots']['A1']
        name = origin['native_build']['binary'] if failure == 'binary' else 'native_build.json'
        (Path(origin['directory']) / name).write_bytes(b'changed')
    elif failure == 'confirmation':
        path = Path(args.plain_confirmation)
        conf = json.loads(path.read_text())
        conf['adapter_case'] = 'A1'
        path.write_text(json.dumps(conf))
    else:
        args.cpu_ids = [0]
    with pytest.raises((ValueError, RuntimeError)):
        benchmark.load_history(args)


def test_auxiliary_binary_abi_and_source_fingerprints(tmp_path, builds):
    _, history = make_history(tmp_path, builds)
    snapshot = packaging.make_snapshot(tmp_path / 'candidate', history['snapshot'], 'M2', builds[1])
    assert 'task1_assembly.pyx' in snapshot['source_sha256']
    assert 'task1_assembly_generated.c' in snapshot['source_sha256']
    binary = Path(snapshot['directory']) / snapshot['assembly_build']['binary']
    binary.write_bytes(b'changed')
    with pytest.raises(ValueError):
        verify_snapshot(snapshot)
    with pytest.raises(RuntimeError):
        runtime.verify_build(snapshot['directory'])


@pytest.mark.parametrize('best,promote,combo,expected', [('M0', False, False, 'M0'), ('M1', True, False, 'M1'),
                                                      ('M2', False, False, 'M0'), ('M2', True, True, 'M2-A1')])
def test_gates_mechanism_only_and_conditional_combination(best, promote, combo, expected):
    rows = []
    def trial(stage, case, repeat=0, diagnostic=False):
        times = {'M0': 10, 'M1': 11, 'M2': 12, 'M1-A1': 9, 'M2-A1': 7}
        times[best] = 8
        if not promote and stage.startswith('assembly-paired'):
            times['M0'] = 7
        if not combo and stage.startswith('combination'):
            times['M1-A1'] = times['M2-A1'] = 15
        row = {'trial_id': str(len(rows)), 'stage': stage, 'method': case, 'repeat': repeat,
               'diagnostic': diagnostic, 'status': 'ok', 'ids_match_reference': True,
               'roundtrip_passed': True, 'total_seconds': times[case]}
        rows.append(row)
        return row
    result = benchmark.run_stages(SimpleNamespace(seed=42, repeats=5, diagnostics=True), trial)
    assert result['recommended'] == expected
    assert any(r['stage'].startswith('combination') for r in rows) == (promote and best != 'M0')
    assert any(r['stage'].startswith('integer-pool-mechanism') for r in rows) == (promote and best == 'M2')


def test_tiny_complete_exploration_and_confirmation(tmp_path, builds):
    args, _ = make_history(tmp_path, builds)
    args.diagnostics = True
    benchmark.exploration(args)
    output = Path(args.output_dir)
    selected = json.loads((output / 'selection.json').read_text())
    assert selected['completed'] and selected['measurements_valid']
    assert selected['upload_status'] == 'disabled'
    rows = json.loads((output / 'results.json').read_text())
    assert all(row['status'] == 'ok' and row['assembly']['converted_ids'] == selected['expected_tokens'] for row in rows)
    assert json.loads((output / 'integrity.json').read_text()) == {'changed_inputs': [], 'source_changed': False}
    # Gate arithmetic is tested independently; toy latency is not a performance claim.
    selected['ready_for_plain_confirmation'] = True
    (output / 'selection.json').write_text(json.dumps(selected))
    args = benchmark.parse_args(['--phase', 'plain-confirmation', '--selection-report', str(output / 'selection.json'),
                                 '--output-dir', str(tmp_path / 'plain'), '--development', '--no-wandb'])
    benchmark.plain_confirmation(args)
    result = json.loads((tmp_path / 'plain/confirmation.json').read_text())
    assert result['status'] == 'ok' and result['assembly_case'] == selected['recommended']


def test_native_snapshot_reload_streaming_and_cache(tmp_path, builds):
    _, history = make_history(tmp_path, builds)
    snapshot = packaging.make_snapshot(tmp_path / 'candidate', history['candidate_snapshots']['A1'], 'M2-A1', builds[1])
    code = f'''
import sys
sys.path.insert(0,{snapshot['directory']!r})
from bpe import BPE
bpe=BPE()
bpe.load({history['tokenizer_prefix']!r})
text={sample()!r}
ids=bpe.encode(text)
assert ids==bpe._encode_serial(text)
assert list(bpe.encode_iterable([text[i:i+3] for i in range(0,len(text),3)]))==ids
bpe.save({str(tmp_path / 'saved')!r})
bpe.load({str(tmp_path / 'saved')!r})
assert bpe._native_encoder_state is None and not bpe._cache
assert bpe.encode(text)==ids and bpe.decode(ids)==text
'''
    subprocess.run([sys.executable, '-c', code], check=True, timeout=15)


def test_partial_construction_under_debug_allocator(builds):
    import os
    directory = builds[1]
    code = f'''
import sys
from array import array
sys.path.insert(0,{str(directory)!r})
import _task1_assembly as native
for reuse in (False, True):
    boxer=native.Boxer(32000,reuse)
    for _ in range(1000):
        try:boxer.convert(array('I',[10000,32000]))
        except ValueError:pass
        else:raise AssertionError('Out of range ID accepted')
    result=boxer.convert(array('I',[10000,10000]))
    del boxer
    assert result==[10000,10000]
'''
    subprocess.run([sys.executable, '-c', code], env={**os.environ, 'PYTHONMALLOC': 'debug'}, check=True, timeout=15)


def test_auxiliary_abi_mismatch_is_explicit(tmp_path, builds):
    from shutil import copytree
    directory = tmp_path / 'assembly'
    copytree(builds[1], directory)
    path = directory / 'build.json'
    metadata = json.loads(path.read_text())
    metadata['soabi'] = 'another-python'
    path.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match='ABI mismatch'):
        runtime.verify_build(directory)


def test_upload_dict_summary_and_finish_failure(tmp_path, builds):
    _, history = make_history(tmp_path, builds)
    snapshot = packaging.make_snapshot(tmp_path / 'candidate', history['snapshot'], 'M2', builds[1])
    (tmp_path / 'manifest.json').write_text(json.dumps({'origin_snapshots': history['candidate_snapshots']}))
    selected = {'completed': True, 'measurements_valid': True, 'recommended': 'M2',
                'snapshot': snapshot, 'candidate_snapshots': {'M2': snapshot}, 'integration': {'passed': True}}
    class Summary:
        def update(self, value):
            assert isinstance(value, dict)
    class Run:
        id, summary = 'assembly-test', Summary()
        def log(self, value):
            pass
        def log_artifact(self, value):
            pass
        def save(self, *args, **kwargs):
            pass
    benchmark.publish(Run(), tmp_path, [], selected)
    @contextmanager
    def failure():
        yield Run()
        raise RuntimeError('Upload failure after successful measurement')
    with benchmark.native.recording_context(failure(), tmp_path, selected):
        pass
    assert selected['completed'] and selected['measurements_valid']
    assert selected['upload_status'] == 'failed'


@pytest.mark.parametrize('signal_name', ['SIGINT', 'SIGTERM'])
def test_assembly_interrupt_reaps_all_workers(tmp_path, builds, signal_name):
    import signal
    import time
    import psutil
    _, history = make_history(tmp_path, builds)
    snapshot = packaging.make_snapshot(tmp_path / 'candidate', history['snapshot'], 'M2', builds[1])
    ready = tmp_path / 'ready.json'
    code = f'''
import sys,json,time
from pathlib import Path
sys.path.insert(0,{snapshot['directory']!r})
from bpe import BPE
import task1_transport_runtime as transport
original=transport.await_ready
def wait(*args):
    original(*args)
    Path({str(ready)!r}).write_text(json.dumps([p.pid for p in args[1]]))
    time.sleep(60)
transport.await_ready=wait
bpe=BPE()
bpe.load({history['tokenizer_prefix']!r})
bpe.encode({sample()!r})
'''
    with (tmp_path / 'signal.log').open('w') as log:
        process = subprocess.Popen([sys.executable, '-c', code], stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 15
            while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(.02)
            assert ready.exists()
            children = json.loads(ready.read_text())
            process.send_signal(getattr(signal, signal_name))
            process.wait(timeout=10)
            assert process.returncode != 0
            assert all(not psutil.pid_exists(pid) for pid in children)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


def test_assembly_conversion_error_reclaims_workers(tmp_path, builds):
    _, history = make_history(tmp_path, builds)
    snapshot = packaging.make_snapshot(tmp_path / 'candidate', history['snapshot'], 'M2', builds[1])
    code = f'''
import sys
sys.path.insert(0,{snapshot['directory']!r})
from bpe import BPE
import task1_transport_runtime as transport
import task1_assembly_runtime as assembly
original=transport.subprocess.Popen
children=[]
def launch(*args,**kwargs):
    child=original(*args,**kwargs)
    children.append(child)
    return child
transport.subprocess.Popen=launch
def fail(*args):raise ValueError('Injected conversion failure')
assembly.NativeCollector.accept=fail
bpe=BPE()
bpe.load({history['tokenizer_prefix']!r})
try:bpe.encode({sample()!r})
except ValueError:
    assert children and all(child.poll() is not None for child in children)
else:raise AssertionError('Partial list returned after conversion failure')
'''
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, timeout=15)


def test_active_offline_plain_pooled_assembly(tmp_path, builds, monkeypatch):
    _, history = make_history(tmp_path, builds)
    snapshot = packaging.make_snapshot(tmp_path / 'candidate', history['snapshot'], 'M2', builds[1])
    for name, value in [('WANDB_MODE', 'offline'), ('WANDB_ENTITY', 'test'), ('STUDENT_ID', 'assembly-test')]:
        monkeypatch.setenv(name, value)
    job = {k: history[k] for k in ('tokenizer_prefix', 'valid_file', 'reference', 'expected_tokens')}
    job.update(snapshot=snapshot, method=snapshot['method'], diagnostic=False, cpus=[], workers=2,
               development=True, plain=True, active_wandb=True, input_sha256=history['input_sha256'])
    result = benchmark.native.launch_job(job, tmp_path / 'job.json', tmp_path / 'result.json', tmp_path / 'trial.log')
    assert result['status'] == 'ok' and result['active_wandb'] and result['patching'] is False
    assert result['assembly']['method'] == 'M2'
    assert Path(result['native_decode']['extension_path']).name == snapshot['native_build']['binary']
