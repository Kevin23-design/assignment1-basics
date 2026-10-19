"""Tiny native reads, ownership, exact transport, provenance and staged gates."""
from contextlib import contextmanager
import gc
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import sysconfig
from types import SimpleNamespace

import pytest

import benchmark_task1_decode_access as benchmark
import build_task1_decode_access as builder
from bpe_decode_runtime import serial
import task1_decode_access_experiment as packaging
import task1_decode_access_runtime as runtime
from tests import test_task1_assembly as assembly_tests
from task1_assembly_experiment import make_snapshot as assembly_snapshot
from tests.test_task1_encode_transport import sample, tokenizer

pytestmark = pytest.mark.skipif(
    sys.version_info[:2] != (3, 13) or bool(sysconfig.get_config_var('Py_GIL_DISABLED')),
    reason='Decode access experiment requires GIL-enabled CPython 3.13',
)


@pytest.fixture(scope='session')
def builds(tmp_path_factory):
    assembly_builds = assembly_tests.builds.__wrapped__(tmp_path_factory)
    directory = tmp_path_factory.mktemp('decode-access') / 'build'
    builder.build(directory)
    return (*assembly_builds, directory)


@pytest.fixture(scope='session')
def extension(builds):
    directory = builds[2]
    build = runtime.verify_build(directory)
    spec = importlib.util.spec_from_file_location('_task1_decode_access', directory / build['binary'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_history(tmp_path, builds):
    args, prior = assembly_tests.make_history(tmp_path, builds[:2])
    origin = assembly_snapshot(tmp_path / 'accepted-M2-A1', prior['candidate_snapshots']['A1'], 'M2-A1', builds[1])
    text = sample() * 8  # Enough spaced actual special matches for 512 cuts.
    Path(prior['valid_file']).write_bytes(text.encode())
    ids = tokenizer()._encode_serial(text)
    benchmark.common.check_reference(ids, prior['reference'], write=True)
    history = {**prior, 'recommended': 'M2-A1', 'snapshot': origin, 'candidate_snapshots': {'M2-A1': origin},
               'measurements_valid': True, 'expected_tokens': len(ids)}
    history['input_sha256'].update({p: benchmark.common.sha256(p) for p in (prior['valid_file'], prior['reference'])})
    report, confirmation_path = tmp_path / 'assembly-selection.json', tmp_path / 'assembly-confirmation.json'
    report.write_text(json.dumps(history))
    confirmation = json.loads(Path(args.plain_confirmation).read_text())
    confirmation.update(method='CY-ED', assembly_case='M2-A1', token_count=len(ids),
                        bpe_source=str(Path(origin['directory']) / 'bpe.py'),
                        assembly={'method': 'M2', 'converted_ids': len(ids),
                                  'extension_path': str(Path(origin['directory']) / origin['assembly_build']['binary'])})
    confirmation['native_decode']['extension_path'] = str(Path(origin['directory']) / origin['native_build']['binary'])
    confirmation_path.write_text(json.dumps(confirmation))
    args = benchmark.parse_args(['--previous-selection', str(report), '--plain-confirmation', str(confirmation_path),
                                 '--build-dir', str(builds[2]), '--development', '--no-wandb', '--repeats', '1',
                                 '--output-dir', str(tmp_path / 'out')])
    return args, history


@pytest.mark.parametrize('indexed', [False, True])
def test_unicode_bytes_empty_and_mutations(extension, indexed):
    vocab = {0: b'\xe4', 1: b'\xb8\xad', 2: b'\xff', 3: b'\r\n', 4: b'', 5: b'<x>'}
    ids = [0, 1, 4, 2, 3, 5]
    stats = {}
    assert extension.decode(vocab, ids, indexed, stats) == serial(vocab, ids)
    assert stats['backend'] == 'native'
    assert stats['index_buffer_bytes'] == (len(ids) * 4 if indexed else 0)
    ids[0], vocab[1] = 5, b'changed'
    assert extension.decode(vocab, ids, indexed) == serial(vocab, ids)
    assert extension.decode(vocab, [], indexed) == ''
    # No requirement to have called encode or use its list/pool objects.
    assert extension.decode(vocab, list(reversed(ids)), indexed) == serial(vocab, list(reversed(ids)))
    stats = {}
    assert extension.profile_decode(vocab, ids, indexed, stats) == serial(vocab, ids)
    assert stats['index_write_included'] is indexed
    assert all(value >= 0 for key, value in stats.items() if key.endswith('_seconds'))


@pytest.mark.parametrize('indexed', [False, True])
@pytest.mark.parametrize('ids', [[-1], [999], [10**200], [0.5], [False, 1.0], [True], [0, []]])
def test_exception_and_nonordinary_integer_compatibility(extension, indexed, ids):
    vocab = {0: b'a', 1: b'b'}
    try:
        expected = serial(vocab, ids)
    except Exception as error:
        with pytest.raises(type(error)) as actual:
            extension.decode(vocab, ids, indexed)
        assert actual.value.args == error.args
    else:
        stats = {}
        assert extension.decode(vocab, ids, indexed, stats) == expected
        assert stats['backend'] == 'python-fallback'


@pytest.mark.parametrize('vocab,ids', [({4: b'x'}, [4]), ({False: b'a', 1: b'b'}, [0, 1]),
                                      ({0: bytearray(b'a')}, [0]), ({0: b'a'}, (0,))])
def test_sparse_and_compatibility_fallback(extension, vocab, ids):
    stats = {}
    assert extension.decode(vocab, ids, True, stats) == serial(vocab, ids)
    assert stats['backend'] == 'python-fallback'


@pytest.mark.parametrize('indexed', [False, True])
def test_refcounts_random_sequences_and_partial_failures(extension, indexed):
    vocab = {i: bytes([i]) for i in range(256)}
    vocab.update({i: b'ab' for i in range(256, 1001)})
    token = int('1000')
    ids = [token] * 50
    refs = sys.getrefcount(token)
    for _ in range(100):
        assert extension.decode(vocab, ids, indexed) == 'ab' * 50
        with pytest.raises(KeyError):
            extension.decode(vocab, [token, 9999], indexed)
    gc.collect()
    assert sys.getrefcount(token) == refs
    rng = random.Random(42)
    for _ in range(100):
        values = [rng.randrange(len(vocab)) for _ in range(rng.randrange(100))]
        assert extension.decode(vocab, values, indexed) == serial(vocab, values)
    with pytest.raises(MemoryError):
        extension._run(vocab, ids, indexed, allocation_limit=0)
    # Failure after table allocation, and after index allocation, releases all buffers.
    for limit in (16, 32, 64):
        with pytest.raises(MemoryError):
            extension._run({0: b'x' * 100, 1: b''}, [0] * 30, indexed, allocation_limit=limit)
    assert extension.decode(vocab, ids, indexed) == 'ab' * 50


def test_sizes_overflow(extension):
    assert extension._checked_size(3, 4) == 12
    for count, width in [(sys.maxsize, 4), (-1, 4), (1, 0)]:
        with pytest.raises(OverflowError):
            extension._checked_size(count, width)
    with pytest.raises(OverflowError):
        extension._checked_total(sys.maxsize, 1)


@pytest.mark.parametrize('case,parts,diagnostic', [('N0', 256, False), ('N1', 64, False), ('N2', 512, False),
                                                  ('N0', 128, True), ('N1', 256, True), ('N2', 256, True)])
def test_frozen_public_interface_and_real_workers(tmp_path, builds, case, parts, diagnostic):
    _, history = make_history(tmp_path, builds)
    snapshot = packaging.make_snapshot(tmp_path / 'candidate', history['snapshot'], case, parts, builds[2], diagnostic)
    job = {k: history[k] for k in ('tokenizer_prefix', 'valid_file', 'reference', 'expected_tokens')}
    job.update(snapshot=snapshot, method='CY-ED', diagnostic=diagnostic, cpus=[], workers=2,
               development=True, plain=False, active_wandb=False, input_sha256=history['input_sha256'])
    result = benchmark.native.launch_job(job, tmp_path / 'job.json', tmp_path / 'result.json', tmp_path / 'trial.log')
    assert result['status'] == 'ok' and result['parts'] == parts
    assert result['assembly']['method'] == 'M2' and result['assembly']['fragments'] == parts
    assert result['ids_match_reference'] and result['roundtrip_passed']
    metrics = result['native_decode']
    if case != 'N0':
        assert metrics['decoder_case'] == case
        assert metrics['index_buffer_bytes'] == (history['expected_tokens'] * 4 if case == 'N2' else 0)
    if diagnostic:
        assert metrics['diagnostic_implementation'] in ('assembly-profile', 'decode-access-profile')
    else:
        plain = benchmark.native.launch_job({**job, 'plain': True}, tmp_path / 'plain-job.json',
                                            tmp_path / 'plain.json', tmp_path / 'plain.log')
        assert plain['status'] == 'ok' and plain['patching'] is False and plain['parts'] == parts
    for build in ('native_build', 'assembly_build'):
        binary = snapshot[build]['binary']
        assert (Path(snapshot['directory']) / binary).read_bytes() == (Path(history['snapshot']['directory']) / binary).read_bytes()
    # Course template is copied byte-for-byte.
    assert (Path(snapshot['directory']) / 'train_bpe.py').read_bytes() == (Path(history['snapshot']['directory']) / 'train_bpe.py').read_bytes()


@pytest.mark.parametrize('best,promote,part_best,part_promote,expected', [
    ('N0', False, 256, False, 'N0'), ('N1', False, 256, False, 'N0'),
    ('N1', True, 64, True, 'N1-P64'), ('N2', True, 128, False, 'N2'), ('N2', True, 512, True, 'N2-P512')])
def test_two_stage_gates(best, promote, part_best, part_promote, expected):
    rows = []
    def trial(stage, case, repeat=0, diagnostic=False):
        base = case.split('-')[0]
        total = {'N0': 10, 'N1': 11, 'N2': 12}[base]
        if base == best:
            total = 8
        if stage.startswith('decode-paired') and not promote:
            total = 7 if base == 'N0' else 8
        if stage.startswith('partition'):
            total = 6 if case == benchmark.case_key(base, part_best) else 8
            if stage.startswith('partition-paired') and not part_promote:
                total = 5 if '-P' not in case else 6
        row = {'trial_id': str(len(rows)), 'repeat': repeat, 'diagnostic': diagnostic,
               'status': 'ok', 'total_seconds': total, 'ids_match_reference': True, 'roundtrip_passed': True}
        rows.append((stage, case))
        return row
    selected = benchmark.run_stages(SimpleNamespace(seed=42, repeats=5, diagnostics=True), trial,
                                    {str(p): True for p in packaging.PARTS})
    assert selected['recommended'] == expected
    assert any(stage.startswith('index-buffer-mechanism') for stage, _ in rows) == (best == 'N2' and promote)


def test_unavailable_partitions_are_excluded():
    calls = []
    def trial(stage, case, repeat=0, diagnostic=False):
        calls.append(case)
        return {'status': 'ok', 'total_seconds': 10 if case == 'N0' else 11, 'repeat': repeat,
                'trial_id': str(len(calls)), 'diagnostic': diagnostic, 'ids_match_reference': True, 'roundtrip_passed': True}
    args = SimpleNamespace(seed=42, repeats=5, diagnostics=True)
    result = benchmark.run_stages(args, trial, {'64': True, '128': False, '256': True, '512': False})
    assert not any('-P128' in case or '-P512' in case for case in calls)
    assert set(result['unavailable_partitions']) == {'128', '512'}
    with pytest.raises(ValueError, match='256'):
        benchmark.run_stages(args, trial, {'256': False})


@pytest.mark.parametrize('failure', ['diagnostic', 'confirmation', 'cpu', 'binary'])
def test_provenance_guards(tmp_path, builds, failure):
    args, history = make_history(tmp_path, builds)
    if failure == 'diagnostic':
        history['snapshot']['assembly_diagnostic'] = True
        Path(args.previous_selection).write_text(json.dumps(history))
    elif failure == 'confirmation':
        path = Path(args.plain_confirmation)
        value = json.loads(path.read_text())
        value['assembly_case'] = 'M1-A1'
        path.write_text(json.dumps(value))
    elif failure == 'cpu':
        args.cpu_ids = [0]
    else:
        (Path(history['snapshot']['directory']) / history['snapshot']['native_build']['binary']).write_bytes(b'changed')
    with pytest.raises((ValueError, RuntimeError)):
        benchmark.load_history(args)


def test_build_abi_and_fingerprint(tmp_path, builds):
    import shutil
    directory = tmp_path / 'build'
    shutil.copytree(builds[2], directory)
    path = directory / 'build.json'
    value = json.loads(path.read_text())
    value['gil_disabled'] = True
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match='ABI'):
        runtime.verify_build(directory)
    value['gil_disabled'] = False
    path.write_text(json.dumps(value))
    (directory / value['binary']).write_bytes(b'wrong')
    with pytest.raises(RuntimeError, match='changed'):
        runtime.verify_build(directory)


def test_complete_toy_exploration_and_confirmation(tmp_path, builds):
    args, _ = make_history(tmp_path, builds)
    args.diagnostics = True
    benchmark.exploration(args)
    out = Path(args.output_dir)
    selection = json.loads((out / 'selection.json').read_text())
    assert selection['completed'] and selection['measurements_valid'] and selection['upload_status'] == 'disabled'
    assert selection['history_context']['context_regression_flag']
    rows = json.loads((out / 'results.json').read_text())
    assert all(row['ids_match_reference'] and row['assembly']['fragments'] == row['parts'] for row in rows)
    assert json.loads((out / 'integrity.json').read_text()) == {'changed_inputs': [], 'source_changed': False}
    # Toy timing is not a speed claim; integration arithmetic has separate tests.
    selection['ready_for_plain_confirmation'] = True
    (out / 'selection.json').write_text(json.dumps(selection))
    args = benchmark.parse_args(['--phase', 'plain-confirmation', '--selection-report', str(out / 'selection.json'),
                                 '--output-dir', str(tmp_path / 'plain'), '--development', '--no-wandb'])
    benchmark.plain_confirmation(args)
    result = json.loads((tmp_path / 'plain/confirmation.json').read_text())
    assert result['status'] == 'ok' and result['decode_access_recommended'] == selection['recommended']


def test_upload_mapping_summary_and_finish_failure(tmp_path):
    (tmp_path / 'manifest.json').write_text(json.dumps({'origin_snapshots': {}}))
    class Summary:
        def update(self, value):
            assert isinstance(value, dict)
    class Run:
        id, summary = 'access-test', Summary()
        def log(self, value):
            pass
        def log_artifact(self, value):
            pass
        def save(self, *args, **kwargs):
            pass
    selected = {'completed': True, 'measurements_valid': True, 'recommended': 'N0', 'candidate_snapshots': {},
                'integration': {'passed': True}}
    benchmark.publish(Run(), tmp_path, [], selected)
    @contextmanager
    def failure():
        yield Run()
        raise RuntimeError('Upload failure after computation')
    with benchmark.native.recording_context(failure(), tmp_path, selected):
        pass
    assert selected['completed'] and selected['measurements_valid'] and selected['upload_status'] == 'failed'


def test_debug_allocator_errors_and_ref_ownership(builds):
    code = f'''
import sys
sys.path.insert(0,{str(builds[2])!r})
import _task1_decode_access as native
vocab={{i: bytes([i]) for i in range(256)}}
for indexed in (False,True):
 for _ in range(1000):
  try:native.decode(vocab,[65,10**200],indexed)
  except KeyError:pass
  else:raise AssertionError('Bad ID accepted')
  try:native._run(vocab,[65]*100,indexed,allocation_limit=10)
  except MemoryError:pass
  else:raise AssertionError('Injected allocation accepted')
 assert native.decode(vocab,[65,66],indexed)=='AB'
'''
    subprocess.run([sys.executable, '-c', code], check=True, env={**os.environ, 'PYTHONMALLOC': 'debug'}, timeout=15)


@pytest.mark.parametrize('signal_name', ['SIGINT', 'SIGTERM'])
def test_new_path_interrupt_reclaims_workers(tmp_path, builds, signal_name):
    import signal
    import time
    import psutil
    _, history = make_history(tmp_path, builds)
    snapshot = packaging.make_snapshot(tmp_path / 'candidate', history['snapshot'], 'N2', 256, builds[2])
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
 time.sleep(30)
transport.await_ready=wait
bpe=BPE();bpe.load({history['tokenizer_prefix']!r})
bpe.encode({(sample()*8)!r})
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
            assert process.returncode != 0 and all(not psutil.pid_exists(pid) for pid in children)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


def test_new_path_worker_failure_and_streaming_reload(tmp_path, builds):
    _, history = make_history(tmp_path, builds)
    snapshot = packaging.make_snapshot(tmp_path / 'candidate', history['snapshot'], 'N1', 64, builds[2])
    code = f'''
import sys
sys.path.insert(0,{snapshot['directory']!r})
from bpe import BPE
import task1_transport_runtime as transport
children=[]
original=transport.subprocess.Popen
def launch(*args,**kwargs):
 process=original(*args,**kwargs);children.append(process);return process
transport.subprocess.Popen=launch
bpe=BPE();bpe.load({history['tokenizer_prefix']!r})
text={(sample()*8)!r}
ids=bpe.encode(text)
assert list(bpe.encode_iterable([text[i:i+3] for i in range(0,len(text),3)]))==ids
bpe.save({str(tmp_path / 'saved')!r});bpe.load({str(tmp_path / 'saved')!r})
assert not bpe._cache and bpe.encode(text)==ids and bpe.decode(ids)==text
bpe.vocab[0]=object()
try:bpe.encode(text)
except RuntimeError:
 assert children and all(process.poll() is not None for process in children)
else:raise AssertionError('Partial result after worker failure')
'''
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, timeout=15)


def test_active_offline_confirmation_uses_new_decoder(tmp_path, builds, monkeypatch):
    _, history = make_history(tmp_path, builds)
    snapshot = packaging.make_snapshot(tmp_path / 'candidate', history['snapshot'], 'N2', 128, builds[2])
    for key, value in [('WANDB_MODE', 'offline'), ('WANDB_ENTITY', 'test'), ('STUDENT_ID', 'decode-access-test')]:
        monkeypatch.setenv(key, value)
    job = {k: history[k] for k in ('tokenizer_prefix', 'valid_file', 'reference', 'expected_tokens')}
    job.update(snapshot=snapshot, method='CY-ED', diagnostic=False, cpus=[], workers=2,
               development=True, plain=True, active_wandb=True, input_sha256=history['input_sha256'])
    result = benchmark.native.launch_job(job, tmp_path / 'job.json', tmp_path / 'result.json', tmp_path / 'trial.log')
    assert result['status'] == 'ok' and result['active_wandb'] and result['patching'] is False
    assert result['native_decode']['decoder_case'] == 'N2' and result['parts'] == 128


def test_preflight_failure_is_recorded(tmp_path, builds, monkeypatch):
    args, _ = make_history(tmp_path, builds)
    def fail(*args):
        raise RuntimeError('Injected split preflight failure')
    monkeypatch.setattr(benchmark.native, 'launch_job', fail)
    with pytest.raises(RuntimeError, match='preflight'):
        benchmark.exploration(args)
    selection = json.loads((Path(args.output_dir) / 'selection.json').read_text())
    assert selection['completed'] is False and selection['measurements_valid'] is False
