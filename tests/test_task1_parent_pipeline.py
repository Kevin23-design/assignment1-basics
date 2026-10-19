"""Small public-interface, pooled ownership, dispatch ordering and real W&B schema tests."""
from array import array
from contextlib import contextmanager
import gc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
from types import SimpleNamespace

import pytest

import benchmark_task1_parent_pipeline as benchmark
import benchmark_task1_decode_access as previous
import build_task1_parent_assembly as builder
import task1_parent_pipeline_experiment as packaging
import task1_parent_pipeline_runtime as runtime
from task1_benchmark_reporting import table_rows
from task1_decode_access_experiment import make_snapshot as decode_snapshot
from tests import test_task1_decode_access as decode_tests

pytestmark = pytest.mark.skipif(
    sys.version_info[:2] != (3, 13) or bool(sysconfig.get_config_var('Py_GIL_DISABLED')),
    reason='Parent native assembly requires GIL-enabled CPython 3.13',
)


@pytest.fixture(scope='session')
def builds(tmp_path_factory):
    old = decode_tests.builds.__wrapped__(tmp_path_factory)
    directory = tmp_path_factory.mktemp('parent-assembly') / 'build'
    builder.build(directory)
    return (*old, directory)


@pytest.fixture(scope='session')
def extension(builds):
    directory = builds[3]
    build = runtime.verify_build(directory)
    spec = importlib.util.spec_from_file_location('_task1_parent_assembly', directory / build['binary'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_history(tmp_path, builds):
    old_args, prior = decode_tests.make_history(tmp_path, builds[:3])
    origin = decode_snapshot(tmp_path / 'accepted-N2', prior['snapshot'], 'N2', 256, builds[2])
    history = {**prior, 'recommended': 'N2', 'snapshot': origin, 'candidate_snapshots': {'N2': origin},
               'upload_status': 'failed', 'upload_error': 'Historical Table schema mismatch'}
    report, confirmation_path = tmp_path / 'decode-selection.json', tmp_path / 'decode-confirmation.json'
    report.write_text(json.dumps(history))
    result = json.loads(Path(old_args.plain_confirmation).read_text())
    result.update(decode_access_case='N2', parts=256,
                  bpe_source=str(Path(origin['directory']) / 'bpe.py'),
                  native_decode={'backend': 'native', 'decoder_case': 'N2',
                                 'index_buffer_bytes': history['expected_tokens'] * 4,
                                 'extension_path': str(Path(origin['directory']) / origin['decode_access_build']['binary'])})
    result['assembly']['extension_path'] = str(Path(origin['directory']) / origin['assembly_build']['binary'])
    confirmation_path.write_text(json.dumps(result))
    args = benchmark.parse_args(['--previous-selection', str(report), '--plain-confirmation', str(confirmation_path),
                                 '--build-dir', str(builds[3]), '--development', '--no-wandb', '--repeats', '1',
                                 '--output-dir', str(tmp_path / 'out')])
    return args, history


def test_append_and_deferred_pool_ownership(extension):
    pool = extension.Pool(32000)
    result = []
    assert pool.append(result, array('I', [10000, 10000])) == 2
    chunks = [array('I', [10000]), array('I'), array('I', [31999, 10000])]
    assembled = pool.assemble_owned(chunks)
    assert chunks == [None, None, None]
    assert assembled == [10000, 31999, 10000]
    assert result[0] is result[1] is assembled[0] is assembled[2]
    value = result[0]
    before = sys.getrefcount(value)
    for _ in range(100):
        with pytest.raises(ValueError):
            pool.assemble_owned([array('I', [10000, 32000])])
        target = []
        with pytest.raises(ValueError):
            pool.append(target, array('I', [10000, 32000]))
        del target
    gc.collect()
    assert sys.getrefcount(value) == before
    del pool
    gc.collect()
    assert assembled == [10000, 31999, 10000]


def test_formats_empty_bounds_and_allocation(extension):
    pool = extension.Pool(3)
    assert pool.append([], array('I')) == 0
    assert pool.assemble_owned([]) == []
    for invalid in (None, array('i', [1]), array('Q', [1]), b'1234', memoryview(array('I', [1]))):
        with pytest.raises(TypeError):
            pool.append([], invalid)
        with pytest.raises(TypeError):
            pool.assemble_owned([invalid])
    with pytest.raises(TypeError):
        pool.append((), array('I'))
    with pytest.raises(TypeError):
        pool.assemble_owned(())
    with pytest.raises(ValueError):
        pool.append([], array('I', [3]))
    with pytest.raises(MemoryError):
        pool.assemble_owned([array('I', [0])], allocation_limit=0)
    for count, more in [(sys.maxsize, 1), (-1, 0), (0, -1), (sys.maxsize, 0)]:
        with pytest.raises(OverflowError):
            extension._checked_count(count, more)
    assert extension._checked_count(3, 4) == 7
    assert pool.assemble_owned([array('I', [0, 1])]) == [0, 1]


@pytest.mark.parametrize('case', ['Q2', 'Q3'])
def test_out_of_order_missing_duplicate_and_unknown(extension, monkeypatch, case):
    monkeypatch.setattr(runtime, 'load_parent', lambda: extension)
    collector = runtime.Collector({i: bytes([i]) for i in range(256)}, 3, case)
    collector.accept(2, array('I', [3]))
    collector.accept(0, array('I'))
    with pytest.raises(RuntimeError):
        collector.finish()
    with pytest.raises(RuntimeError):
        collector.accept(0, array('I'))
    for index, values in [(True, array('I')), (3, array('I')), (1, array('i', [1]))]:
        with pytest.raises(RuntimeError):
            collector.accept(index, values)
    collector.accept(1, array('I', [1, 2]))
    assert collector.finish() == [1, 2, 3]
    assert collector.metrics['converted_ids'] == 3 and collector.metrics['temporary_id_lists'] == 0
    assert collector.metrics['retained_raw_bytes_peak'] >= 12
    bad = runtime.Collector({0: b'a'}, 1, case)
    with pytest.raises(ValueError):
        bad.accept(0, array('I', [99]))
        bad.finish()
    if case == 'Q3':
        assert bad.chunks is None


@pytest.mark.parametrize('case', packaging.CASES)
@pytest.mark.parametrize('diagnostic', [False, True])
def test_real_public_interface_and_frozen_artifacts(tmp_path, builds, case, diagnostic):
    _, history = make_history(tmp_path, builds)
    origin = history['snapshot']
    snapshot = packaging.make_snapshot(tmp_path / 'candidate', origin, case, builds[3], diagnostic)
    job = {k: history[k] for k in ('tokenizer_prefix', 'valid_file', 'reference', 'expected_tokens')}
    job.update(snapshot=snapshot, method='CY-ED', diagnostic=diagnostic, cpus=[], workers=2,
               development=True, plain=False, active_wandb=False, input_sha256=history['input_sha256'])
    result = benchmark.native.launch_job(job, tmp_path / 'job.json', tmp_path / 'result.json', tmp_path / 'trial.log')
    assert result['status'] == 'ok' and result['ids_match_reference'] and result['roundtrip_passed']
    assert result['parts'] == 256 and result['assembly']['fragments'] == 256
    assert result['native_decode']['decoder_case'] == 'N2'
    if case in ('Q2', 'Q3'):
        assert result['assembly']['method'] == case and result['assembly']['temporary_id_lists'] == 0
    if diagnostic:
        assert result['transport']['finish_assembly_seconds'] >= 0
    else:
        plain = benchmark.native.launch_job({**job, 'plain': True}, tmp_path / 'plain-job.json',
                                            tmp_path / 'plain.json', tmp_path / 'plain.log')
        assert plain['status'] == 'ok' and plain['patching'] is False and plain['parent_pipeline_case'] == case
    for key in ('native_build', 'assembly_build', 'decode_access_build'):
        assert snapshot[key]['files'] == origin[key]['files']
        assert (Path(snapshot['directory']) / snapshot[key]['binary']).read_bytes() == (
            Path(origin['directory']) / origin[key]['binary']).read_bytes()
    for name in ('bpe.py', 'train_bpe.py'):
        assert (Path(snapshot['directory']) / name).read_bytes() == (Path(origin['directory']) / name).read_bytes()
    if case in ('Q0', 'Q1') and not diagnostic:
        assert (Path(snapshot['directory']) / 'task1_pipeline_runtime.py').read_bytes() == (
            Path(origin['directory']) / 'task1_pipeline_runtime.py').read_bytes()


def test_dispatch_precedes_slow_consumption(tmp_path, builds):
    _, history = make_history(tmp_path, builds)
    origin = history['snapshot']
    snapshots = {case: packaging.make_snapshot(tmp_path / case, origin, case, builds[3]) for case in ('Q0', 'Q1')}
    for case, snapshot in snapshots.items():
        code = f'''
import sys
sys.path.insert(0,{snapshot['directory']!r})
import task1_transport_runtime as t
events=[]
class Connection:
 def send(self,message):events.append('send-'+message[0])
c=Connection()
def messages(active,processes,stats,diagnostic):
 yield c,('part',0,None)
 yield c,('part',1,None)
 yield c,('done',0,{{}})
t.timed_messages=messages
t.drive_pipes([c],[],{{'diagnostic':False,'ranges':[(0,1),(1,2)],'text':'ab'}},
              lambda *args:events.append('assemble'),{{}})
expected={(['send-job','assemble','send-job','assemble','send-stop'] if case=='Q0' else ['send-job','send-job','assemble','send-stop','assemble'])!r}
assert events==expected,(events,expected)
'''
        subprocess.run([sys.executable, '-c', code], check=True, timeout=10)


@pytest.mark.parametrize('best,promote,expected', [('Q0', False, 'Q0'), ('Q1', True, 'Q1'),
                                               ('Q2', False, 'Q0'), ('Q2', True, 'Q2'), ('Q3', True, 'Q3')])
def test_gate_and_explanatory_comparison(best, promote, expected):
    calls = []
    def trial(stage, case, repeat=0, diagnostic=False):
        seconds = 8 if case == best else 10
        if stage.startswith('paired') and not promote:
            seconds = 7 if case == 'Q0' else 8
        calls.append((stage, case))
        return {'status': 'ok', 'total_seconds': seconds, 'repeat': repeat, 'diagnostic': diagnostic,
                'ids_match_reference': True, 'roundtrip_passed': True, 'trial_id': str(len(calls))}
    result = benchmark.run_stages(SimpleNamespace(seed=42, repeats=5, diagnostics=True), trial)
    assert result['recommended'] == expected
    assert any(stage.startswith('assembly-mechanism') for stage, _ in calls) == (expected in ('Q2', 'Q3'))


def test_heterogeneous_table_cells_and_old_publisher_regression(tmp_path):
    import wandb
    rows = [{'trial_id': 'a', 'diagnostic': True, 'native_decode': {'backend': 'native', 'validate_length_seconds': 1.2},
             'system_load_before': {'average': [1.0, 2.0]}, 'cpu_affinity': [0, 1]},
            {'trial_id': 'b', 'diagnostic': False, 'native_decode': {'decoder_case': 'N2', 'index_buffer_bytes': 32},
             'system_load_before': {'utc': 'now', 'other': True}, 'cpu_affinity': []},
            {'trial_id': 'c', 'diagnostic': True, 'native_decode': {'extra': {'nested': [False, 1]}}}]
    columns = ['trial_id', 'diagnostic', 'native_decode', 'system_load_before', 'cpu_affinity']
    table = wandb.Table(columns=columns, data=table_rows(columns, rows))
    assert all(isinstance(item, str) for item in table.get_column('native_decode'))
    assert [json.loads(item) for item in table.get_column('native_decode')] == [r['native_decode'] for r in rows]
    (tmp_path / 'manifest.json').write_text(json.dumps({'origin_snapshots': {}}))
    class Summary:
        def update(self, value):
            assert isinstance(value, dict)
    class Run:
        id, summary = 'parent-test', Summary()
        def log(self, value):
            assert all(isinstance(cell, str) for cell in next(iter(value.values())).get_column('native_decode'))
        def log_artifact(self, value):
            pass
        def save(self, *args, **kwargs):
            pass
    selection = {'completed': True, 'measurements_valid': True, 'recommended': 'Q0', 'candidate_snapshots': {},
                 'integration': {'passed': True}}
    benchmark.publish(Run(), tmp_path, rows, selection)
    previous.publish(Run(), tmp_path, rows, selection)
    @contextmanager
    def failed_finish():
        yield Run()
        raise RuntimeError('Upload failure after valid computation')
    with benchmark.native.recording_context(failed_finish(), tmp_path, selection):
        pass
    assert selection['measurements_valid'] and selection['upload_status'] == 'failed'


def test_toy_full_exploration_and_plain(tmp_path, builds):
    args, _ = make_history(tmp_path, builds)
    args.diagnostics = True
    benchmark.exploration(args)
    out = Path(args.output_dir)
    selected = json.loads((out / 'selection.json').read_text())
    assert selected['completed'] and selected['measurements_valid'] and selected['upload_status'] == 'disabled'
    assert selected['history_context']['upload_status'] == 'failed'
    rows = json.loads((out / 'results.json').read_text())
    assert all(row['parts'] == 256 and row['ids_match_reference'] and row['native_decode']['decoder_case'] == 'N2' for row in rows)
    selected['ready_for_plain_confirmation'] = True  # Toy latency is not a speed claim; arithmetic tested above.
    (out / 'selection.json').write_text(json.dumps(selected))
    args = benchmark.parse_args(['--phase', 'plain-confirmation', '--selection-report', str(out / 'selection.json'),
                                 '--development', '--no-wandb', '--output-dir', str(tmp_path / 'plain')])
    benchmark.plain_confirmation(args)
    result = json.loads((tmp_path / 'plain/confirmation.json').read_text())
    assert result['status'] == 'ok' and result['parent_pipeline_recommended'] == selected['recommended']


@pytest.mark.parametrize('failure', ['diagnostic', 'confirmation', 'binary', 'cpu'])
def test_history_guards(tmp_path, builds, failure):
    args, history = make_history(tmp_path, builds)
    if failure == 'diagnostic':
        history['snapshot']['decode_access_diagnostic'] = True
        Path(args.previous_selection).write_text(json.dumps(history))
    elif failure == 'confirmation':
        path = Path(args.plain_confirmation)
        data = json.loads(path.read_text())
        data['parts'] = 128
        path.write_text(json.dumps(data))
    elif failure == 'binary':
        (Path(history['snapshot']['directory']) / history['snapshot']['decode_access_build']['binary']).write_bytes(b'changed')
    else:
        args.cpu_ids = [0]
    with pytest.raises((ValueError, RuntimeError)):
        benchmark.load_history(args)


def test_debug_allocator_partial_lists(builds):
    code = f'''
import sys,gc
from array import array
sys.path.insert(0,{str(builds[3])!r})
import _task1_parent_assembly as n
pool=n.Pool(32000)
for _ in range(1000):
 try:pool.assemble_owned([array('I',[10000,32000])])
 except ValueError:pass
 else:raise AssertionError('Unknown ID accepted')
 try:pool.assemble_owned([array('I',[1])],allocation_limit=0)
 except MemoryError:pass
 else:raise AssertionError('Allocation limit ignored')
 assert pool.assemble_owned([array('I',[10000,10000])])==[10000,10000]
'''
    subprocess.run([sys.executable, '-c', code], check=True, env={**os.environ, 'PYTHONMALLOC': 'debug'}, timeout=15)


@pytest.mark.parametrize('case', ['Q2', 'Q3'])
def test_accept_failure_stops_dispatched_workers(tmp_path, builds, case):
    _, history = make_history(tmp_path, builds)
    snapshot = packaging.make_snapshot(tmp_path / 'candidate', history['snapshot'], case, builds[3])
    code = f'''
import sys
sys.path.insert(0,{snapshot['directory']!r})
from bpe import BPE
import task1_transport_runtime as t
import task1_parent_pipeline_runtime as r
children=[]
original=t.subprocess.Popen
def launch(*args,**kwargs):
 child=original(*args,**kwargs);children.append(child);return child
t.subprocess.Popen=launch
def fail(*args):raise ValueError('Injected parent accept failure after next dispatch')
r.Collector.accept=fail
bpe=BPE();bpe.load({history['tokenizer_prefix']!r})
text=open({history['valid_file']!r},encoding='utf-8',newline='').read()
try:bpe.encode(text)
except ValueError:
 assert children and all(child.poll() is not None for child in children)
else:raise AssertionError('Partial result returned')
'''
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, timeout=15)


@pytest.mark.parametrize('signal_name', ['SIGINT', 'SIGTERM'])
def test_parent_interrupt_cleanup(tmp_path, builds, signal_name):
    import signal
    import time
    import psutil
    _, history = make_history(tmp_path, builds)
    snapshot = packaging.make_snapshot(tmp_path / 'candidate', history['snapshot'], 'Q3', builds[3])
    ready = tmp_path / 'ready.json'
    code = f'''
import sys,json,time
from pathlib import Path
sys.path.insert(0,{snapshot['directory']!r})
from bpe import BPE
import task1_transport_runtime as t
original=t.await_ready
def wait(*args):
 original(*args)
 Path({str(ready)!r}).write_text(json.dumps([p.pid for p in args[1]]))
 time.sleep(30)
t.await_ready=wait
bpe=BPE();bpe.load({history['tokenizer_prefix']!r})
bpe.encode(open({history['valid_file']!r},encoding='utf-8',newline='').read())
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


def test_active_offline_plain_new_assembly(tmp_path, builds, monkeypatch):
    _, history = make_history(tmp_path, builds)
    snapshot = packaging.make_snapshot(tmp_path / 'candidate', history['snapshot'], 'Q3', builds[3])
    for key, value in [('WANDB_MODE', 'offline'), ('WANDB_ENTITY', 'test'), ('STUDENT_ID', 'parent-test')]:
        monkeypatch.setenv(key, value)
    job = {k: history[k] for k in ('tokenizer_prefix', 'valid_file', 'reference', 'expected_tokens')}
    job.update(snapshot=snapshot, method='CY-ED', diagnostic=False, cpus=[], workers=2,
               development=True, plain=True, active_wandb=True, input_sha256=history['input_sha256'])
    result = benchmark.native.launch_job(job, tmp_path / 'job.json', tmp_path / 'result.json', tmp_path / 'trial.log')
    assert result['status'] == 'ok' and result['active_wandb'] and result['patching'] is False
    assert result['assembly']['method'] == 'Q3' and result['native_decode']['decoder_case'] == 'N2'


def test_parent_abi_fingerprints_and_new_path_validation(tmp_path, builds):
    import shutil
    directory = tmp_path / 'build'
    shutil.copytree(builds[3], directory)
    path = directory / 'build.json'
    data = json.loads(path.read_text())
    data['soabi'] = 'different-interpreter'
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match='ABI'):
        runtime.verify_build(directory)
    data['soabi'] = sysconfig.get_config_var('SOABI')
    path.write_text(json.dumps(data))
    (directory / data['binary']).write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='changed'):
        runtime.verify_build(directory)


def test_no_finish_after_failed_append(extension, monkeypatch):
    monkeypatch.setattr(runtime, 'load_parent', lambda: extension)
    collector = runtime.Collector({0: b'a'}, 1, 'Q2')
    with pytest.raises(ValueError):
        collector.accept(0, array('I', [0, 99]))
    with pytest.raises(RuntimeError, match='Incomplete'):
        collector.finish()
