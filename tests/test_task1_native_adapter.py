"""Loader-only native exploration, using tiny reusable native fixtures."""
import ast
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import benchmark_task1_native_adapter as benchmark
import build_task1_native as builder
import task1_native_adapter_experiment as packaging
import task1_native_experiment as previous_packaging
from tests.test_task1_encode_transport import history_fixture, sample


@pytest.fixture(scope='session')
def native_build(tmp_path_factory):
    pytest.importorskip('Cython')
    directory = tmp_path_factory.mktemp('adapter-native') / 'build'
    builder.build(directory)
    return directory


def adapter_history(tmp_path, native_build):
    _, original = history_fixture(tmp_path)
    origins = {method: previous_packaging.make_snapshot(tmp_path / ('prior-' + method), original['snapshot'],
                                                       method, native_build) for method in ('CY-D', 'CY-ED')}
    history = {**original, 'recommended': 'CY-D', 'measurements_valid': True,
               'candidate_snapshots': origins, 'snapshot': origins['CY-D']}
    report, confirmation = tmp_path / 'native-selection.json', tmp_path / 'native-confirmation.json'
    report.write_text(json.dumps(history))
    confirmation.write_text(json.dumps({
        'status': 'ok', 'method': 'CY-D', 'patching': False, 'source_changed': False,
        'roundtrip_passed': True, 'ids_match_reference': True, 'cpu_affinity': [],
        'active_wandb': False, 'token_count': history['expected_tokens'],
        'bpe_source': str(Path(origins['CY-D']['directory']) / 'bpe.py'), 'total_seconds': 10.7,
        'native_decode': {'backend': 'native', 'extension_path': str(
            Path(origins['CY-D']['directory']) / origins['CY-D']['native_build']['binary'])},
        'context_regression_flag': True, 'context_ratio_to_plain_median': 1.13}))
    args = benchmark.parse_args(['--previous-selection', str(report), '--plain-confirmation', str(confirmation),
                                 '--development', '--no-wandb', '--repeats', '1', '--output-dir', str(tmp_path / 'out')])
    return args, history


def loader_namespace(tmp_path, wrong_path=False):
    expected = tmp_path / 'test.so'
    module = SimpleNamespace(__file__=str(tmp_path / 'wrong.so' if wrong_path else expected))
    calls = []
    class Loader:
        def exec_module(self, value):
            calls.append('execute')
    util = SimpleNamespace(spec_from_file_location=lambda *args: SimpleNamespace(loader=Loader()),
                           module_from_spec=lambda _: module)
    def validate(*args, **kwargs):
        calls.append('validate')
        return {'binary': expected.name}
    ns = {'_loaded': None, '_DIRECTORY': tmp_path, 'Path': Path, 'verify_build': validate,
          'importlib': SimpleNamespace(util=util)}
    exec(packaging.FAST_LOADER, ns)
    return ns, module, calls


def test_cached_loader_does_no_path_metadata_or_hash_work(tmp_path):
    ns, module, calls = loader_namespace(tmp_path)
    assert ns['load_native']() is module
    assert calls == ['validate', 'execute']
    def forbidden(*args, **kwargs):
        raise AssertionError('Cached load must not resolve/read/hash/validate')
    ns['verify_build'], ns['Path'], ns['_DIRECTORY'] = forbidden, forbidden, object()
    for _ in range(100):
        assert ns['load_native']() is module
    assert calls == ['validate', 'execute']


@pytest.mark.parametrize('failure', ['path', 'verify', 'execute'])
def test_failed_first_load_does_not_cache_module(tmp_path, failure):
    ns, module, calls = loader_namespace(tmp_path, wrong_path=failure == 'path')
    if failure == 'verify':
        def fail(*args, **kwargs):
            raise RuntimeError('bad ABI or binary fingerprint')
        ns['verify_build'] = fail
    if failure == 'execute':
        def fail(*args, **kwargs):
            raise ImportError('injected extension import error')
        ns['importlib'].util.module_from_spec = fail
    with pytest.raises((RuntimeError, ImportError)):
        ns['load_native']()
    assert ns['_loaded'] is None
    if failure == 'path':
        module.__file__ = str(tmp_path / 'test.so')
        assert ns['load_native']() is module
        assert calls.count('validate') == 2


def test_normal_candidate_changes_only_loader(tmp_path, native_build):
    _, history = adapter_history(tmp_path, native_build)
    origin = history['candidate_snapshots']['CY-ED']
    fixed = packaging.make_snapshot(tmp_path / 'fixed', origin, 'A1')
    changed = [name for name, digest in fixed['source_sha256'].items() if origin['source_sha256'].get(name) != digest]
    assert set(changed) == {'native_build.json', 'task1_native_runtime.py'}
    def without_loader(path):
        return [ast.dump(node) for node in ast.parse(path.read_text()).body
                if getattr(node, 'name', None) != 'load_native']
    assert without_loader(Path(fixed['directory']) / 'task1_native_runtime.py') == without_loader(
        Path(origin['directory']) / 'task1_native_runtime.py')
    packaging.assert_identical_native([origin, fixed, history['candidate_snapshots']['CY-D']])
    assert 'adapter_loader_metrics' not in (Path(fixed['directory']) / 'task1_native_runtime.py').read_text()


@pytest.mark.parametrize('case', ['A0', 'OLD-CY-ED', 'A1'])
def test_diagnostic_counts_real_calls_and_resolutions(tmp_path, native_build, case):
    _, history = adapter_history(tmp_path, native_build)
    origin = history['candidate_snapshots']['CY-D' if case == 'A0' else 'CY-ED']
    snapshot = packaging.make_snapshot(tmp_path / 'diagnostic', origin, case, diagnostic=True)
    spec = importlib.util.spec_from_file_location('adapter_runtime_test', Path(snapshot['directory']) / 'task1_native_runtime.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.load_native()
    first_resolutions = module._adapter_loader_metrics['path_resolve_calls']
    module.load_native()
    module.load_native()
    metrics = module._adapter_loader_metrics
    assert metrics['calls'] == 3 and metrics['validation_calls'] == 1
    assert metrics['load_seconds'] >= 0 and metrics['validation_seconds'] >= 0
    if case == 'A1':
        assert metrics['path_resolve_calls'] == first_resolutions
    else:
        assert metrics['path_resolve_calls'] == first_resolutions + 4


@pytest.mark.parametrize('case,diagnostic', [('A0', False), ('A1', False), ('A0', True),
                                           ('A1', True), ('OLD-CY-ED', True)])
def test_real_interface_and_independent_diagnostic_snapshots(tmp_path, native_build, case, diagnostic):
    args, history = adapter_history(tmp_path, native_build)
    origin = history['candidate_snapshots']['CY-D' if case == 'A0' else 'CY-ED']
    snapshot = packaging.make_snapshot(tmp_path / 'candidate', origin, case, diagnostic=diagnostic)
    job = {k: history[k] for k in ('tokenizer_prefix', 'valid_file', 'reference', 'expected_tokens')}
    job.update(snapshot=snapshot, method=snapshot['method'], diagnostic=diagnostic, cpus=[], workers=2,
               development=True, plain=False, active_wandb=False, input_sha256=history['input_sha256'])
    result = benchmark.native.launch_job(job, tmp_path / 'job.json', tmp_path / 'result.json', tmp_path / 'trial.log')
    assert result['status'] == 'ok' and result['ids_match_reference'] and result['roundtrip_passed']
    assert result['native_decode']['backend'] == 'native'
    if diagnostic:
        assert result['native_decode']['loader']['calls'] == 1
        if case != 'A0':
            for worker in result['worker_metadata']:
                assert worker['native_loading']['calls'] == worker['native_encoding']['native_calls']
                assert worker['native_loading']['validation_calls'] == 1
    else:
        assert 'loader' not in result['native_decode']
        plain = benchmark.native.launch_job({**job, 'plain': True}, tmp_path / 'plain-job.json',
                                            tmp_path / 'plain.json', tmp_path / 'plain.log')
        assert plain['status'] == 'ok' and plain['patching'] is False


def test_fixed_runtime_random_merges_cache_streaming_and_reload(tmp_path, native_build):
    _, history = adapter_history(tmp_path, native_build)
    fixed = packaging.make_snapshot(tmp_path / 'candidate', history['candidate_snapshots']['CY-ED'], 'A1')
    code = f'''
import sys,random
sys.path.insert(0,{fixed['directory']!r})
from bpe import BPE
bpe=BPE()
bpe.load({history['tokenizer_prefix']!r})
rng=random.Random(42)
for _ in range(100):
    text=''.join(rng.choice('ab 中🙂\\r\\n\\0') for _ in range(rng.randrange(60)))
    assert bpe._encode_uncached(text, 2) == bpe._python_encode_uncached(text, 2)
text={sample()!r}
ids=bpe._encode_serial(text)
assert list(bpe.encode_iterable([text[i:i+3] for i in range(0,len(text),3)])) == ids
bpe.load({history['tokenizer_prefix']!r})
assert bpe._native_encoder_state is None and bpe._native_encode_metrics is None
assert bpe._encode_serial(text) == ids
bpe._cache.clear()
for piece in ('a','b','c'):bpe._encode_uncached(piece,2)
assert set(bpe._cache)=={{'c'}}
'''
    subprocess.run([sys.executable, '-c', code], check=True)


@pytest.mark.parametrize('failure', ['source', 'binary', 'abi', 'input', 'cpu', 'incomplete', 'confirmation'])
def test_history_rejects_mismatched_provenance(tmp_path, native_build, failure):
    args, history = adapter_history(tmp_path, native_build)
    origin = history['candidate_snapshots']['CY-ED']
    if failure in ('source', 'binary'):
        name = 'task1_native_runtime.py' if failure == 'source' else origin['native_build']['binary']
        (Path(origin['directory']) / name).write_bytes(b'changed')
    elif failure == 'abi':
        path = Path(origin['directory']) / 'native_build.json'
        build = json.loads(path.read_text())
        build['soabi'] = 'wrong-abi'
        path.write_text(json.dumps(build))
        # Maintain source hashes to exercise ABI validation rather than the earlier hash guard.
        origin['source_sha256']['native_build.json'] = benchmark.common.sha256(path)
        origin['native_build'] = build
        (Path(origin['directory']) / 'snapshot.json').write_text(json.dumps(origin))
        Path(args.previous_selection).write_text(json.dumps(history))
    elif failure == 'input':
        Path(history['valid_file']).write_text('changed')
    elif failure == 'cpu':
        args.cpu_ids = [0]
    elif failure == 'incomplete':
        history['completed'] = False
        Path(args.previous_selection).write_text(json.dumps(history))
    else:
        path = Path(args.plain_confirmation)
        value = json.loads(path.read_text())
        value['native_decode']['extension_path'] = '/other/module.so'
        path.write_text(json.dumps(value))
    with pytest.raises((ValueError, RuntimeError)):
        benchmark.load_history(args)


def test_live_context_warning_does_not_become_correctness_failure(tmp_path, native_build):
    args, _ = adapter_history(tmp_path, native_build)
    _, _, _, _, context = benchmark.load_history(args)
    assert context['context_regression_flag'] is True
    assert context['context_ratio_to_plain_median'] == 1.13


@pytest.mark.parametrize('wins,expected', [(5, 'A1'), (3, 'A0')])
def test_stage_counts_order_and_no_negative_control_ranking(wins, expected):
    rows = []
    def trial(stage, method, repeat=0, diagnostic=False):
        row = {'trial_id': str(len(rows)), 'stage': stage, 'method': method, 'repeat': repeat,
               'diagnostic': diagnostic, 'status': 'ok', 'ids_match_reference': True, 'roundtrip_passed': True,
               'total_seconds': 8 if method == 'A0' else (6 if repeat <= wins else 9)}
        rows.append(row)
        return row
    selection = benchmark.run_stages(SimpleNamespace(seed=42, repeats=5, diagnostics=True), trial)
    assert selection['recommended'] == expected
    assert len([r for r in rows if r['diagnostic']]) == 3
    assert len([r for r in rows if r['stage'] == 'screening']) == 10
    assert len([r for r in rows if r['stage'].startswith('paired-')]) == 10
    assert all(r['diagnostic'] for r in rows if r['method'] == 'OLD-CY-ED')


def test_tiny_complete_exploration_and_plain_confirmation(tmp_path, native_build):
    args, _ = adapter_history(tmp_path, native_build)
    args.diagnostics = True
    benchmark.exploration(args)
    output = Path(args.output_dir)
    selection = json.loads((output / 'selection.json').read_text())
    assert selection['completed'] and selection['measurements_valid']
    assert selection['upload_status'] == 'disabled'
    rows = json.loads((output / 'results.json').read_text())
    assert len(rows) == 10
    assert all(r['system_load_before']['utc'] and r['system_load_after']['utc'] for r in rows)
    assert all(r['status'] == 'ok' for r in rows)
    assert json.loads((output / 'integrity.json').read_text()) == {'changed_inputs': [], 'source_changed': False}
    # Tiny timing is noisy. Exercise confirmation independently from gate arithmetic,
    # which is already covered by the existing five-pair integration threshold tests.
    selection['ready_for_plain_confirmation'] = True
    (output / 'selection.json').write_text(json.dumps(selection))
    confirmation_args = benchmark.parse_args(['--phase', 'plain-confirmation', '--selection-report',
                                              str(output / 'selection.json'), '--output-dir', str(tmp_path / 'plain'),
                                              '--development', '--no-wandb'])
    benchmark.plain_confirmation(confirmation_args)
    confirmation = json.loads((tmp_path / 'plain/confirmation.json').read_text())
    assert confirmation['status'] == 'ok' and confirmation['patching'] is False
    assert confirmation['adapter_case'] in ('A0', 'A1')


def test_upload_mapping_and_failure_keep_computation_valid(tmp_path, native_build):
    _, history = adapter_history(tmp_path, native_build)
    origins = history['candidate_snapshots']
    snapshots = {case: packaging.make_snapshot(tmp_path / case, origins['CY-D' if case == 'A0' else 'CY-ED'], case)
                 for case in ('A0', 'A1')}
    (tmp_path / 'manifest.json').write_text(json.dumps({'origin_snapshots': origins}))
    class Summary:
        def update(self, mapping):
            assert isinstance(mapping, dict)
    class Run:
        id, summary = 'adapter-test', Summary()
        def log(self, value):
            pass
        def log_artifact(self, value):
            pass
        def save(self, *args, **kwargs):
            pass
    selection = {'completed': True, 'measurements_valid': True, 'recommended': 'A1',
                 'candidate_snapshots': snapshots, 'snapshot': snapshots['A1'], 'integration': {'passed': True}}
    benchmark.publish(Run(), tmp_path, [], selection)
    @contextmanager
    def fail_finish():
        yield Run()
        raise RuntimeError('simulated upload failure')
    with benchmark.native.recording_context(fail_finish(), tmp_path, selection):
        pass
    assert selection['completed'] and selection['measurements_valid']
    assert selection['upload_status'] == 'failed'
    assert json.loads((tmp_path / 'upload_failure.json').read_text())['measurements_valid']


def test_worker_native_error_reaps_all_owned_processes(tmp_path, native_build):
    _, history = adapter_history(tmp_path, native_build)
    fixed = packaging.make_snapshot(tmp_path / 'candidate', history['candidate_snapshots']['CY-ED'], 'A1')
    code = f"""
import sys
sys.path.insert(0,{fixed['directory']!r})
from bpe import BPE
import task1_transport_runtime as transport
original=transport.subprocess.Popen
children=[]
def launch(*args,**kwargs):
    child=original(*args,**kwargs)
    children.append(child)
    return child
transport.subprocess.Popen=launch
bpe=BPE()
bpe.load({history['tokenizer_prefix']!r})
del bpe.vocab[228]
try:
    bpe.encode({sample()!r})
except RuntimeError:
    assert children and all(p.poll() is not None for p in children)
else:
    raise AssertionError('Native encoding error returned partial output')
"""
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, timeout=15)


@pytest.mark.parametrize('sig_name', ['SIGINT', 'SIGTERM'])
def test_fast_worker_interrupt_cleanup(tmp_path, native_build, sig_name):
    import signal
    import time
    import psutil
    _, history = adapter_history(tmp_path, native_build)
    fixed = packaging.make_snapshot(tmp_path / 'candidate', history['candidate_snapshots']['CY-ED'], 'A1')
    ready = tmp_path / 'ready.json'
    code = f"""
import sys,time,json
from pathlib import Path
sys.path.insert(0,{fixed['directory']!r})
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
"""
    with (tmp_path / 'signal.log').open('w') as log:
        process = subprocess.Popen([sys.executable, '-c', code], stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 15
            while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(.02)
            assert ready.exists()
            children = json.loads(ready.read_text())
            process.send_signal(getattr(signal, sig_name))
            process.wait(timeout=10)
            assert process.returncode != 0
            assert all(not psutil.pid_exists(pid) for pid in children)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


def test_end_audit_catches_binary_or_source_changes(tmp_path, native_build):
    args, history = adapter_history(tmp_path, native_build)
    fixed = packaging.make_snapshot(tmp_path / 'candidate', history['candidate_snapshots']['CY-ED'], 'A1')
    manifest = {'source_sha256': benchmark.common.source_fingerprints()}
    benchmark.audit(tmp_path, manifest, history['input_sha256'], [fixed])
    binary = Path(fixed['directory']) / fixed['native_build']['binary']
    binary.write_bytes(b'changed after a cached load')
    with pytest.raises(ValueError, match='source changed'):
        benchmark.audit(tmp_path, manifest, history['input_sha256'], [fixed])
