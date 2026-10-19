"""Native helper tests on tiny inputs only; full OWT is never loaded or trained."""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import subprocess

import pytest

import benchmark_task1_native as benchmark
import build_task1_native as builder
from bpe import BPE
from bpe_decode_runtime import serial
import task1_native_experiment as packaging
import task1_native_runtime as runtime
from tests.test_task1_encode_transport import history_fixture, tokenizer, sample


@pytest.fixture(scope='session')
def native_build(tmp_path_factory):
    pytest.importorskip('Cython')
    directory = tmp_path_factory.mktemp('native-parent') / 'build'
    builder.build(directory)
    return directory


@pytest.fixture(scope='session')
def native(native_build):
    metadata = runtime.verify_build(native_build)
    spec = importlib.util.spec_from_file_location('_task1_native', native_build / metadata['binary'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_decode_bytes_fallbacks_mutation_and_errors(native):
    vocab = {i: bytes([i]) for i in range(256)}
    for ids in ([], [0], list('中文🙂'.encode()), [0xe4], [0xe4, 0xb8, 0xad, 0, 0xf0, 0x9f, 0x99, 0x82],
                [True, False], [1.0, 2], (0, 1)):
        assert native.decode(vocab, ids) == serial(vocab, ids)
    metrics = {}
    assert native.decode(vocab, [1, 2], metrics) == '\x01\x02'
    assert metrics['backend'] == 'native'
    vocab[1] = b'changed'
    assert native.decode(vocab, [1]) == 'changed'
    assert native.decode({0: b'a', 1000000: b'b'}, [1000000, 0]) == 'ba'
    assert native.decode({0: b'', 1: b'\0x'}, [0, 1, 0]) == '\0x'
    for ids in ([-1], [256], [2**200], [-2**200], [float('nan')], [[1]]):
        with pytest.raises(Exception) as before:
            serial(vocab, ids)
        with pytest.raises(type(before.value)):
            native.decode(vocab, ids)
    with pytest.raises(TypeError):
        native.decode({0: object()}, [0])


def test_checked_lengths(native):
    assert native._checked_total(12, 4) == 16
    assert native._checked_total(sys.maxsize, 0) == sys.maxsize
    for total, size in ((sys.maxsize, 1), (sys.maxsize - 1, 2), (-1, 1), (1, -1)):
        with pytest.raises(OverflowError):
            native._checked_total(total, size)


def test_encoder_preserves_rank_all_occurrences_and_arbitrary_ids(native):
    merges = [(b'a', b'a'), (b'a', b'b'), (b'aa', b'b'), (b'aa', b'aa'), (b'ab', b'ab')]
    # Deliberately duplicate byte values and shuffle IDs, including non-byte IDs.
    vocab = {i + 300: bytes([i]) for i in range(256)}
    vocab.update({999 + i: a + b for i, (a, b) in enumerate(merges)})
    vocab[9] = b'a'
    bpe = BPE(vocab, merges, [])
    state = native.Encoder(bpe._ranks, bpe._ids, len(bpe.merges))
    for text in ['', 'aaaaa', 'aaaab', 'ababab', 'aaaaabaaaaab', ''.join(chr(i) for i in range(256))]:
        assert state.encode(text.encode()) == bpe._encode_uncached(text)
    rng = random.Random(42)
    for _ in range(500):
        text = ''.join(rng.choice('aaab b中🙂\r\n\0') for _ in range(rng.randrange(60)))
        assert state.encode(text.encode()) == bpe._encode_uncached(text)
    # Duplicate pairs use the effective _ranks dictionary, not enumerate(vocab).
    bpe = BPE({**{i: bytes([i]) for i in range(256)}, 300: b'ab', 301: b'bc'},
              [(b'a', b'b'), (b'b', b'c'), (b'a', b'b')], [])
    state = native.Encoder(bpe._ranks, bpe._ids, len(bpe.merges))
    assert state.encode(b'abc') == bpe._encode_uncached('abc')
    bad = native.Encoder({(b'a', b'b'): 0}, {i: j for i, j in []}, 1)
    with pytest.raises(KeyError, match='ab'):
        bad.encode(b'ab')


def test_runtime_cache_limit(native_build, monkeypatch):
    monkeypatch.setattr(runtime, '_DIRECTORY', native_build)
    monkeypatch.setattr(runtime, '_loaded', None)
    bpe = tokenizer()
    for value in ['ab', 'a', 'b', 'c']:
        assert runtime.encode_uncached(bpe, value, 2) == bpe._encode_uncached(value)
    bpe._cache.clear()
    runtime.encode_uncached(bpe, 'a', 2)
    runtime.encode_uncached(bpe, 'b', 2)
    runtime.encode_uncached(bpe, 'c', 2)
    assert set(bpe._cache) == {'c'}
    runtime.encode_uncached(bpe, 'a' * 129, 2)
    assert set(bpe._cache) == {'c'}
    assert bpe._native_encode_metrics['native_calls'] > 0


@pytest.mark.parametrize('method', packaging.METHODS)
def test_candidate_whole_interface_and_plain_mode(tmp_path, native_build, method):
    args, history = history_fixture(tmp_path)
    snapshot = packaging.make_snapshot(tmp_path / 'candidate', history['snapshot'], method, native_build)
    job = {k: history[k] for k in ('tokenizer_prefix', 'valid_file', 'reference', 'expected_tokens')}
    job.update(method=method, stage='screening', diagnostic=True, cpus=[], workers=2, development=True,
               input_sha256=history['input_sha256'], snapshot=snapshot, active_wandb=False)
    result = benchmark.launch_job(job, tmp_path / 'job.json', tmp_path / 'result.json', tmp_path / 'trial.log')
    assert result['status'] == 'ok'
    assert result['roundtrip_passed'] and result['ids_match_reference']
    if method != 'B0':
        assert result['native_decode']['backend'] == 'native'
    if method == 'CY-ED':
        assert all(w['native_encoding']['native_calls'] > 0 and not w['native_encoding']['fallback_calls']
                   for w in result['worker_metadata'])
    plain = benchmark.launch_job({**job, 'plain': True}, tmp_path / 'plain.job.json',
                                 tmp_path / 'plain.json', tmp_path / 'plain.log')
    assert plain['status'] == 'ok' and plain['patching'] is False
    if method != 'B0':
        assert plain['native_extension_loaded_during']
    assert Path(snapshot['directory'], 'train_bpe.py').read_bytes() == Path('train_bpe.py').read_bytes()


def test_build_and_snapshot_fingerprints(tmp_path, native_build):
    _, history = history_fixture(tmp_path)
    snapshot = packaging.make_snapshot(tmp_path / 'candidate', history['snapshot'], 'CY-D', native_build)
    assert any(name.endswith('.so') or name.endswith('.pyd') for name in snapshot['source_sha256'])
    assert 'task1_native.pyx' in snapshot['source_sha256']
    assert 'task1_native_generated.c' in snapshot['source_sha256']
    binary = Path(snapshot['directory']) / snapshot['native_build']['binary']
    binary.write_bytes(b'changed')
    with pytest.raises(ValueError, match='source changed'):
        packaging.verify_snapshot(snapshot)
    with pytest.raises(RuntimeError, match='changed'):
        runtime.verify_build(snapshot['directory'])


def test_abi_and_missing_module_fail_explicitly(tmp_path, native_build, monkeypatch):
    from shutil import copytree
    directory = tmp_path / 'build'
    copytree(native_build, directory)
    path = directory / 'build.json'
    metadata = json.loads(path.read_text())
    metadata['soabi'] = 'wrong-abi'
    path.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match='ABI mismatch'):
        runtime.verify_build(directory)
    monkeypatch.setattr(runtime, '_DIRECTORY', tmp_path / 'missing')
    monkeypatch.setattr(runtime, '_loaded', None)
    with pytest.raises(RuntimeError, match='explicitly run'):
        runtime.load_native()


@pytest.mark.parametrize('decode_win,encode_win,expected', [(False, True, 'B0'), (True, False, 'CY-D'),
                                                          (True, True, 'CY-ED')])
def test_gated_stages(decode_win, encode_win, expected):
    rows = []
    def trial(stage, method, repeat=0, diagnostic=False):
        times = {'B0': 11, 'CY-D': 8 if decode_win else 11, 'CY-ED': 6 if encode_win else 8}
        row = {'trial_id': str(len(rows)), 'stage': stage, 'method': method, 'repeat': repeat,
               'diagnostic': diagnostic, 'status': 'ok', 'total_seconds': times[method],
               'roundtrip_passed': True, 'ids_match_reference': True}
        rows.append(row)
        return row
    selection = benchmark.run_stages(SimpleNamespace(seed=42, repeats=5, diagnostics=True), trial)
    assert selection['recommended'] == expected
    assert any(r['method'] == 'CY-ED' for r in rows) == decode_win


def test_summary_upload_accepts_mapping_only_and_failure_does_not_invalidate(tmp_path, native_build, monkeypatch):
    _, history = history_fixture(tmp_path)
    snapshot = packaging.make_snapshot(tmp_path / 'candidate', history['snapshot'], 'B0', native_build)
    (tmp_path / 'manifest.json').write_text(json.dumps({'baseline_snapshot': history['snapshot']}))
    class Summary:
        def update(self, mapping):
            assert isinstance(mapping, dict)
    class Run:
        summary, id = Summary(), 'test'
        def log(self, _):
            pass
        def log_artifact(self, _):
            pass
        def save(self, *args, **kwargs):
            pass
    selected = {'completed': True, 'recommended': 'B0', 'snapshot': snapshot, 'integration': {'passed': True}}
    benchmark.publish(Run(), tmp_path, [], selected)
    @contextmanager
    def failing_finish():
        yield Run()
        raise RuntimeError('upload failed')
    with benchmark.recording_context(failing_finish(), tmp_path, selected):
        pass
    assert selected['completed'] and selected['upload_status'] == 'failed'
    assert json.loads((tmp_path / 'upload_failure.json').read_text())['measurements_valid']


def test_tiny_complete_native_exploration(tmp_path, native_build):
    args, _ = history_fixture(tmp_path)
    args.build_dir = str(native_build)
    args.diagnostics = True
    benchmark.exploration(args)
    output = Path(args.output_dir)
    selection = json.loads((output / 'selection.json').read_text())
    assert selection['completed'] and selection['measurements_valid']
    assert selection['upload_status'] == 'disabled'
    assert json.loads((output / 'integrity.json').read_text()) == {'changed_inputs': [], 'source_changed': False}
    assert (output / 'results.csv').exists()


def test_native_rebuild_streaming_and_python_fallback(tmp_path, native_build):
    _, history = history_fixture(tmp_path)
    snapshot = packaging.make_snapshot(tmp_path / 'candidate', history['snapshot'], 'CY-ED', native_build)
    code = f"""
import sys
sys.path.insert(0, {snapshot['directory']!r})
from bpe import BPE
bpe=BPE()
bpe.load({history['tokenizer_prefix']!r})
text={sample()!r}
ids=bpe._encode_serial(text)
assert bpe._native_encoder_state is not None
assert list(bpe.encode_iterable([text[i:i+3] for i in range(0,len(text),3)])) == ids
bpe.load({history['tokenizer_prefix']!r})
assert bpe._native_encoder_state is None and bpe._native_encode_metrics is None
assert bpe._encode_serial(text) == ids
bpe._ranks={{(b'a',b'b'):len(bpe.merges)+1}}
bpe._native_encoder_state=None
bpe._cache.clear()
assert bpe._encode_uncached('ab') == bpe._python_encode_uncached('ab')
assert bpe._native_encode_metrics['fallback_calls'] == 1
"""
    subprocess.run([sys.executable, '-c', code], check=True)


def test_native_source_is_recorded_and_binary_not_decoded_as_text(tmp_path, monkeypatch):
    import hashlib
    import wandb
    from wandb_record import experiment
    (tmp_path / 'helper.pyx').write_text('cdef int x = 1\n')
    (tmp_path / 'helper.so').write_bytes(b'\xff\x00binary')
    for name, value in [('STUDENT_ID', 'test'), ('WANDB_ENTITY', 'test')]:
        monkeypatch.setenv(name, value)
    class Summary(dict):
        pass
    class Run:
        id, summary = 'record-test', Summary()
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def log_artifact(self, _):
            pass
        def save(self, *args, **kwargs):
            pass
    monkeypatch.setattr(wandb, 'init', lambda **kwargs: Run())
    with experiment('native-source-test', {}, root=tmp_path):
        pass
    output = tmp_path / 'outputs' / 'record-test'
    assert (output / 'source' / 'helper.pyx').is_file()
    assert (output / 'source' / 'helper.so').read_bytes() == b'\xff\x00binary'
    dump = (output / 'source_dump.txt').read_text()
    assert 'BINARY helper.so' in dump and hashlib.sha256(b'\xff\x00binary').hexdigest() in dump


def test_native_active_offline_plain_confirmation(tmp_path, native_build, monkeypatch):
    _, history = history_fixture(tmp_path)
    snapshot = packaging.make_snapshot(tmp_path / 'candidate', history['snapshot'], 'CY-D', native_build)
    monkeypatch.setenv('WANDB_MODE', 'offline')
    monkeypatch.setenv('WANDB_ENTITY', 'test')
    monkeypatch.setenv('STUDENT_ID', 'native-test')
    job = {k: history[k] for k in ('tokenizer_prefix', 'valid_file', 'reference', 'expected_tokens')}
    job.update(method='CY-D', cpus=[], workers=2, development=True, plain=True,
               input_sha256=history['input_sha256'], snapshot=snapshot, active_wandb=True)
    result = benchmark.launch_job(job, tmp_path / 'job.json', tmp_path / 'result.json', tmp_path / 'trial.log')
    assert result['status'] == 'ok' and result['active_wandb'] and result['patching'] is False
    assert result['native_decode']['backend'] == 'native'


@pytest.mark.parametrize('sig_name', ['SIGINT', 'SIGTERM'])
def test_native_worker_interrupt_cleanup(tmp_path, native_build, sig_name):
    import signal
    import time
    import psutil
    _, history = history_fixture(tmp_path)
    snapshot = packaging.make_snapshot(tmp_path / 'candidate', history['snapshot'], 'CY-ED', native_build)
    ready = tmp_path / 'ready.json'
    code = f'''
import sys,time,json
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
            process.send_signal(getattr(signal, sig_name))
            process.wait(timeout=10)
            assert process.returncode != 0
            assert all(not psutil.pid_exists(pid) for pid in children)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
