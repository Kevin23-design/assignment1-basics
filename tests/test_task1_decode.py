"""CPU-only toy decode/integration tests. No full OWT or training."""
import ast
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil
import pytest

from bpe import BPE
import bpe_decode_runtime as runtime
import benchmark_task1_decode as benchmark
import task1_decode_experiment as integration


def tokenizer():
    return BPE({**{i: bytes([i]) for i in range(256)}, 256: b'ab'}, [(b'a', b'b')],
               ['<x>', '<x>long', '<|endoftext|>'])


def text():
    return ('ab 中文🙂\x00\r\n  <x>long尾<x><|endoftext|>' * 100) + '终🙂'


@pytest.mark.parametrize('method', runtime.METHODS)
def test_exact_bytes_invalid_ids_and_sparse_vocab(method):
    bpe = tokenizer()
    ids = bpe._encode_serial(text())
    stats = {}
    assert runtime.decode_ids(bpe.vocab, ids, method, stats, 2, True) == text()
    assert runtime.decode_ids(bpe.vocab, [], method, max_workers=2, development=True) == ''
    assert runtime.decode_ids(bpe.vocab, [0xe4], method, max_workers=2, development=True) == '\ufffd'
    assert runtime.decode_ids(bpe.vocab, [0, 0xe4, 0xb8, 0xad], method, max_workers=2, development=True) == '\x00中'
    for bad in ([-1], [len(bpe.vocab)], [2**100], [1, -1], [float('nan')]):
        with pytest.raises(KeyError):
            runtime.decode_ids(bpe.vocab, bad, method, max_workers=2, development=True)
    assert runtime.decode_ids({0: b'a\x00', 1000000: b'b'}, [0, 1000000], method, development=True) == 'a\x00b'
    if method.startswith('P'):
        assert stats['effective_workers'] == 2
        assert all(not w['forbidden_imports'] and w['gpu_visibility'] == ''
                   and set(w['threads'].values()) == {'1'} for w in stats['worker_startup'])


@pytest.mark.parametrize('method', ['N128', 'N1M'])
def test_numpy_utf8_cross_blocks_and_table_changes(method, monkeypatch):
    monkeypatch.setattr(runtime, 'BLOCK', 2)
    monkeypatch.setattr(runtime, 'TASK', 2)
    vocab = {i: bytes([i]) for i in range(256)}
    assert runtime.decode_ids(vocab, [0xe4, 0xb8, 0xad, 0], method, development=True) == '中\x00'
    vocab[0] = b'x'
    assert runtime.decode_ids(vocab, [0], method, development=True) == 'x'


@pytest.mark.parametrize('failure', ['startup', 'worker'])
def test_decoder_failure_reaps_all_children(failure, monkeypatch):
    started = []
    original = runtime.subprocess.Popen
    def launch(*args, **kwargs):
        if failure == 'startup' and started:
            raise OSError('injected startup failure')
        process = original(*args, **kwargs)
        started.append(process)
        return process
    monkeypatch.setattr(runtime.subprocess, 'Popen', launch)
    vocab = {0: b'a', 1: b'b'} if failure == 'startup' else {0: b'a'}
    ids = [0] * 20 if failure == 'startup' else [0, 1]
    # Bypass the public invalid-ID fallback to inject a worker-side vocabulary error.
    if failure == 'worker':
        monkeypatch.setattr(runtime, 'TASK', 1)
        # A bad value injects a worker-side failure inside raw join.
        vocab = {0: b'a', 1: object()}
    with pytest.raises((OSError, RuntimeError)):
        runtime.parallel_decode(vocab, ids, 2, [])
    assert started and all(p.poll() is not None for p in started)


def fixture_files(tmp_path):
    bpe = tokenizer()
    prefix = str(tmp_path / 'tokenizer')
    bpe.save(prefix)
    valid = tmp_path / 'valid.txt'
    valid.write_bytes(text().encode())
    previous = tmp_path / 'previous'
    previous.mkdir()
    inputs = [str(valid)] + [prefix + s for s in ('.vocab.json', '.merges.json', '.config.json')]
    hashes = {p: benchmark.common.sha256(p) for p in inputs}
    (previous / 'manifest.json').write_text(json.dumps({
        'arguments': {'tokenizer_prefix': prefix, 'valid_file': str(valid)}, 'input_sha256': hashes,
        'workers': 16, 'parts': 256, 'cache_capacity': 1048576, 'decode_block_size': 131072}))
    (previous / 'selection.json').write_text(json.dumps({'completed': True, 'recommended': 'D3',
                                                        'recommended_decoder': 'C2', 'confirmation_seconds': 11.167}))
    reference = tmp_path / 'reference.u32le'
    benchmark.common.check_reference(bpe._encode_serial(text()), reference, write=True)
    return prefix, valid, previous, reference, hashes


@pytest.mark.parametrize('method', ['R0', 'N128', 'P4'])
def test_plain_source_is_unpatched_and_matches_experiment(tmp_path, method):
    prefix, valid, _, reference, hashes = fixture_files(tmp_path)
    snapshot = integration.make_snapshot(tmp_path / 'plain', method, 2, True)
    original = (benchmark.ROOT / 'bpe.py').read_bytes()
    assert (Path(snapshot['directory']) / 'train_bpe.py').read_bytes() == (benchmark.ROOT / 'train_bpe.py').read_bytes()
    tree = ast.parse((Path(snapshot['directory']) / 'bpe.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BPE')
    decode = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'decode')
    assert 'selected_' not in ast.unparse(decode)
    job = {'plain': True, 'snapshot': snapshot, 'tokenizer_prefix': prefix, 'valid_file': str(valid),
           'reference': str(reference), 'expected_tokens': reference.stat().st_size // 4, 'input_sha256': hashes,
           'cpus': [], 'development': True, 'active_wandb': False}
    out = tmp_path / 'result.json'
    benchmark.launch_job(job, tmp_path / 'job.json', out, tmp_path / 'plain.log')
    row = json.loads(out.read_text())
    assert row['status'] == 'ok' and row['patching'] is False and row['ids_match_reference']
    assert row['bpe_source'] == str(Path(snapshot['directory']) / 'bpe.py')
    assert (benchmark.ROOT / 'bpe.py').read_bytes() == original
    integration.verify_snapshot(snapshot)
    (Path(snapshot['directory']) / 'bpe_decode_runtime.py').write_text('# changed')
    with pytest.raises(ValueError, match='source changed'):
        integration.verify_snapshot(snapshot)


def test_integration_thresholds_and_failed_results():
    def rows(seconds):
        return [{'repeat': i, 'status': 'ok', 'ids_match_reference': True, 'roundtrip_passed': True,
                 'diagnostic': False, 'total_seconds': value} for i, value in enumerate(seconds, 1)]
    reference = rows([10] * 5)
    assert integration.integration_decision(reference, rows([10.2] * 5), 5)['passed']
    assert not integration.integration_decision(reference, rows([10.4] * 5), 5)['passed']
    assert not integration.integration_decision(reference, rows([10, 10, 10, 10.6, 10.6]), 5)['passed']
    bad = rows([10] * 5)
    bad[0]['ids_match_reference'] = False
    with pytest.raises(ValueError, match='exact'):
        integration.integration_decision(reference, bad, 5)


def test_stage_counts_and_no_self_pairing():
    def run(best):
        args = benchmark.parse_args(['--diagnostics'])
        records = []
        def trial(stage, method, repeat=0, diagnostic=False):
            row = {'trial_id': str(len(records)), 'stage': stage, 'method': method, 'repeat': repeat,
                   'diagnostic': diagnostic, 'status': 'ok', 'ids_match_reference': True,
                   'roundtrip_passed': True, 'decode_seconds': {'R0': 5, 'N128': 3, 'N1M': 4,
                                                              'P4': 2, 'P8': 3, 'P16': 4}[method],
                   'total_seconds': 8 if method == best else 12}
            records.append(row)
            return row
        return benchmark.run_stages(args, trial), records
    selected, records = run('P4')
    assert selected['recommended'] == 'P4' and selected['nominees'] == ('R0', 'N128', 'P4')
    assert len([r for r in records if r['stage'] == 'decoder-screening']) == 18
    assert len([r for r in records if r['stage'] != 'decoder-screening' and not r['diagnostic']]) == 26
    selected, records = run('R0')
    assert selected['recommended'] == 'R0' and not any(r['stage'].startswith('paired') for r in records)


def test_small_whole_exploration_and_confirmation(tmp_path):
    prefix, valid, previous, reference, _ = fixture_files(tmp_path)
    out = tmp_path / 'experiment'
    script = str(benchmark.ROOT / 'benchmark_task1_decode.py')
    command = [sys.executable, script, '--development', '--no-wandb', '--tokenizer-prefix', prefix,
               '--valid-file', str(valid), '--previous-results-dir', str(previous), '--reference-tokens', str(reference),
               '--decoder-repeats', '1', '--repeats', '1', '--diagnostics', '--output-dir', str(out)]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=90)
    selected = json.loads((out / 'selection.json').read_text())
    rows = json.loads((out / 'results.json').read_text())
    assert selected['completed'] and all(r['status'] == 'ok' and r['ids_match_reference'] for r in rows)
    assert len([r for r in rows if r['stage'] == 'decoder-screening']) == 6
    assert len([r for r in rows if r['stage'] == 'integration-plain']) == 1
    assert json.loads((out / 'integrity.json').read_text()) == {'changed_inputs': [], 'source_changed': False}
    assert '解码专项' in (out / 'report.md').read_text()
    # Toy timing is noisy; verify the orchestration separately after testing the real gate above.
    selected['ready_for_plain_confirmation'] = True
    (out / 'selection.json').write_text(json.dumps(selected))
    confirm = tmp_path / 'confirmation'
    subprocess.run([sys.executable, script, '--phase', 'plain-confirmation', '--development', '--no-wandb',
                    '--selection-report', str(out / 'selection.json'), '--output-dir', str(confirm)],
                   check=True, capture_output=True, text=True, timeout=30)
    result = json.loads((confirm / 'confirmation.json').read_text())
    assert result['status'] == 'ok' and result['patching'] is False and not result['active_wandb']
    snapshot = selected['snapshot']
    (Path(snapshot['directory']) / 'bpe.py').write_text('# mutated')
    with pytest.raises(ValueError, match='source changed'):
        benchmark.plain_confirmation(benchmark.parse_args([
            '--phase', 'plain-confirmation', '--development', '--no-wandb',
            '--selection-report', str(out / 'selection.json'), '--output-dir', str(tmp_path / 'must-not-start')]))


@pytest.mark.parametrize('signum', [signal.SIGINT, signal.SIGTERM])
def test_decoder_signal_cleanup(tmp_path, signum):
    ready = tmp_path / 'ready'
    script = tmp_path / 'caller.py'
    script.write_text('''import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import bpe_decode_runtime as runtime
original = runtime.receive
def slow(*args):
    for connection, message in original(*args):
        if message[0] == 'part':
            Path(sys.argv[2]).touch()
            time.sleep(30)
        yield connection, message
runtime.receive = slow
try:
    runtime.parallel_decode({0: b'a'}, [0] * 100, 2, [])
except KeyboardInterrupt:
    pass
''')
    process = subprocess.Popen([sys.executable, str(script), str(benchmark.ROOT), str(ready)],
                               start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    children = []
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(.02)
        assert ready.exists()
        children = psutil.Process(process.pid).children(recursive=True)
        assert len(children) == 2
        process.send_signal(signum)
        assert process.wait(timeout=10) == 0
        assert not psutil.wait_procs(children, timeout=2)[1]
    finally:
        runtime.stop([process])
        for child in children:
            if child.is_running():
                child.kill()


def test_slow_consumer_preserves_file_order_and_utf8(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'TASK', 2)
    monkeypatch.setattr(runtime, 'BLOCK', 1)
    original = runtime.receive
    def delayed(*args):
        for connection, message in original(*args):
            if message[0] == 'part':
                time.sleep(.002)
            yield connection, message
    monkeypatch.setattr(runtime, 'receive', delayed)
    raw = '中文🙂\x00\r\n'.encode() * 20
    assert runtime.parallel_decode({i: bytes([i]) for i in range(256)}, list(raw), 2, []) == raw.decode()


def test_live_offline_wandb_plain_confirmation(tmp_path, monkeypatch):
    prefix, valid, _, reference, hashes = fixture_files(tmp_path)
    snapshot = integration.make_snapshot(tmp_path / 'plain', 'R0', 2, True)
    monkeypatch.setenv('WANDB_MODE', 'offline')
    monkeypatch.setenv('WANDB_ENTITY', 'test-entity')
    monkeypatch.setenv('WANDB_PROJECT', 'local-test')
    monkeypatch.setenv('STUDENT_ID', 'student')
    monkeypatch.setenv('WANDB_DIR', str(tmp_path))
    monkeypatch.setenv('WANDB_SILENT', 'true')
    job = {'plain': True, 'snapshot': snapshot, 'tokenizer_prefix': prefix, 'valid_file': str(valid),
           'reference': str(reference), 'expected_tokens': reference.stat().st_size // 4, 'input_sha256': hashes,
           'cpus': [], 'development': True, 'active_wandb': True}
    result_path = tmp_path / 'result.json'
    benchmark.launch_job(job, tmp_path / 'job.json', result_path, tmp_path / 'active.log')
    result = json.loads(result_path.read_text())
    assert result['active_wandb'] and result['wandb_mode'] == 'offline' and result['patching'] is False
    assert result['status'] == 'ok' and result['run_id'] and result['ids_match_reference']
    assert list((Path(snapshot['directory']) / 'outputs').glob('*/source/bpe.py'))


def test_resource_and_provenance_preflights(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark.os, 'sched_getaffinity', lambda pid: set(range(15)), raising=False)
    with pytest.raises(ValueError, match='16 distinct'):
        benchmark.choose_cpus(benchmark.parse_args([]), 16)
    monkeypatch.setattr(runtime.os, 'sched_getaffinity', lambda pid: {0, 1}, raising=False)
    with pytest.raises(RuntimeError, match='Insufficient'):
        runtime.decode_ids({0: b'a'}, [0] * runtime.TASK, 'P4')
    with pytest.raises(ValueError, match='five full'):
        benchmark.exploration(benchmark.parse_args(['--repeats', '4']))
    report = tmp_path / 'selection.json'
    report.write_text(json.dumps({'completed': True, 'ready_for_plain_confirmation': False}))
    with pytest.raises(ValueError, match='gate did not pass'):
        benchmark.plain_confirmation(benchmark.parse_args(['--selection-report', str(report)]))
