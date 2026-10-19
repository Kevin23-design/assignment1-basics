"""Small-text pipeline checks: no full OWT, tokenizer training, GPU or W&B upload."""
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil
import pytest

from bpe import BPE
import bpe_memory_runtime as baseline
import benchmark_task1_pipeline as benchmark
import task1_pipeline_runtime as runtime


def tokenizer():
    return BPE({**{i: bytes([i]) for i in range(256)}, 256: b'ab'}, [(b'a', b'b')],
               ['<x>', '<x>long', '<|endoftext|>'])


def text():
    return ('ab 中文🙂\r\n\t  <x>long尾<x>\n<|endoftext|>' * 100) + '终🙂'


@pytest.mark.parametrize('method', runtime.DECODERS)
@pytest.mark.parametrize('block_size', [1, 2, 7, 131072])
def test_decoder_bytes_boundaries_invalid_and_live_vocab(method, block_size):
    bpe = tokenizer()
    ids = bpe._encode_serial(text())
    assert runtime.decode(bpe, ids, method, block_size=block_size) == text()
    assert runtime.decode(bpe, [], method) == ''
    assert runtime.decode(bpe, [0xe4], method, block_size=block_size) == '\ufffd'
    assert runtime.decode(bpe, [0xe4, 0xb8, 0xad], method, block_size=block_size) == '中'
    for bad in ([-1], [999999], [97, -1, 98]):
        with pytest.raises(KeyError):
            runtime.decode(bpe, bad, method, block_size=block_size)
    bpe.vocab[97] = b'z'
    assert runtime.decode(bpe, [97], method) == 'z'
    sparse = BPE({0: b'a', 1000000: b'b'})
    assert runtime.decode(sparse, [0, 1000000], method) == 'ab'


@pytest.mark.parametrize('case', runtime.CASES)
@pytest.mark.parametrize('diagnostic', [False, True])
def test_public_pipeline_exact_ids_and_diagnostics(case, diagnostic):
    bpe = tokenizer()
    original = BPE.decode
    with runtime.selected_pipeline(case, 'C3' if case in ('D2', 'D3') else 'C0', diagnostic, 2) as (stats, calls):
        ids = bpe.encode(text())
        assert bpe.decode(ids) == text()
    assert BPE.decode is original
    assert ids == bpe._encode_serial(text()) and calls == ['D']
    assert sum(w['tokens'] for w in stats['worker_metadata']) == len(ids)
    if diagnostic:
        assert len(stats['process_creation']) == len(stats['initialization_send']) == 2
        assert stats['startup_seconds'] >= 0 and stats['assembly_seconds'] >= 0
        assert stats['decoder']['utf8_seconds'] >= 0
        assert all(w['tokenizer_rebuild_seconds'] >= 0 and w['init_receive_seconds'] >= 0
                   and not w['forbidden_imports'] and w['gpu_visibility'] == '' for w in stats['worker_startup'])


@pytest.mark.parametrize('case,expected', [('D0', ['start', 'init', 'start', 'init']),
                                           ('D1', ['start', 'start', 'init', 'init'])])
def test_initialization_order(case, expected, monkeypatch):
    events = []
    popen = runtime.transport.subprocess.Popen
    send = runtime.transport.Connection.send
    def launch(*args, **kwargs):
        events.append('start')
        return popen(*args, **kwargs)
    def transmit(connection, obj):
        if isinstance(obj, dict) and 'state' in obj:
            events.append('init')
        return send(connection, obj)
    monkeypatch.setattr(runtime.transport.subprocess, 'Popen', launch)
    monkeypatch.setattr(runtime.transport.Connection, 'send', transmit)
    bpe = tokenizer()
    with runtime.selected_pipeline(case, 'C0', development_workers=2):
        assert bpe.encode(text()) == bpe._encode_serial(text())
    assert events == expected


@pytest.mark.parametrize('failure', ['startup', 'worker'])
def test_failure_reaps_all_owned_workers(failure, monkeypatch):
    started = []
    popen = runtime.transport.subprocess.Popen
    def launch(*args, **kwargs):
        if failure == 'startup' and started:
            raise OSError('injected startup failure')
        process = popen(*args, **kwargs)
        started.append(process)
        return process
    monkeypatch.setattr(runtime.transport.subprocess, 'Popen', launch)
    bpe = tokenizer()
    if failure == 'worker':
        del bpe.vocab[0xe4]
    ranges = baseline._partition_ranges(bpe, text(), 32)
    with pytest.raises((OSError, RuntimeError)):
        runtime.launch_all_first(bpe, text(), ranges, [], 2, False, {})
    assert started and all(p.poll() is not None for p in started)


def test_stages_decoder_nomination_full_count_and_no_self_pairing():
    def run(best):
        rows = []
        args = benchmark.parse_args(['--diagnostics'])
        def trial(stage, case, repeat=0, decoder='C0', diagnostic=False):
            row = {'trial_id': str(len(rows)), 'stage': stage, 'case': case, 'decoder': decoder,
                   'repeat': repeat, 'diagnostic': diagnostic, 'status': 'ok', 'roundtrip_passed': True,
                   'ids_match_reference': True, 'decode_seconds': {'C0': 5, 'C1': 4, 'C2': 3, 'C3': 2}[decoder],
                   'total_seconds': (8 if case == best else 16)}
            rows.append(row)
            return row
        return benchmark.run_stages(args, trial), rows
    selected, rows = run('D3')
    assert selected['recommended'] == 'D3' and selected['decoder_nominee'] == 'C3'
    assert selected['paired_confirmation']['paired_wins'] == 5
    assert sum(r['stage'] == 'decoder-screening' for r in rows) == 12
    assert sum(r['case'] in runtime.CASES and not r['diagnostic'] for r in rows) == 31
    assert sum(r['diagnostic'] for r in rows) == 2
    assert rows[-1]['stage'] == 'confirmation'
    selected, rows = run('D0')
    assert selected['recommended'] == 'D0' and not any(r['stage'].startswith('paired') for r in rows)
    assert sum(r['diagnostic'] for r in rows) == 1
    with pytest.raises(ValueError, match='correctness'):
        benchmark.medians([{'status': 'ok', 'ids_match_reference': False}], ['C0'], 1, 'decode_seconds')


def fixture_files(tmp_path):
    bpe = tokenizer()
    prefix = str(tmp_path / 'tokenizer')
    bpe.save(prefix)
    valid = tmp_path / 'valid.txt'
    valid.write_bytes(text().encode())
    history = tmp_path / 'previous'
    history.mkdir()
    inputs = [str(valid)] + [prefix + s for s in ('.vocab.json', '.merges.json', '.config.json')]
    hashes = {p: benchmark.common.sha256(p) for p in inputs}
    manifest = {'arguments': {'valid_file': str(valid), 'tokenizer_prefix': prefix}, 'input_sha256': hashes,
                'workers': 16, 'parts': 256, 'cache_capacity': 1048576, 'decode_block_size': 131072}
    (history / 'manifest.json').write_text(json.dumps(manifest))
    (history / 'selection.json').write_text(json.dumps({'completed': True, 'recommended': 'D',
                                                       'confirmation_seconds': 15.621562}))
    return prefix, valid, history, hashes


def test_history_mismatch_and_u32_format(tmp_path):
    _, _, history, hashes = fixture_files(tmp_path)
    assert benchmark.historical_inputs(history, hashes)['confirmation_seconds'] == 15.621562
    with pytest.raises(ValueError, match='fingerprint'):
        benchmark.historical_inputs(history, {**hashes, next(iter(hashes)): 'bad'})
    reference = tmp_path / 'ref'
    benchmark.common.check_reference([0, 32000, 65537], reference, write=True)
    assert benchmark.read_ids(reference) == [0, 32000, 65537]
    reference.write_bytes(b'x')
    with pytest.raises(ValueError, match='uint32'):
        benchmark.read_ids(reference)


def test_worker_never_trains_and_rejects_wrong_reference(tmp_path, monkeypatch):
    prefix, valid, _, _ = fixture_files(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail('Exploration must not call BPE.train')
    monkeypatch.setattr(BPE, 'train', forbidden)
    bpe = tokenizer()
    reference = tmp_path / 'ref'
    benchmark.common.check_reference(bpe._encode_serial(text()), reference, write=True)
    job = {'stage': 'screening', 'case': 'D3', 'decoder': 'C3', 'tokenizer_prefix': prefix,
           'valid_file': str(valid), 'reference': str(reference), 'workers': 2, 'cpus': [],
           'development': True, 'diagnostic': False, 'expected_tokens': len(bpe._encode_serial(text()))}
    result = tmp_path / 'result.json'
    benchmark.worker({**job, 'stage': 'reference', 'case': 'serial', 'write_reference': True}, result)
    assert json.loads(result.read_text())['reference_generated'] is True
    benchmark.worker(job, result)
    assert json.loads(result.read_text())['ids_match_reference']
    benchmark.common.check_reference([0] * job['expected_tokens'], reference, write=True)
    with pytest.raises(ValueError, match='comparison'):
        benchmark.worker(job, result)
    assert json.loads(result.read_text())['status'] == 'failed'


def test_small_complete_experiment_and_unchanged_course_entry(tmp_path):
    prefix, valid, history, _ = fixture_files(tmp_path)
    out = tmp_path / 'experiment'
    entry = Path(benchmark.ROOT / 'train_bpe.py').read_bytes()
    subprocess.run([sys.executable, str(Path(benchmark.__file__).resolve()), '--development', '--no-wandb',
                    '--tokenizer-prefix', prefix, '--valid-file', str(valid), '--previous-results-dir', str(history),
                    '--repeats', '1', '--decoder-repeats', '1', '--diagnostics', '--output-dir', str(out)],
                   check=True, capture_output=True, text=True, timeout=60)
    rows = json.loads((out / 'results.json').read_text())
    assert rows[0]['stage'] == 'reference'
    assert all(r['status'] == 'ok' and r['ids_match_reference'] and r['roundtrip_passed'] for r in rows[1:])
    assert len([r for r in rows if r['stage'] == 'decoder-screening']) == 4
    assert len([r for r in rows if r['stage'] == 'screening']) == 4
    assert json.loads((out / 'manifest.json').read_text())['training'] is False
    selected = json.loads((out / 'selection.json').read_text())
    assert selected['completed'] and selected['confirmation_trial'] == rows[-1]['trial_id']
    assert json.loads((out / 'integrity.json').read_text()) == {'changed_inputs': [], 'source_changed': False}
    assert not list(out.glob('reference-*')) and entry == Path(benchmark.ROOT / 'train_bpe.py').read_bytes()
    report = (out / 'report.md').read_text()
    assert '不属于完整成绩' in report and '消融' in report and '不承诺达到5秒' in report


@pytest.mark.parametrize('signum', [signal.SIGINT, signal.SIGTERM])
def test_signal_cleanup_of_all_first_workers(tmp_path, signum):
    ready = tmp_path / 'ready'
    script = tmp_path / 'caller.py'
    script.write_text('''import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from bpe import BPE
import task1_pipeline_runtime as runtime
bpe = BPE({i: bytes([i]) for i in range(256)}, [], ['<x>'])
text = '中文🙂 ab<x>' * 1000
accept = runtime.transport.Collector.accept
def slow(self, index, values):
    Path(sys.argv[2]).touch()
    time.sleep(30)
    accept(self, index, values)
runtime.transport.Collector.accept = slow
try:
    with runtime.selected_pipeline('D3', 'C3', development_workers=2):
        bpe.encode(text)
except KeyboardInterrupt:
    pass
''')
    process = subprocess.Popen([sys.executable, str(script), str(benchmark.ROOT), str(ready)],
                               start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    children = []
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline and process.poll() is None:
            time.sleep(.02)
        assert ready.exists()
        children = psutil.Process(process.pid).children(recursive=True)
        assert len(children) == 2
        process.send_signal(signum)
        assert process.wait(timeout=10) == 0
        _, alive = psutil.wait_procs(children, timeout=2)
        assert not alive
    finally:
        runtime.transport.stop_processes([process], groups=True)
        for child in children:
            if child.is_running():
                child.kill()


def test_cloud_preflights_and_source_mutation_invalidate_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, 'restrict_environment', lambda: None)
    with pytest.raises(ValueError, match='five full and three decoder'):
        benchmark.execute(benchmark.parse_args(['--repeats', '4']))
    monkeypatch.setattr(benchmark.os, 'sched_getaffinity', lambda pid: set(range(8)), raising=False)
    with pytest.raises(ValueError, match='16 distinct allowed'):
        benchmark.execute(benchmark.parse_args([]))
    prefix, valid, history, _ = fixture_files(tmp_path)
    out = tmp_path / 'mutation'
    versions = iter([{'code': 'before'}, {'code': 'after'}])
    monkeypatch.setattr(benchmark.common, 'source_fingerprints', lambda: next(versions))
    monkeypatch.setattr(benchmark, 'affinity', lambda *args, **kwargs: None)
    def launch(job, *paths):
        if job['write_reference']:
            benchmark.common.check_reference(tokenizer()._encode_serial(text()), job['reference'], write=True)
        return {'status': 'ok', 'ids_match_reference': True, 'roundtrip_passed': True,
                'token_count': len(tokenizer()._encode_serial(text())), 'total_seconds': 1.,
                'decode_seconds': 1., 'diagnostic': job['diagnostic']}
    monkeypatch.setattr(benchmark, 'launch_job', launch)
    args = benchmark.parse_args(['--development', '--no-wandb', '--repeats', '1', '--decoder-repeats', '1',
                                 '--tokenizer-prefix', prefix, '--valid-file', str(valid),
                                 '--previous-results-dir', str(history), '--output-dir', str(out)])
    with pytest.raises(ValueError, match='Source/input changed'):
        benchmark.execute(args)
    assert json.loads((out / 'integrity.json').read_text())['source_changed']
    assert not (out / 'selection.json').exists() and (out / 'failure.json').exists()
