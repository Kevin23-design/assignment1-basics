"""Small CPU-only transport/provenance tests; never train or read full OWT."""
from array import array
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import psutil
import pytest

from bpe import BPE
import benchmark_task1_encode_transport as benchmark
import task1_decode_experiment as packaging
import task1_encode_transport_runtime as runtime
import task1_transport_runtime as transport


def tokenizer():
    return BPE({**{i: bytes([i]) for i in range(256)}, 256: b'ab'}, [(b'a', b'b')],
               ['<x>', '<x>long', '<|endoftext|>'])


def sample():
    return 'ab中文🙂\r\n  \t<x>long尾<x><|endoftext|>' * 80 + '终🙂'


@pytest.mark.parametrize('method', runtime.METHODS)
def test_real_workers_order_roundtrip_and_cache(method, monkeypatch):
    bpe, text, stats = tokenizer(), sample(), {}
    # Force highly unequal tasks and repeated reuse of each worker's slot.
    ranges = [(i, min(i + 7, len(text))) for i in range(0, len(text), 7)]
    # A single-byte vocabulary plus 'ab' can legitimately merge across arbitrary
    # cuts; use one-character ranges so this transport-only expectation is exact.
    ranges = [(i, i + 1) for i in range(len(text))]
    expected = [value for a, b in ranges for value in bpe._encode_serial(text[a:b])]
    original = transport.Collector.accept
    def slow(self, index, values):
        if index % 300 == 0:
            time.sleep(.002)
        original(self, index, values)
    monkeypatch.setattr(transport.Collector, 'accept', slow)
    ids = runtime.launch(bpe, text, ranges, [], 2, method, True, stats)
    assert ids == expected
    assert bpe.decode(ids) == text
    assert sum(x['tasks'] for x in stats['worker_metadata']) == len(ranges)
    assert all(not x['forbidden_imports'] and x['gpu_visibility'] == ''
               and set(x['threads'].values()) == {'1'} for x in stats['worker_startup'])
    if method != 'T0':
        assert stats['state_serializations'] == 1
    for name in stats.get('shm_names', []):
        with pytest.raises(FileNotFoundError):
            runtime.shared_memory.SharedMemory(name=name, track=False)


def test_collector_independent_copy_before_slot_reuse():
    owners, stats = [], {}
    try:
        spec, _ = runtime.prepare_shared('T2', 'abcdef', [(0, 6)], 1, owners, stats)
        block = owners[0]
        block.buf[:8] = array('I', [4, 5]).tobytes()
        values = transport.copy_slot(block, {'slot': 0, 'count': 2}, spec['slot_bytes'])
        collector = transport.Collector(2)
        collector.accept(1, values)
        block.buf[:8] = array('I', [9, 9]).tobytes()
        collector.accept(0, array('I', [1]))
        assert collector.finish() == [1, 4, 5]
    finally:
        transport.release_blocks(owners)


@pytest.mark.parametrize('method', ['T2', 'T3'])
def test_capacity_check_precedes_allocation(method, monkeypatch):
    monkeypatch.setattr(runtime.sys, 'platform', 'linux')
    monkeypatch.setattr(runtime.os, 'statvfs', lambda _: SimpleNamespace(f_bavail=0, f_frsize=4096))
    owners, stats = [], {}
    with pytest.raises(transport.Unavailable, match='including reserve'):
        runtime.prepare_shared(method, '中🙂', [(0, 2)], 2, owners, stats)
    assert not owners
    assert stats['shm_required_bytes'] > runtime.RESERVE


@pytest.mark.parametrize('failure', ['startup', 'worker'])
@pytest.mark.parametrize('method', ['T1', 'T2', 'T3'])
def test_failure_reaps_children_and_unlinks_shared(method, failure, monkeypatch):
    bpe, stats, children = tokenizer(), {}, []
    original = runtime.subprocess.Popen
    def launch(*args, **kwargs):
        if failure == 'startup' and children:
            raise OSError('injected startup failure')
        child = original(*args, **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(runtime.subprocess, 'Popen', launch)
    if failure == 'worker':
        bpe.vocab[0] = object()  # Picklable malformed vocab fails during worker reconstruction.
    with pytest.raises((OSError, RuntimeError)):
        runtime.launch(bpe, 'ab🙂', [(0, 2), (2, 3)], [], 2, method, True, stats)
    assert children and all(child.poll() is not None for child in children)
    for name in stats.get('shm_names', []):
        with pytest.raises(FileNotFoundError):
            runtime.shared_memory.SharedMemory(name=name, track=False)


def history_fixture(tmp_path):
    bpe = tokenizer()
    prefix = str(tmp_path / 'tokenizer')
    bpe.save(prefix)
    valid, reference = tmp_path / 'valid.txt', tmp_path / 'reference.u32le'
    valid.write_bytes(sample().encode('utf-8'))
    ids = bpe._encode_serial(sample())
    benchmark.common.check_reference(ids, reference, write=True)
    hashes = {str(p): benchmark.common.sha256(p) for p in [valid, reference] + [Path(prefix + suffix)
              for suffix in ('.vocab.json', '.merges.json', '.config.json')]}
    snapshot = packaging.make_snapshot(tmp_path / 'accepted-source', 'R0', 2, True)
    selection = {'completed': True, 'recommended': 'R0', 'ready_for_plain_confirmation': True,
                 'integration': {'passed': True}, 'snapshot': snapshot, 'development': True,
                 'tokenizer_prefix': prefix, 'valid_file': str(valid), 'reference': str(reference),
                 'expected_tokens': len(ids), 'input_sha256': hashes, 'cpu_affinity': []}
    confirmation = {'status': 'ok', 'patching': False, 'source_changed': False, 'method': 'R0',
                    'token_count': len(ids), 'cpu_affinity': [], 'active_wandb': False,
                    'bpe_source': str(Path(snapshot['directory']) / 'bpe.py'),
                    'roundtrip_passed': True, 'ids_match_reference': True}
    selected, confirmed = tmp_path / 'previous-selection.json', tmp_path / 'confirmation.json'
    selected.write_text(json.dumps(selection))
    confirmed.write_text(json.dumps(confirmation))
    args = benchmark.parse_args(['--development', '--no-wandb', '--previous-selection', str(selected),
                                 '--plain-confirmation', str(confirmed), '--repeats', '1',
                                 '--output-dir', str(tmp_path / 'out')])
    return args, selection


@pytest.mark.parametrize('method', runtime.METHODS)
def test_unpatched_snapshot_and_trial_use_accepted_source(tmp_path, method):
    args, history = history_fixture(tmp_path)
    accepted, hashes = benchmark.load_history(args)
    snapshot = benchmark.make_snapshot(tmp_path / 'plain', accepted['snapshot'], method)
    job = {k: history[k] for k in ('tokenizer_prefix', 'valid_file', 'reference', 'expected_tokens')}
    job.update(method=method, stage='screening', diagnostic=True, cpus=[], workers=2, development=True,
               input_sha256=hashes, baseline_snapshot=accepted['snapshot'], snapshot=snapshot, active_wandb=False)
    result = benchmark.launch_job(job, tmp_path / 'job.json', tmp_path / 'result.json', tmp_path / 'trial.log')
    assert result['status'] == 'ok'
    assert result['bpe_source'] == str(Path(accepted['snapshot']['directory']) / 'bpe.py')
    result = benchmark.launch_job({**job, 'plain': True}, tmp_path / 'plain.job.json',
                                 tmp_path / 'plain.json', tmp_path / 'plain.log')
    assert result['status'] == 'ok' and result['patching'] is False
    assert Path(snapshot['directory'], 'train_bpe.py').read_bytes() == Path('train_bpe.py').read_bytes()


@pytest.mark.parametrize('change', ['confirmation', 'source', 'input', 'cpu', 'formal'])
def test_provenance_gate(tmp_path, change):
    args, history = history_fixture(tmp_path)
    if change == 'confirmation':
        path = Path(args.plain_confirmation)
        value = json.loads(path.read_text())
        value['source_changed'] = True
        path.write_text(json.dumps(value))
    elif change == 'source':
        Path(history['snapshot']['directory'], 'bpe.py').write_text('# changed')
    elif change == 'input':
        Path(history['valid_file']).write_text('changed')
    elif change == 'cpu':
        args.cpu_ids = [0]
    else:
        args.development = False
    with pytest.raises(ValueError):
        benchmark.load_history(args)


def test_stage_selection_skips_unavailable_and_requires_paired_improvement():
    args = SimpleNamespace(seed=42, repeats=5, diagnostics=True)
    rows = []
    def trial(stage, method, repeat=0, diagnostic=False):
        row = {'trial_id': str(len(rows)), 'stage': stage, 'method': method, 'repeat': repeat,
               'diagnostic': diagnostic, 'status': 'unavailable' if method == 'T3' else 'ok',
               'error': 'capacity', 'roundtrip_passed': True, 'ids_match_reference': True,
               'total_seconds': {'T0': 11, 'T1': 10.9, 'T2': 10, 'T3': 1}[method]}
        rows.append(row)
        return row
    selected = benchmark.run_stages(args, trial)
    assert selected['recommended'] == 'T2'
    assert selected['paired_confirmation']['paired_wins'] == 5
    assert selected['unavailable'] == {'T3': 'capacity'}
    assert len([r for r in rows if r['method'] == 'T3']) == 1
    assert len([r for r in rows if r['stage'].startswith('paired-')]) == 10


@pytest.mark.parametrize('sig', [signal.SIGINT, signal.SIGTERM])
def test_signal_reaps_workers_and_shared_blocks(tmp_path, sig):
    state = tmp_path / 'ready.json'
    code = f'''
import json,time
from pathlib import Path
from bpe import BPE
import task1_encode_transport_runtime as runtime
import task1_transport_runtime as transport
original = transport.await_ready
def wait(*args):
    original(*args)
    Path({str(state)!r}).write_text(json.dumps({{'children': [p.pid for p in args[1]], 'names': args[2]['shm_names']}}))
    time.sleep(60)
transport.await_ready = wait
bpe=BPE({{i:bytes([i]) for i in range(256)}}, [], [])
runtime.launch(bpe, 'abc', [(0,1),(1,3)], [], 2, 'T3', True, {{}})
'''
    with (tmp_path / 'signal.log').open('w') as log:
        process = subprocess.Popen([sys.executable, '-c', code], stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 15
            while not state.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(.02)
            assert state.exists()
            metadata = json.loads(state.read_text())
            process.send_signal(sig)
            process.wait(timeout=10)
            assert process.returncode != 0
            assert all(not psutil.pid_exists(pid) for pid in metadata['children'])
            for name in metadata['names']:
                with pytest.raises(FileNotFoundError):
                    runtime.shared_memory.SharedMemory(name=name, track=False)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


def test_small_complete_suite_and_report(tmp_path):
    args, _ = history_fixture(tmp_path)
    args.diagnostics = True
    benchmark.exploration(args)
    out = Path(args.output_dir)
    selection = json.loads((out / 'selection.json').read_text())
    assert selection['completed']
    assert set(selection['screening_medians_seconds']) == set(runtime.METHODS)
    rows = json.loads((out / 'results.json').read_text())
    assert all(row['status'] == 'ok' for row in rows)
    assert all(row['token_count'] == selection['expected_tokens'] for row in rows)
    assert '编码传输探索' in (out / 'report.md').read_text()
    assert (out / 'results.csv').exists()
    assert json.loads((out / 'integrity.json').read_text()) == {'source_changed': False, 'changed_inputs': []}
