"""Small texts only: public API, transport protocols, resources and experiment orchestration."""
from array import array
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import psutil
import pytest

from bpe import BPE
import bpe_memory_runtime as baseline
import benchmark_task1_transport as benchmark
import task1_transport_runtime as runtime


def tokenizer():
    vocab = dict(enumerate(bytes([i]) for i in range(256)))
    vocab[256] = b'ab'
    return BPE(vocab, [(b'a', b'b')], ['<x>', '<x>long', 'a<x>', '🙂special'])


def text():
    return 'a<x> ab 中文🙂\r\n  <x>long\n<x> 后续 ab\n🙂special\t' * 80

@pytest.mark.parametrize('case', runtime.CASES)
@pytest.mark.parametrize('diagnostic', [False, True])
def test_real_public_api_exact_ids_cleanup_and_worker_isolation(case, diagnostic, monkeypatch):
    bpe = tokenizer()
    original = baseline._launch
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3')
    monkeypatch.setenv('OMP_NUM_THREADS', '7')
    with runtime.selected_backend(case, diagnostic, development_workers=2) as (stats, calls):
        ids = bpe.encode(text())
        restored = bpe.decode(ids)
    assert baseline._launch is original
    assert not bpe._cache and calls == [case]
    assert type(ids) is list and ids == bpe._encode_serial(text())
    assert restored == text()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '3' and os.environ['OMP_NUM_THREADS'] == '7'
    if case == 'B' and not diagnostic:
        assert stats == {'baseline_unmodified': True}
    else:
        metadata = stats.get('worker_metadata', stats.get('coordinator', {}).get('worker_metadata'))
        assert sum(item['tasks'] for item in metadata) == 32
        assert sum(item['tokens'] for item in metadata) == len(ids)
        assert all(item['cache_entries'] <= 1048576 for item in metadata)
        if diagnostic:
            info = stats if case == 'D' else stats['coordinator']
            assert all(not item['forbidden_imports'] for item in info['worker_startup'])
            assert all(item['gpu_visibility'] == '' for item in info['worker_startup'])
            assert all(set(item['threads'].values()) == {'1'} for item in info['worker_startup'])
            assert stats['assembly_seconds'] >= 0
    if case == 'S':
        for name in stats['shm_names']:
            with pytest.raises(FileNotFoundError):
                runtime.shared_memory.SharedMemory(name=name, track=False)


def test_out_of_order_duplicate_and_incomplete_collector():
    collector = runtime.Collector(3)
    collector.accept(2, array('I', [5, 6]))
    assert collector.ids == []
    collector.accept(0, array('I', [1, 2]))
    assert collector.ids == [1, 2]
    with pytest.raises(RuntimeError, match='Incomplete'):
        collector.finish()
    with pytest.raises(RuntimeError, match='Duplicate'):
        collector.accept(2, array('I', [99]))
    collector.accept(1, array('I', [3, 4]))
    assert collector.finish() == [1, 2, 3, 4, 5, 6]


def test_shared_bound_is_input_derived_and_slots_are_reused():
    bpe, owners, stats = tokenizer(), [], {}
    ranges = baseline._partition_ranges(bpe, text(), 32)
    descriptor, byte_ranges = runtime.prepare_shared(text(), ranges, 2, owners, stats)
    try:
        assert len(byte_ranges) == 32
        assert descriptor['slot_bytes'] == 4 * max(len(text()[lo:hi].encode()) for lo, hi in ranges)
        for index, ((lo, hi), (start, end)) in enumerate(zip(ranges, byte_ranges)):
            assert bytes(owners[0].buf[start:end]).decode('utf-8') == text()[lo:hi]
            values = array('I', bpe._encode_serial(text()[lo:hi]))
            offset = (index % 2) * descriptor['slot_bytes']
            owners[1].buf[offset:offset + len(values) * 4] = values.tobytes()
            assert runtime.copy_slot(owners[1], {'slot': index % 2, 'count': len(values)},
                                     descriptor['slot_bytes']) == values
    finally:
        runtime.release_blocks(owners)


def test_shared_capacity_and_version_skip_without_fallback(monkeypatch):
    bpe = tokenizer()
    monkeypatch.setattr(runtime.sys, 'platform', 'linux')
    monkeypatch.setattr(runtime.os, 'statvfs', lambda path: SimpleNamespace(f_bavail=0, f_frsize=1))
    def unexpected(*args, **kwargs):
        pytest.fail('Capacity preflight must happen before shared-memory allocation')
    monkeypatch.setattr(runtime.shared_memory, 'SharedMemory', unexpected)
    with runtime.selected_backend('S', development_workers=2) as (stats, calls):
        with pytest.raises(runtime.Unavailable, match='/dev/shm'):
            bpe.encode(text())
    assert calls == ['S'] and stats['shm_available_bytes'] == 0
    monkeypatch.setattr(runtime.sys, 'version_info', (3, 12))
    with pytest.raises(runtime.Unavailable, match='3.13'):
        runtime.prepare_shared(text(), [(0, len(text()))], 2, [], {})


@pytest.mark.parametrize('case', ['R', 'S', 'D'])
def test_failed_worker_aborts_and_releases_shared_memory(case):
    bpe = tokenizer()
    del bpe.vocab[97]
    with runtime.selected_backend(case, diagnostic=True, development_workers=2) as (stats, calls):
        with pytest.raises(RuntimeError, match='[Ww]orker'):
            bpe.encode('a ' + text())
    for name in stats.get('shm_names', []):
        with pytest.raises(FileNotFoundError):
            runtime.shared_memory.SharedMemory(name=name, track=False)


@pytest.mark.parametrize('case', ['R', 'S', 'D'])
def test_slow_consumer_has_no_backpressure_deadlock(case, monkeypatch):
    original = runtime.Collector.accept
    def slow(self, index, values):
        time.sleep(.002)
        original(self, index, values)
    monkeypatch.setattr(runtime.Collector, 'accept', slow)
    bpe = tokenizer()
    with runtime.selected_backend(case, development_workers=2):
        ids = bpe.encode(text())
    assert ids == bpe._encode_serial(text())


@pytest.mark.parametrize('case', ['R', 'S', 'D'])
@pytest.mark.parametrize('signum', [signal.SIGINT, signal.SIGTERM])
def test_signals_reap_owned_process_groups_and_shared_memory(tmp_path, case, signum):
    # Interrupt a disposable caller after its real child processes have started.
    script = tmp_path / 'caller.py'
    ready = tmp_path / 'ready'
    owners_file = tmp_path / 'owners.json'
    script.write_text('''import sys, signal, json
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import task1_transport_runtime as rt
from bpe import BPE
case, ready, owners = sys.argv[2:]
def interrupted(signum, frame):
    raise KeyboardInterrupt('fixture cancellation')
signal.signal(signal.SIGTERM, interrupted)
old = rt.Collector.accept
def slow(self, index, values):
    Path(ready).write_text('ready')
    import time
    time.sleep(10)
    return old(self, index, values)
rt.Collector.accept = slow
vocab = dict(enumerate(bytes([i]) for i in range(256)))
bpe = BPE(vocab, [], ['<x>'])
try:
    with rt.selected_backend(case, True, 2) as (stats, calls):
        try:
            bpe.encode('abc 中文🙂<x> ' * 200)
        finally:
            Path(owners).write_text(json.dumps(stats.get('shm_names', [])))
except KeyboardInterrupt:
    pass
''', encoding='utf-8')
    process = subprocess.Popen([sys.executable, str(script), str(benchmark.ROOT), case, str(ready), str(owners_file)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    descendants = []
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail('Disposable caller exited before cancellation')
            time.sleep(.02)
        assert ready.exists()
        descendants = psutil.Process(process.pid).children(recursive=True)
        process.send_signal(signum)
        assert process.wait(timeout=10) == 0
        _, alive = psutil.wait_procs(descendants, timeout=3)
        assert all(p.status() in (psutil.STATUS_DEAD, psutil.STATUS_ZOMBIE) for p in alive)
        for name in json.loads(owners_file.read_text()):
            with pytest.raises(FileNotFoundError):
                runtime.shared_memory.SharedMemory(name=name, track=False)
    finally:
        runtime.stop_processes([process], groups=True)
        for p in descendants:
            if p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
                p.kill()


def test_stages_count_pairing_unavailable_and_confirmation():
    args = benchmark.parse_args(['--diagnostics'])
    rows = []
    def trial(stage, case, repeat=0, diagnostic=False):
        row = {'stage': stage, 'case': case, 'repeat': repeat, 'diagnostic': diagnostic,
               'status': 'unavailable' if case == 'S' else 'ok',
               'total_seconds': {'B': 18., 'R': 15., 'S': 1., 'D': 12.}[case],
               'roundtrip_passed': True, 'ids_match_reference': True, 'trial_id': str(len(rows))}
        rows.append(row)
        return row
    selected = benchmark.run_stages(args, trial)
    assert len(rows) == 33 and sum(not row['diagnostic'] for row in rows) == 31
    assert selected['recommended'] == 'D' and selected['paired_confirmation']['paired_wins'] == 5
    assert 'S' not in selected['screening_medians_seconds']
    assert rows[0]['stage'] == 'diagnostic-baseline' and rows[-1]['stage'] == 'confirmation'
    assert selected['confirmation_trial'] == rows[-1]['trial_id']


def test_incorrect_screening_and_cloud_resource_preflights(tmp_path, monkeypatch):
    bad = [{'case': 'B', 'diagnostic': False, 'status': 'ok', 'total_seconds': 1,
            'roundtrip_passed': True, 'ids_match_reference': False}]
    with pytest.raises(ValueError, match='correctness'):
        benchmark.fastest(bad, 1)
    monkeypatch.setattr(benchmark, 'restrict_environment', lambda: None)
    args = benchmark.parse_args(['--repeats', '4'])
    with pytest.raises(ValueError, match='five'):
        benchmark.execute(args)
    args = benchmark.parse_args(['--output-dir', str(tmp_path / 'never-created')])
    monkeypatch.setattr(benchmark.os, 'sched_getaffinity', lambda pid: set(range(8)), raising=False)
    with pytest.raises(ValueError, match='16 distinct allowed CPUs'):
        benchmark.execute(args)
    assert not Path(args.output_dir).exists()


def test_source_mutation_invalidates_selection(tmp_path, monkeypatch):
    bpe = tokenizer()
    prefix = str(tmp_path / 'tokenizer')
    bpe.save(prefix)
    valid = tmp_path / 'valid.txt'
    valid.write_bytes(text().encode())
    ids = bpe._encode_serial(text())
    out = tmp_path / 'mutation'
    versions = iter([{'source': 'before'}, {'source': 'after'}])
    monkeypatch.setattr(benchmark.common, 'source_fingerprints', lambda: next(versions))
    monkeypatch.setattr(benchmark, 'affinity', lambda *args, **kwargs: None)
    monkeypatch.setattr(benchmark, 'restrict_environment', lambda: None)
    def launch(job, *paths):
        if job['write_reference']:
            benchmark.common.check_reference(ids, job['reference'], write=True)
        return {'case': job['case'], 'status': 'ok', 'total_seconds': 1, 'token_count': len(ids),
                'ids_match_reference': True, 'roundtrip_passed': True, 'diagnostic': job['diagnostic']}
    monkeypatch.setattr(benchmark, 'launch_job', launch)
    args = benchmark.parse_args(['--development', '--no-wandb', '--repeats', '1', '--tokenizer-prefix', prefix,
                                 '--valid-file', str(valid), '--output-dir', str(out)])
    with pytest.raises(ValueError, match='Source/input changed'):
        benchmark.execute(args)
    assert json.loads((out / 'integrity.json').read_text())['source_changed']
    assert not (out / 'selection.json').exists() and (out / 'failure.json').exists()


def test_launcher_cleanup_reaches_separate_descendant_session(tmp_path):
    script = tmp_path / 'session.py'
    ready = tmp_path / 'ready'
    script.write_text('''import subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
Path(sys.argv[1]).write_text(str(child.pid))
time.sleep(60)
''', encoding='utf-8')
    process = subprocess.Popen([sys.executable, str(script), str(ready)], start_new_session=True)
    child = None
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready.exists()
        child = psutil.Process(int(ready.read_text()))
        # The descendant owns a separate session and must also receive termination.
        benchmark.stop_trial(process)
        assert process.poll() is not None
        _, alive = psutil.wait_procs([child], timeout=2)
        assert all(p.status() in (psutil.STATUS_DEAD, psutil.STATUS_ZOMBIE) for p in alive)
    finally:
        runtime.stop_processes([process], groups=True)
        if child is not None and child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
            child.kill()


def test_reference_and_full_trial_failure_is_not_ranked(tmp_path):
    bpe = tokenizer()
    prefix = str(tmp_path / 'tokenizer')
    bpe.save(prefix)
    valid = tmp_path / 'valid.txt'
    valid.write_bytes(text().encode())
    reference = tmp_path / 'reference.tokens'
    benchmark.common.check_reference([1], reference, write=True)
    job = {'case': 'D', 'diagnostic': False, 'cpus': [], 'workers': 2, 'tokenizer_prefix': prefix,
           'valid_file': str(valid), 'write_reference': False, 'development': True,
           'expected_tokens': None, 'reference': str(reference)}
    result = tmp_path / 'result.json'
    with pytest.raises(ValueError, match='Reference'):
        benchmark.worker(job, result)
    assert json.loads(result.read_text())['status'] == 'failed'


def test_small_whole_experiment_template_and_source_integrity(tmp_path):
    bpe = tokenizer()
    prefix = str(tmp_path / 'tokenizer')
    bpe.save(prefix)
    valid = tmp_path / 'valid.txt'
    valid.write_bytes(text().encode())
    out = tmp_path / 'transport'
    entry = benchmark.ROOT / 'train_bpe.py'
    source = entry.read_bytes()
    subprocess.run([sys.executable, str(Path(benchmark.__file__).resolve()), '--development', '--no-wandb',
                    '--tokenizer-prefix', prefix, '--valid-file', str(valid), '--repeats', '1',
                    '--diagnostics', '--output-dir', str(out)], check=True, capture_output=True, text=True, timeout=60)
    rows = json.loads((out / 'results.json').read_text())
    assert len(rows) == 10 and rows[0]['stage'] == 'reference'
    assert all(row['status'] == 'ok' and row['ids_match_reference'] and row['roundtrip_passed'] for row in rows[1:])
    assert all(row['returned_type'] == 'list' for row in rows if row['status'] == 'ok')
    final = json.loads((out / 'selection.json').read_text())
    assert final['completed'] and final['confirmation_trial'] == rows[-1]['trial_id']
    assert final['confirmation_seconds'] == rows[-1]['total_seconds']
    assert json.loads((out / 'integrity.json').read_text()) == {'source_changed': False, 'changed_inputs': []}
    assert not list(out.glob('reference-*'))
    assert '跨进程时间重叠' in (out / 'report.md').read_text()
    assert entry.read_bytes() == source
