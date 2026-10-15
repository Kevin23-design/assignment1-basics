"""Task-one small-memory fixtures only; no OWT benchmark or W&B upload."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from bpe import BPE
import benchmark_task1_efficiency as benchmark
from task1_memory_encoding import InsufficientBoundaries, decode_blocked, encode_parallel_memory, split_text


def tokenizer():
    vocab = dict(enumerate(bytes([i]) for i in range(256)))
    vocab[256] = b'ab'
    return BPE(vocab, [(b'a', b'b')], ['<x>', '<x>long', 'a<x>', '🙂special'])


def sample():
    return ('a<x> <x>long ab 中文🙂\r\n  <x> 新文档 ab\n' * 30)


def test_safe_split_and_missing_boundaries():
    bpe = tokenizer()
    text = sample()
    for count in [1, 2, 4, 8]:
        ranges = split_text(text, bpe, count, '<x>')
        assert len(ranges) == count
        assert ranges[0][0] == 0 and ranges[-1][1] == len(text)
        assert all(first[1] == second[0] for first, second in zip(ranges, ranges[1:]))
        actual_ends = {m.end() for m in bpe._special_re.finditer(text) if m.group() == '<x>'}
        assert all(end in actual_ends for _, end in ranges[:-1])
        assert ''.join(text[start:end] for start, end in ranges) == text
    with pytest.raises(InsufficientBoundaries):
        split_text('a<x> <x>long', bpe, 2, '<x>')
    with pytest.raises(ValueError):
        split_text(text, bpe, 2, '<missing>')


@pytest.mark.parametrize('workers,multiplier', [(1, 1), (2, 1), (2, 2), (2, 4)])
def test_memory_parallel_exact_ids(tmp_path, workers, multiplier):
    bpe = tokenizer()
    prefix = str(tmp_path / 'tokenizer')
    bpe.save(prefix)
    ids, metrics = encode_parallel_memory(sample(), bpe, prefix, workers, multiplier, '<x>', [], diagnostic=True)
    assert type(ids) is list and all(type(value) is int for value in ids)
    assert ids == bpe.encode(sample())
    assert bpe.decode(ids) == sample()
    assert metrics['parts'] == workers * multiplier
    assert len(metrics['worker_metadata']) == workers
    assert metrics['cache_diagnostics']['queries'] > 0
    assert all(item['cache_entries'] <= 1048576 for item in metrics['worker_metadata'])
    assert sum(item['tokens'] for item in metrics['worker_metadata']) == len(ids)
    for item in metrics['worker_metadata']:
        assert len(item['task_diagnostics']) == item['tasks']
        assert all(item[key] >= 0 for key in ('encode_seconds', 'pack_seconds', 'send_seconds', 'job_wait_seconds'))
    assert metrics['pipeline_diagnostics']['join_seconds'] >= 0
    assert not list(tmp_path.glob('*.tokens'))


@pytest.mark.parametrize('block', [1, 2, 3, 64])
def test_blocked_decode_preserves_utf8_and_replacement(block):
    bpe = tokenizer()
    ids = bpe.encode('中文🙂 ab<x>') + [255]
    assert decode_blocked(bpe, ids, block) == bpe.decode(ids)
    assert decode_blocked(bpe, [], block) == ''
    diagnostics = {}
    assert decode_blocked(bpe, ids, block, diagnostics=diagnostics) == bpe.decode(ids)
    assert all(diagnostics[key] >= 0 for key in ('lookup_seconds', 'block_join_seconds',
                                                'final_join_seconds', 'utf8_seconds'))
    with pytest.raises(ValueError):
        decode_blocked(bpe, ids, 0)


def test_worker_failure_does_not_hang(tmp_path):
    bpe = tokenizer()
    with pytest.raises(RuntimeError, match='worker'):
        encode_parallel_memory(sample(), bpe, str(tmp_path / 'missing-tokenizer'), 2, 1, '<x>', [])


def row(method, workers, seconds, status='ok'):
    return {'method': method, 'workers': workers, 'partitions_per_worker': 1, 'decode_block_size': 0,
            'total_seconds': seconds, 'status': status, 'diagnostic': False}


def test_selection_uses_total_time_and_complete_repeats():
    rows = [row('single', 1, 103), row('parallel', 2, 100)]
    assert benchmark.choose(rows, 1)[0]['method'] == 'single'
    rows[0]['total_seconds'] = 103.1
    assert benchmark.choose(rows, 1)[0]['workers'] == 2
    rows.append(row('parallel', 16, 1, status='unavailable'))
    assert benchmark.choose(rows, 1)[0]['workers'] == 2
    with pytest.raises(ValueError):
        benchmark.choose(rows, 3)
    normal = row('single', 1, 100)
    blocked = {**normal, 'decode_block_size': 65536, 'total_seconds': 99}
    assert benchmark.choose([normal, blocked], 1, tolerance=0)[0]['decode_block_size'] == 65536


def test_reference_detects_wrong_ids_and_length(tmp_path):
    path = tmp_path / 'ref.u32le'
    benchmark.check_reference([1, 2, 3], path, write=True)
    benchmark.check_reference([1, 2, 3], path)
    for ids in [[1, 3, 2], [1, 2], [1, 2, 3, 4]]:
        with pytest.raises(ValueError):
            benchmark.check_reference(ids, path)


def test_stage_report_keeps_diagnostics_out_of_ranking(tmp_path):
    rows = [{**row('single', 1, 2), 'stage': 'workers', 'repeat': 1},
            {**row('single', 1, .1), 'stage': 'diagnostic-original', 'repeat': 0,
             'diagnostic': True, 'cache_diagnostics': {'queries': 10, 'hit_rate': .9, 'clears': 0}}]
    benchmark.write_report(tmp_path, rows, {})
    report = (tmp_path / 'report.md').read_text()
    assert '2.000' in report and '0.100' not in report
    assert '独立诊断' in report


def test_small_full_pipeline_with_separate_reference_and_confirmation(tmp_path):
    bpe = tokenizer()
    prefix = str(tmp_path / 'tokenizer')
    bpe.save(prefix)
    valid = tmp_path / 'valid.txt'
    valid.write_bytes(sample().encode())
    out = tmp_path / 'experiment'
    script = Path(__file__).resolve().parents[1] / 'benchmark_task1_efficiency.py'
    subprocess.run([sys.executable, str(script), '--tokenizer-prefix', prefix, '--valid-file', str(valid),
                    '--separator', '<x>', '--workers', '1', '2', '--repeats', '1',
                    '--partition-multipliers', '1', '2', '--decode-block-sizes', '2', '16',
                    '--diagnostics', '--development', '--no-wandb', '--output-dir', str(out)],
                   check=True, capture_output=True, text=True)
    selection = json.loads((out / 'selection.json').read_text())
    assert selection['completed'] and not selection['submission_provenance_verified']
    results = json.loads((out / 'results.json').read_text())
    assert results[0]['stage'] == 'reference' and results[0]['reference_generated']
    assert results[-1]['stage'] == 'confirmation'
    assert all(r['roundtrip_passed'] and r['ids_match_reference'] for r in results[1:] if r['status'] == 'ok')
    assert all(r['returned_type'] == 'list' for r in results if r['status'] == 'ok')
    assert any(r['diagnostic'] for r in results)
    assert list(out.glob('trial-*-profiles/parent.prof'))
    assert not list(out.glob('reference-*'))
    assert json.loads((out / 'integrity.json').read_text()) == {'changed_inputs': [], 'source_changed': False}


def test_parallel_winner_advances_to_partition_and_decode_stages(tmp_path, monkeypatch):
    bpe = tokenizer()
    prefix = str(tmp_path / 'tokenizer')
    bpe.save(prefix)
    valid = tmp_path / 'valid.txt'
    text = sample()
    valid.write_bytes(text.encode())
    ids = bpe.encode(text)
    out = tmp_path / 'experiment'
    # Orchestrator state machine only; real worker/IPC behavior is tested above.
    def launch(job, job_path, result_path, log_path):
        spec = job['spec']
        if job['write_reference']:
            benchmark.check_reference(ids, job['reference'], write=True)
        seconds = 10 if spec['method'] == 'single' else (8 if spec['workers'] == 1 else 2)
        if spec['partitions_per_worker'] == 2:
            seconds *= .8
        if spec['decode_block_size'] == 2:
            seconds *= .5
        return {**spec, 'status': 'ok', 'total_seconds': seconds, 'token_count': len(ids),
                'roundtrip_passed': True, 'ids_match_reference': not job['write_reference'],
                'diagnostic': job['diagnostic'], 'first_200': benchmark.first_tokens(bpe, ids, text)}
    monkeypatch.setattr(benchmark, 'launch_job', launch)
    monkeypatch.setattr(benchmark, 'affinity', lambda *args, **kwargs: None)
    monkeypatch.setattr(benchmark, 'restrict_environment', lambda: None)
    args = benchmark.parse_args(['--development', '--no-wandb', '--tokenizer-prefix', prefix,
                                 '--valid-file', str(valid), '--separator', '<x>', '--workers', '1',
                                 '--repeats', '1', '--partition-multipliers', '1', '2',
                                 '--decode-block-sizes', '2', '--output-dir', str(out)])
    args.workers = [1, 2]
    if hasattr(benchmark.os, 'sched_getaffinity'):
        monkeypatch.setattr(benchmark.os, 'sched_getaffinity', lambda pid: {0, 1})
    benchmark.execute(args)
    selected = json.loads((out / 'selection.json').read_text())['selected']
    assert selected == {'method': 'parallel', 'workers': 2, 'partitions_per_worker': 2, 'decode_block_size': 2}
    rows = json.loads((out / 'results.json').read_text())
    assert sum(row['stage'] == 'partitions' for row in rows) == 2
    assert rows[-1]['stage'] == 'confirmation'
