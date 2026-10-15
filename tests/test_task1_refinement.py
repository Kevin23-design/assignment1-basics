"""Small fixtures and orchestrator checks; never run the full cloud benchmark locally."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

import benchmark_task1_efficiency as benchmark
from tests.test_task1_efficiency import sample, tokenizer


def paired_rows(times):
    return [{'repeat': index, 'total_seconds': seconds, 'status': 'ok', 'diagnostic': False,
             'roundtrip_passed': True, 'ids_match_reference': True}
            for index, seconds in enumerate(times, 1)]


def test_refinement_defaults_preserve_original_suite():
    broad = benchmark.parse_args([])
    assert broad.workers == [1, 2, 4, 8, 16] and broad.repeats == 3
    assert broad.partition_multipliers == [1, 2, 4]
    assert broad.decode_block_sizes == [65536, 262144, 1048576]
    assert broad.output_dir == 'outputs/task1-efficiency'
    refined = benchmark.parse_args(['--suite', 'local-refinement'])
    assert refined.workers == [16] and refined.repeats == 5
    assert refined.partition_multipliers == [2, 4, 8, 16]
    assert refined.decode_block_sizes == [65536, 131072, 262144, 524288]
    assert refined.output_dir == 'outputs/task1-refinement'


@pytest.mark.parametrize('times,promote,wins', [
    ([97] * 5, True, 5), ([97.01] * 5, False, 5),
    ([90, 90, 90, 110, 110], False, 3), ([90, 90, 90, 90, 110], True, 4),
    ([100] * 5, False, 0),
])
def test_paired_threshold_and_consistency(times, promote, wins):
    decision = benchmark.paired_decision(paired_rows([100] * 5), paired_rows(times), 5)
    assert decision['promote_candidate'] is promote
    assert decision['paired_wins'] == wins and decision['required_paired_wins'] == 4


@pytest.mark.parametrize('corruption', ['missing', 'duplicate', 'diagnostic', 'ids', 'roundtrip', 'failed'])
def test_paired_invalid_results_stop(corruption):
    baseline, candidate = paired_rows([100] * 5), paired_rows([90] * 5)
    if corruption == 'missing':
        candidate.pop()
    elif corruption == 'duplicate':
        candidate[0]['repeat'] = 2
    elif corruption == 'diagnostic':
        candidate[0]['diagnostic'] = True
    elif corruption == 'ids':
        candidate[0]['ids_match_reference'] = False
    elif corruption == 'roundtrip':
        candidate[0]['roundtrip_passed'] = False
    else:
        candidate[0]['status'] = 'unavailable'
    with pytest.raises(ValueError):
        benchmark.paired_decision(baseline, candidate, 5)


@pytest.mark.parametrize('improve', [True, False])
def test_refinement_fifty_trials_and_independent_pair_selection(improve):
    args = benchmark.parse_args(['--suite', 'local-refinement', '--diagnostics'])
    calls = []
    def trial(stage, spec, repeat=0, diagnostic=False):
        seconds = 12.
        if spec['decode_block_size'] == 131072:
            seconds -= .1
        if spec['partitions_per_worker'] == 8:
            seconds -= 1.
        if stage == 'paired-candidate' and not improve:
            seconds = 11.9  # Search winner does not reproduce a >=3% gain.
        row = {**spec, 'stage': stage, 'repeat': repeat, 'diagnostic': diagnostic, 'total_seconds': seconds,
               'status': 'ok', 'roundtrip_passed': True, 'ids_match_reference': True}
        calls.append(row)
        return row
    selected, selections, confirmation = benchmark.run_refinement(args, trial)
    assert len([r for r in calls if not r['diagnostic']]) == 50
    assert len([r for r in calls if r['diagnostic']]) == 2
    assert selections['paired_confirmation']['promote_candidate'] is improve
    assert selected['partitions_per_worker'] == (8 if improve else 4)
    assert selected['decode_block_size'] == (131072 if improve else 262144)
    assert confirmation['repeat'] == 5
    assert confirmation['stage'] == ('paired-candidate' if improve else 'paired-baseline')
    pair = [r for r in calls if r['stage'].startswith('paired-')]
    assert all({r['stage'] for r in pair[i:i + 2]} == {'paired-baseline', 'paired-candidate'}
               for i in range(0, 10, 2))


@pytest.mark.parametrize('extra', [['--workers', '8'], ['--repeats', '3'],
                                  ['--decode-block-sizes', '131072'], ['--partition-multipliers', '8']])
def test_cloud_refinement_rejects_missing_controls_or_insufficient_repeats(extra):
    args = benchmark.parse_args(['--suite', 'local-refinement', *extra])
    with pytest.raises(ValueError, match='refinement|Refinement'):
        benchmark.execute(args)


def test_cloud_refinement_insufficient_cpus_stops_before_output(tmp_path, monkeypatch):
    out = tmp_path / 'not-created'
    args = benchmark.parse_args(['--suite', 'local-refinement', '--output-dir', str(out)])
    monkeypatch.setattr(benchmark.os, 'sched_getaffinity', lambda pid: set(range(8)), raising=False)
    monkeypatch.setattr(benchmark, 'restrict_environment', lambda: None)
    with pytest.raises(ValueError, match='no automatic downgrade'):
        benchmark.execute(args)
    assert not out.exists()


def test_same_configuration_never_claims_an_improvement():
    args = benchmark.parse_args(['--suite', 'local-refinement', '--decode-block-sizes', '262144',
                                 '--partition-multipliers', '4'])
    def trial(stage, spec, repeat=0, diagnostic=False):
        # Force an apparent timing gain for identical configurations in the independent rounds.
        return {**spec, **paired_rows([90 if stage == 'paired-candidate' else 100])[0],
                'stage': stage, 'repeat': repeat}
    selected, selections, confirmation = benchmark.run_refinement(args, trial)
    decision = selections['paired_confirmation']
    assert decision['paired_wins'] == 5 and not decision['promote_candidate']
    assert decision['reason'] == 'candidate_matches_current_configuration'
    assert selected == selections['baseline'] and confirmation['stage'] == 'paired-baseline'


def test_small_refinement_pipeline_and_diagnostic_outputs(tmp_path):
    bpe = tokenizer()
    prefix = str(tmp_path / 'tokenizer')
    bpe.save(prefix)
    valid = tmp_path / 'valid.txt'
    valid.write_bytes(sample().encode())
    out = tmp_path / 'refined'
    script = Path(benchmark.__file__).resolve()
    subprocess.run([sys.executable, str(script), '--suite', 'local-refinement', '--development', '--no-wandb',
                    '--tokenizer-prefix', prefix, '--valid-file', str(valid), '--separator', '<x>',
                    '--workers', '1', '--repeats', '1', '--partition-multipliers', '1', '4',
                    '--decode-block-sizes', '16', '262144', '--diagnostics', '--output-dir', str(out)],
                   check=True, capture_output=True, text=True, timeout=60)
    selection = json.loads((out / 'selection.json').read_text())
    assert selection['suite'] == 'local-refinement' and selection['completed']
    rows = json.loads((out / 'results.json').read_text())
    assert rows[0]['stage'] == 'reference' and len(rows) == 9
    confirmation = next(row for row in rows if row['trial_id'] == selection['confirmation_trial'])
    assert confirmation['stage'].startswith('paired-') and confirmation['repeat'] == 1
    assert selection['confirmation_seconds'] == confirmation['total_seconds']
    diagnostics = [row for row in rows if row['diagnostic']]
    assert len(diagnostics) == 2
    assert all(row['decode_diagnostics']['utf8_seconds'] >= 0 for row in diagnostics)
    assert all(row['pipeline_diagnostics']['join_seconds'] >= 0 for row in diagnostics)
    assert all(row['ids_match_reference'] and row['roundtrip_passed'] for row in rows[1:])
    assert '分项诊断' in (out / 'report.md').read_text()
    assert len(list(out.glob('trial-*-profiles/parent.prof'))) == 2
    assert not list(out.glob('reference-*'))
