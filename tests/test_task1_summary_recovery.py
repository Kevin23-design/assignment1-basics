"""Regression tests for strict W&B summaries and saved-result recovery."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

import benchmark_task1_encode_transport as benchmark
import recover_task1_encode_transport as recovery
from tests.test_task1_encode_transport import history_fixture


class StrictSummary(dict):
    def update(self, mapping):
        super().update(mapping)


def test_publish_uses_mapping_only_summary(tmp_path, monkeypatch):
    _, history = history_fixture(tmp_path)
    (tmp_path / 'manifest.json').write_text(json.dumps({'baseline_snapshot': history['snapshot']}))
    class Artifact:
        def __init__(self, *args, **kwargs):
            pass
        def add_file(self, *args, **kwargs):
            pass
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(Table=lambda **kwargs: kwargs, Artifact=Artifact))
    run = SimpleNamespace(id='test', summary=StrictSummary(), log=lambda _: None,
                          log_artifact=lambda _: None, save=lambda *args, **kwargs: None)
    benchmark.publish(run, tmp_path, [], {'recommended': 'T0', 'integration': {'passed': True},
                                         'snapshot': history['snapshot']})
    assert run.summary['completed'] is True
    assert run.summary['integration_passed'] is True


def test_only_exact_logging_corrections_are_accepted():
    hashes = benchmark.common.source_fingerprints()
    for name, replacements in recovery.FIXES.items():
        source = (benchmark.ROOT / name).read_text()
        for old, new in replacements:
            assert source.count(new) == 1
            source = source.replace(new, old)
        hashes[name] = hashlib.sha256(source.encode()).hexdigest()
    assert set(recovery.check_sources(hashes)) == set(recovery.FIXES)
    hashes['bpe.py'] = '0' * 64
    with pytest.raises(ValueError, match='Non-logging'):
        recovery.check_sources(hashes)


def failed_run(tmp_path):
    args, history = history_fixture(tmp_path)
    args.repeats = 5
    out = tmp_path / 'saved-run'
    out.mkdir()
    rows, schedule = [], []
    snapshot = benchmark.make_snapshot(out / 'plain-source', history['snapshot'], 'T2')
    _, hashes = benchmark.load_history(args)
    def trial(stage, method, repeat=0, diagnostic=False):
        record = {'trial_id': f'trial-{len(rows) + 1:03d}', 'stage': stage, 'method': method,
                  'repeat': repeat, 'diagnostic': diagnostic}
        total = {'T0': 11., 'T1': 10.9, 'T2': 10., 'T3': 1.}[method]
        row = {**record, 'status': 'unavailable' if method == 'T3' else 'ok', 'error': 'capacity',
               'total_seconds': total, 'encode_seconds': total - 4, 'decode_seconds': 3., 'compare_seconds': 1.,
               'token_count': history['expected_tokens'], 'returned_type': 'list', 'cpu_affinity': [],
               'roundtrip_passed': True, 'ids_match_reference': True, 'patching': stage != 'integration-plain'}
        job = {**record, **{k: history[k] for k in ('tokenizer_prefix', 'valid_file', 'reference', 'expected_tokens')},
               'cpus': [], 'workers': 2, 'development': True, 'input_sha256': hashes,
               'baseline_snapshot': history['snapshot'], 'snapshot': snapshot, 'plain': stage == 'integration-plain'}
        benchmark.common.save_json(out / (record['trial_id'] + '.json'), row)
        benchmark.common.save_json(out / (record['trial_id'] + '.job.json'), job)
        rows.append(row)
        schedule.append(record)
        return row
    selected = benchmark.run_stages(args, trial)
    experimental, plain = [], []
    for repeat in range(1, args.repeats + 1):
        experimental.append(trial('integration-experimental', 'T2', repeat))
        plain.append(trial('integration-plain', 'T2', repeat))
    error = repr(TypeError("SummaryDict.update() got an unexpected keyword argument 'completed'"))
    selected.update(**{k: history[k] for k in ('tokenizer_prefix', 'valid_file', 'reference', 'expected_tokens',
                                              'development', 'cpu_affinity')},
                    snapshot=snapshot, input_sha256=hashes, completed=False, ready_for_plain_confirmation=False,
                    integration=benchmark.integration_decision(experimental, plain, args.repeats), error=error)
    files = {'manifest.json': {'arguments': vars(args), 'workers': 2, 'cpu_affinity': [],
                               'input_sha256': hashes, 'baseline_snapshot': history['snapshot'],
                               'source_sha256': benchmark.common.source_fingerprints()},
             'selection.json': selected, 'failure.json': {'completed': False, 'error': error},
             'integrity.json': {'changed_inputs': [], 'source_changed': False},
             'results.json': rows, 'schedule.json': schedule}
    for name, value in files.items():
        benchmark.common.save_json(out / name, value)
    return out


def test_recover_replays_decisions_and_preserves_originals(tmp_path, monkeypatch):
    out = failed_run(tmp_path)
    originals = {p: p.read_bytes() for p in out.glob('*.json')}
    def no_benchmark(*args, **kwargs):
        raise AssertionError('Recovery must never launch a trial')
    monkeypatch.setattr(benchmark, 'launch_job', no_benchmark)
    result = recovery.recover(out)
    assert result['completed'] and result['ready_for_plain_confirmation']
    assert result['recommended'] == 'T2'
    assert not result['recovery']['timings_recomputed']
    assert not result['recovery']['wandb_publication_repaired']
    assert all(p.read_bytes() == raw for p, raw in originals.items())
    assert (out / 'recovered-selection.json').exists()
    with pytest.raises(FileExistsError):
        recovery.recover(out)


def test_recovered_report_works_with_plain_confirmation(tmp_path):
    out = failed_run(tmp_path)
    recovery.recover(out)
    args = benchmark.parse_args([
        '--phase', 'plain-confirmation', '--development', '--no-wandb',
        '--selection-report', str(out / 'recovered-selection.json'),
        '--output-dir', str(tmp_path / 'confirmed'),
    ])
    benchmark.previous.plain_confirmation(args)
    result = json.loads((Path(args.output_dir) / 'confirmation.json').read_text())
    assert result['status'] == 'ok' and result['patching'] is False
    assert result['ids_match_reference'] and result['roundtrip_passed']


@pytest.mark.parametrize('problem', ['wrong_error', 'integrity', 'input', 'trial', 'selection', 'gate'])
def test_recovery_refuses_invalid_or_failed_results(tmp_path, problem):
    out = failed_run(tmp_path)
    if problem == 'input':
        history = json.loads((out / 'selection.json').read_text())
        Path(history['valid_file']).write_text('changed')
    else:
        name = {'wrong_error': 'failure.json', 'integrity': 'integrity.json', 'trial': 'trial-001.json',
                'selection': 'selection.json', 'gate': 'selection.json'}[problem]
        path = out / name
        data = json.loads(path.read_text())
        if problem == 'wrong_error':
            data['error'] = 'RuntimeError("worker failed")'
        elif problem == 'integrity':
            data['source_changed'] = True
        elif problem == 'trial':
            data['ids_match_reference'] = False
        elif problem == 'selection':
            data['recommended'] = 'T1'
        else:
            data['integration']['passed'] = False
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        recovery.recover(out)
    assert not (out / 'recovered-selection.json').exists()
