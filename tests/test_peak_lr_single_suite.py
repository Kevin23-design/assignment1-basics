"""J uses the existing training path; only small CPU fixtures and fake launches."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import launch_context_lr_experiments as launcher
import train_context_lr_experiment as runner
from transformer_module import get_lr_cosine_schedule


def test_j_only_changes_peak_lr_from_h(tmp_path):
    h, j = (runner.case_parameters(case) for case in ('H', 'J'))
    assert {k for k in h if h[k] != j[k]} == {'learning_rate'}
    assert h['learning_rate'] == 1e-3 and j['learning_rate'] == 2e-3
    c = runner.case_config({'resume': 'stale'}, 'J', list(range(16)), 'study', tmp_path)
    assert c['experiment_suite'] == 'peak-lr-single'
    assert (c['num_layers'], c['d_model'], c['num_heads'], c['d_ff']) == (4, 512, 8, 1408)
    assert (c['context_length'], c['validation_context_length'], c['batch_size']) == (512, 512, 64)
    assert c['tokens_per_step'] == 32768 and c['expected_validation_targets'] == 66401097
    assert c['max_train_seconds'] == 21600 and c['reserve_seconds'] == 600
    assert c['optimizer_betas'] == (.9, .999) and c['optimizer_eps'] == 1e-8
    assert c['weight_decay'] == .1 and c['max_grad_norm'] == 1.
    assert c['seed'] == 42 and c['precision'] == 'bf16' and c['rope_theta'] == 10000
    assert c['min_learning_rate'] == 3e-5 and c['eval_batch_size'] == 16
    assert c['resume'] is None and c['previous_elapsed_seconds'] == 0
    assert c['bpe_encode_workers'] == 16 and c['bpe_worker_cache_capacity'] == 1048576
    assert c['historical_reference']['run_id'] == '2cbae6a7f35e'
    def lr(step):
        return get_lr_cosine_schedule(step, c['learning_rate'], c['min_learning_rate'],
                                      c['warmup_steps'], c['cosine_steps'])
    assert lr(0) == 0 and lr(499) < lr(500) == .002
    assert lr(65000) == lr(90000) == 3e-5
    assert c['warmup_target_tokens'] == 16384000 and c['cosine_target_tokens'] == 2129920000


def test_single_cpu_requirements_and_legacy_partitions():
    args = launcher.parse_args(['--suite', 'peak-lr-single'])
    assert args.gpus == [1] and args.output_dir == 'outputs/peak-lr-2e3-single'
    assert runner.SUITES[args.suite]['disk_bytes'] == 25_000_000_000
    allowed = list(range(3, 51, 2))
    assert launcher.allocate_cpus(reversed(allowed), args.suite) == {'J': allowed[:16]}
    assert launcher.allocate_cpus(allowed[:16], args.suite) == {'J': allowed[:16]}
    with pytest.raises(ValueError, match='16 allowed'):
        launcher.allocate_cpus(allowed[:15], args.suite)
    for suite in ('context-lr', 'optimizer-batch', 'best-lr-reference'):
        with pytest.raises(ValueError, match='64 allowed'):
            launcher.allocate_cpus(range(63), suite)
    for flags in (['--gpus', '1', '2'], ['--gpus', '-1'], ['--batch-size', '128']):
        with pytest.raises(SystemExit):
            launcher.parse_args(['--suite', args.suite, *flags])


def configure_launcher(tmp_path, monkeypatch, free):
    args = launcher.parse_args(['--suite', 'peak-lr-single', '--output-dir', str(tmp_path / 'outputs/J')])
    monkeypatch.setattr(launcher, 'ROOT', tmp_path)
    monkeypatch.setattr(launcher.os, 'sched_getaffinity', lambda pid: set(range(20, 36)), raising=False)
    assignments = []
    monkeypatch.setattr(launcher.os, 'sched_setaffinity', lambda pid, cpus: assignments.append(cpus), raising=False)
    monkeypatch.setenv('WANDB_ENTITY', 'student')
    monkeypatch.setenv('STUDENT_ID', 'student')
    monkeypatch.setattr(Path, 'is_file', lambda path: True)
    monkeypatch.setattr(launcher.shutil, 'disk_usage', lambda path: SimpleNamespace(free=free))
    monkeypatch.setattr(launcher, 'source_fingerprints', lambda: {'source': 'same'})
    return args, assignments


def test_disk_preflight_has_no_process_side_effects(tmp_path, monkeypatch):
    args, _ = configure_launcher(tmp_path, monkeypatch, 24_999_999_999)
    def forbidden(*args, **kwargs):
        pytest.fail('Must not start a process before disk preflight succeeds')
    monkeypatch.setattr(launcher.subprocess, 'Popen', forbidden)
    with pytest.raises(ValueError, match='25 GB'):
        launcher.launch(args)
    assert not Path(args.output_dir).exists()


def result(ppl=35., status='completed'):
    return {'case': 'J', 'status': status, 'source_recorded': True,
            'summary': {'source_changed': [], 'val/ppl': ppl, 'val/loss': 3.5,
                        'val/token_count': 66401097, 'time/total_seconds': 21150., 'time/within_6h': True,
                        'train/final_step': 60000, 'train/target_tokens': 1966080000,
                        'train/tokens_per_second': 95000., 'gpu/peak_allocated_bytes': 32_000_000_000,
                        'time/train_stop_reason': 'reserve_plus_two_last_steps'}}


@pytest.mark.parametrize('exit_code', [0, 1])
def test_launcher_starts_exactly_one_case_with_gpu_one(tmp_path, monkeypatch, exit_code):
    args, assignments = configure_launcher(tmp_path, monkeypatch, 25_000_000_000)
    calls = []
    class FakeProcess:
        def __init__(self, command, **kwargs):
            calls.append((command, kwargs))
            out = Path(command[-1])
            out.mkdir()
            (out / 'result.json').write_text(json.dumps(result(status='oom' if exit_code else 'completed')))
        def poll(self):
            return exit_code
        def wait(self):
            return exit_code
    monkeypatch.setattr(launcher.subprocess, 'Popen', FakeProcess)
    monkeypatch.setattr(launcher.time, 'sleep', lambda seconds: None)
    assert launcher.launch(args) == exit_code
    assert assignments == [set(range(20, 36))] and len(calls) == 1
    command, settings = calls[0]
    assert settings['env']['CUDA_VISIBLE_DEVICES'] == '1' and settings['start_new_session']
    assert command[command.index('--case') + 1] == 'J'
    assert command[command.index('--suite') + 1] == 'peak-lr-single'
    manifest = json.loads((Path(args.output_dir) / 'manifest.json').read_text())
    assert list(manifest['cases']) == ['J'] and manifest['historical_reference'] == runner.H_HISTORY
    with pytest.raises(FileExistsError):
        launcher.launch(args)


@pytest.mark.parametrize('ppl,status,expected', [(35., 'completed', True), (37., 'completed', False),
                                               (36.14949117609303, 'completed', False),
                                               (30., 'oom', None), (float('nan'), 'completed', None)])
def test_history_report_eligibility_and_source_integrity(tmp_path, monkeypatch, ppl, status, expected):
    directory = tmp_path / 'J'
    directory.mkdir()
    (directory / 'result.json').write_text(json.dumps(result(ppl, status)))
    manifest = {'suite': 'peak-lr-single', 'study_id': 'study', 'source_fingerprints': {'code': 'same'},
                'cases': {'J': {'gpu': 1, 'cpu_ids': list(range(16))}}}
    monkeypatch.setattr(launcher, 'source_fingerprints', lambda: {'code': 'same'})
    aggregate = launcher.collect(tmp_path, manifest, {'J': 0 if status == 'completed' else 1})
    assert aggregate['historical_comparison']['lower_ppl_than_H'] is expected
    assert aggregate['historical_reference']['target_tokens'] == 1800273920
    with (tmp_path / 'results.csv').open() as handle:
        row, = csv.DictReader(handle)
    assert row['learning_rate'] == '0.002' and row['target_tokens'] == '1966080000'
    assert row['tokens_per_second'] == '95000.0' and row['peak_allocated_bytes'] == '32000000000'
    assert row['train_stop_reason'] == 'reserve_plus_two_last_steps'
    report = (tmp_path / 'report.md').read_text()
    assert '36.14949' in report and '2cbae6a7f35e' in report and '2e-3' in report
    assert '单卡负载' in report and '不能全部归因于 LR' in report
    assert 'B/A、D/C' not in report
    if expected is None:
        assert '未自动接入 dev' not in report
    monkeypatch.setattr(launcher, 'source_fingerprints', lambda: {'code': 'changed'})
    changed = launcher.collect(tmp_path, manifest, {'J': 0})
    assert changed['historical_comparison']['lower_ppl_than_H'] is None
    assert not changed['results'][0]['eligible']
