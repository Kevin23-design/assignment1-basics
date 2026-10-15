"""Fixed G/H/I configuration, deployment and report checks without GPU training."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import launch_context_lr_experiments as launcher
import train_context_lr_experiment as runner
from transformer_module import get_lr_cosine_schedule


def configs(tmp_path):
    return {case: runner.case_config({}, case, list(range(16)), 'study', tmp_path / case)
            for case in ('G', 'H', 'I')}


def test_best_baseline_and_single_variable_comparison(tmp_path):
    g, h, i = configs(tmp_path).values()
    assert {key for key in g if g[key] != h[key]} == {'experiment_case', 'output_dir', 'learning_rate'}
    baseline = runner.case_config({}, 'B', list(range(16)), 'study', tmp_path / 'G')
    assert {key for key in g if g[key] != baseline[key]} == {'experiment_case', 'experiment_suite'}
    assert g['learning_rate'] == 6e-4 and h['learning_rate'] == 1e-3
    assert (g['context_length'], g['batch_size'], g['num_heads'], g['d_ff']) == (512, 64, 8, 1408)
    assert (i['context_length'], i['batch_size'], i['num_heads'], i['d_ff']) == (256, 128, 16, 1344)
    assert i['optimizer_betas'] == (.9, .95) and i['weight_decay'] == .01
    assert i['learning_rate'] == .01 and i['min_learning_rate'] == .001 and i['seed'] == 1337
    assert i['warmup_steps'] == i['cosine_steps'] // 10 == 6500
    assert i['min_learning_rate'] == i['learning_rate'] * .1
    assert i['reference_adaptation']['revision'] == 'c7bd4a373c1c56fc6cb02cc3f848c13d02a2b825'
    assert i['reference_adaptation']['entry_url'].endswith('/train.py')
    for config in (g, h, i):
        assert config['tokens_per_step'] == 32768 and config['cosine_target_tokens'] == 2129920000
        assert config['validation_context_length'] == config['context_length']
        assert config['num_layers'] == 4 and config['d_model'] == 512
        assert config['max_grad_norm'] == 1.0 and config['precision'] == 'bf16'
        assert config['max_train_seconds'] == 21600 and config['reserve_seconds'] == 600
        assert config['bpe_encode_workers'] == 16 and config['bpe_worker_cache_capacity'] == 1048576
        assert config['resume'] is None and config['previous_elapsed_seconds'] == 0
    assert g['warmup_target_tokens'] == 16384000 and i['warmup_target_tokens'] == 212992000


@pytest.mark.parametrize('case', ['G', 'H', 'I'])
def test_schedule_boundaries(tmp_path, case):
    c = configs(tmp_path)[case]
    def lr(step):
        return get_lr_cosine_schedule(step, c['learning_rate'], c['min_learning_rate'],
                                      c['warmup_steps'], c['cosine_steps'])
    assert lr(0) == 0
    assert lr(c['warmup_steps'] - 1) < lr(c['warmup_steps'])
    assert lr(c['warmup_steps']) == pytest.approx(c['learning_rate'])
    assert lr(64999) >= c['min_learning_rate']
    assert lr(65000) == pytest.approx(c['min_learning_rate'])
    assert lr(90000) == pytest.approx(c['min_learning_rate'])


def test_resources_cli_and_gpu_mapping(tmp_path):
    args = launcher.parse_args(['--suite', 'best-lr-reference'])
    assert args.gpus == [1, 2, 3] and args.output_dir == 'outputs/best-lr-reference-3gpu'
    assert runner.SUITES[args.suite]['disk_bytes'] == 75_000_000_000
    allowed = list(range(100, 180, 1))
    allocation = launcher.allocate_cpus(reversed(allowed), args.suite)
    assert allocation == {'G': allowed[16:32], 'H': allowed[32:48], 'I': allowed[48:64]}
    assert len(set().union(*map(set, allocation.values()))) == 48
    for gpu, case in zip(args.gpus, allocation, strict=True):
        command, env = launcher.case_command(case, allocation[case], 'study', tmp_path / case, gpu)
        assert env['CUDA_VISIBLE_DEVICES'] == str(gpu)
        assert command[command.index('--suite') + 1] == args.suite
        assert command[command.index('--case') + 1] == case
    with pytest.raises(ValueError, match='64'):
        launcher.allocate_cpus(range(63), args.suite)
    for gpus in (['1', '2'], ['1', '1', '3'], ['-1', '2', '3']):
        with pytest.raises(SystemExit):
            launcher.parse_args(['--suite', args.suite, '--gpus', *gpus])
    with pytest.raises(SystemExit):
        launcher.parse_args(['--suite', args.suite, '--learning-rate', '.1'])


def test_disk_preflight_stops_before_any_process(tmp_path, monkeypatch):
    args = launcher.parse_args(['--suite', 'best-lr-reference', '--output-dir', str(tmp_path / 'outputs/new')])
    monkeypatch.setattr(launcher, 'ROOT', tmp_path)
    monkeypatch.setattr(launcher.os, 'sched_getaffinity', lambda pid: set(range(64)), raising=False)
    monkeypatch.setattr(launcher.os, 'sched_setaffinity', lambda *args: None, raising=False)
    monkeypatch.setenv('WANDB_ENTITY', 'student')
    monkeypatch.setenv('STUDENT_ID', 'student')
    monkeypatch.setattr(Path, 'is_file', lambda path: True)
    monkeypatch.setattr(launcher.shutil, 'disk_usage', lambda path: SimpleNamespace(free=74_999_999_999))
    def no_process(*args, **kwargs):
        pytest.fail('Preflight failure must not launch a process')
    monkeypatch.setattr(launcher.subprocess, 'Popen', no_process)
    with pytest.raises(ValueError, match='75 GB'):
        launcher.launch(args)
    assert not Path(args.output_dir).exists()


@pytest.mark.parametrize('failed', [None, 'G', 'I'])
def test_reports_split_validation_protocols_and_isolate_failures(tmp_path, monkeypatch, failed):
    monkeypatch.setattr(launcher, 'source_fingerprints', lambda: {'source': 'same'})
    allocation = launcher.allocate_cpus(range(64), 'best-lr-reference')
    manifest = {'study_id': 'study', 'suite': 'best-lr-reference', 'source_fingerprints': {'source': 'same'},
                'cases': {case: {'gpu': gpu, 'cpu_ids': allocation[case]}
                          for gpu, case in enumerate(allocation, 1)}}
    codes = {}
    for case in allocation:
        directory = tmp_path / case
        directory.mkdir()
        codes[case] = 1 if case == failed else 0
        summary = {'source_changed': [], 'val/ppl': {'G': 40., 'H': 39., 'I': 30.}[case],
                   'val/token_count': 66401097, 'time/total_seconds': 21100., 'time/within_6h': True}
        result = {'status': 'oom' if case == failed else 'completed', 'source_recorded': True,
                  'summary': summary, 'wandb_url': 'https://example.com/' + case}
        (directory / 'result.json').write_text(json.dumps(result))
    result = launcher.collect(tmp_path, manifest, codes)
    assert [r['eligible'] for r in result['results']] == [case != failed for case in allocation]
    assert result['comparison_groups'] == {'512': ['G', 'H'], '256': ['I']}
    assert result['reference_adaptation'] == runner.REFERENCE_ADAPTATION
    with (tmp_path / 'results.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert [r['num_heads'] for r in rows] == ['8', '8', '16']
    assert [r['min_learning_rate'] for r in rows] == ['3e-05', '3e-05', '0.001']
    assert [r['seed'] for r in rows] == ['42', '42', '1337']
    report = (tmp_path / 'report.md').read_text()
    gh, i = report.split('## 512 窗口：G/H 主对照')[1].split('## 256 窗口：I 整套配置探索')
    assert '| G |' in gh and '| H |' in gh and '| I |' not in gh
    assert '| I |' in i and '| G |' not in i and '| H |' not in i
    assert '有效配置按完整验证 PPL 排序' not in report
    assert '六小时适配版' in report and '不是完整复现' in report and runner.REFERENCE_REVISION in report
    assert '6500' in report and '40.18098' in report
    monkeypatch.setattr(launcher, 'source_fingerprints', lambda: {'source': 'changed'})
    changed = launcher.collect(tmp_path, manifest, codes)
    assert not any(r['eligible'] for r in changed['results'])
