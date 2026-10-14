"""Small CPU checks for E/F, token-aligned schedules, reports and stop diagnostics."""
import csv
import json
from types import SimpleNamespace

import pytest

import launch_context_lr_experiments as launcher
import train_context_lr_experiment as runner
import transformer_main as training
from transformer_module import get_lr_cosine_schedule


def configs(tmp_path):
    return {case: runner.case_config({}, case, list(range(32, 48)), 'study', tmp_path / case)
            for case in ['E', 'F']}


def test_fixed_candidates_and_token_budgets(tmp_path):
    e, f = (configs(tmp_path)[case] for case in ['E', 'F'])
    assert (e['batch_size'], e['learning_rate'], e['warmup_steps'], e['cosine_steps']) == (128, 3e-4, 500, 65000)
    assert (f['batch_size'], f['learning_rate'], f['warmup_steps'], f['cosine_steps']) == (256, 6e-4, 250, 32500)
    for config in [e, f]:
        assert config['optimizer_betas'] == (0.9, .95)
        assert config['weight_decay'] == .01 and config['max_grad_norm'] == 2.0
        assert config['context_length'] == config['validation_context_length'] == 256
        assert config['min_learning_rate'] == 3e-5
        assert config['warmup_target_tokens'] == 16384000
        assert config['cosine_target_tokens'] == 2129920000
        assert config['experiment_suite'] == 'optimizer-batch'
        assert config['resume'] is None and config['previous_elapsed_seconds'] == 0
    assert f['tokens_per_step'] == 2 * e['tokens_per_step']


@pytest.mark.parametrize('step', [0, 100, 498, 500, 1000, 64998, 65000, 66000])
def test_normalized_schedule_at_equal_tokens(tmp_path, step):
    parameters = configs(tmp_path)
    normalized = []
    for case, index in [('E', step), ('F', step // 2)]:
        config = parameters[case]
        lr = get_lr_cosine_schedule(index, config['learning_rate'], config['min_learning_rate'],
                                    config['warmup_steps'], config['cosine_steps'])
        if index < config['warmup_steps']:
            normalized.append(lr / config['learning_rate'])
        else:
            normalized.append((lr - config['min_learning_rate']) /
                              (config['learning_rate'] - config['min_learning_rate']))
    assert normalized[0] == pytest.approx(normalized[1])


def test_old_defaults_and_new_cli_defaults():
    old = launcher.parse_args([])
    assert old.suite == 'context-lr' and old.gpus == [0, 1, 2, 3]
    assert old.output_dir == 'outputs/context-lr-2x2'
    new = launcher.parse_args(['--suite', 'optimizer-batch'])
    assert new.gpus == [2, 3] and new.output_dir == 'outputs/optimizer-batch-2gpu'
    assert runner.SUITES[new.suite]['disk_bytes'] == 50000000000
    assert runner.SUITES[old.suite]['disk_bytes'] == 100000000000
    for args in [['--suite', 'optimizer-batch', '--gpus', '0', '1', '2', '3'],
                 ['--gpus', '2', '3'], ['--suite', 'optimizer-batch', '--gpus', '2', '2']]:
        with pytest.raises(SystemExit):
            launcher.parse_args(args)


def test_rear_cpu_partition_and_gpu_commands(tmp_path):
    allocation = launcher.allocate_cpus(range(10, 90), 'optimizer-batch')
    assert allocation == {'E': list(range(42, 58)), 'F': list(range(58, 74))}
    assert not set(allocation['E']) & set(allocation['F'])
    original = launcher.allocate_cpus(range(10, 90))
    assert allocation['E'] == original['C'] and allocation['F'] == original['D']
    with pytest.raises(ValueError, match='64'):
        launcher.allocate_cpus(range(63), 'optimizer-batch')
    for case, gpu in [('E', 2), ('F', 3)]:
        command, environment = launcher.case_command(case, allocation[case], 'study', tmp_path / case, gpu)
        assert environment['CUDA_VISIBLE_DEVICES'] == str(gpu)
        assert command[command.index('--suite') + 1] == 'optimizer-batch'
        assert command[command.index('--case') + 1] == case


def test_optimizer_report_fields_and_failure_isolation(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, 'source_fingerprints', lambda: {'source': 'same'})
    manifest = {'study_id': 'study', 'suite': 'optimizer-batch', 'source_fingerprints': {'source': 'same'},
                'cases': {case: {'gpu': gpu, 'cpu_ids': list(range(32 + i * 16, 48 + i * 16))}
                          for i, (case, gpu) in enumerate([('E', 2), ('F', 3)])}}
    for case in ['E', 'F']:
        path = tmp_path / case
        path.mkdir()
        summary = {'source_changed': [], 'val/ppl': 40., 'val/token_count': 66401097,
                   'time/total_seconds': 21000, 'time/within_6h': True,
                   'time/train_stop_reason': 'reserve_plus_two_last_steps',
                   'time/train_stop_remaining_seconds': 600.5,
                   'time/train_stop_last_step_seconds': .3,
                   'time/train_stop_step_safety_seconds': .6}
        result = {'status': 'completed' if case == 'E' else 'oom', 'source_recorded': True,
                  'summary': summary}
        (path / 'result.json').write_text(json.dumps(result))
    result = launcher.collect(tmp_path, manifest, {'E': 0, 'F': 1})
    assert [r['eligible'] for r in result['results']] == [True, False]
    with (tmp_path / 'results.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert [r['tokens_per_step'] for r in rows] == ['32768', '65536']
    assert [r['cosine_steps'] for r in rows] == ['65000', '32500']
    assert all(r['optimizer_beta2'] == '0.95' and r['weight_decay'] == '0.01' for r in rows)
    assert rows[0]['train_stop_step_safety_seconds'] == '0.6'
    report = (tmp_path / 'report.md').read_text()
    assert 'E=32,768' in report and 'F=65,536' in report
    assert '每步均为 32,768' not in report
    assert '45.18785' in report and '41.75639' in report


def test_training_stop_diagnostics_and_dynamic_tokens(tmp_path, monkeypatch):
    from bpe import BPE
    prefix = str(tmp_path / 'tokenizer')
    BPE(dict(enumerate(bytes([i]) for i in range(256))), [], ['<|endoftext|>']).save(prefix)
    source = tmp_path / 'input.txt'
    source.write_text('abcd <|endoftext|> efgh\n' * 30)
    finished, logs = [], []
    run = SimpleNamespace(id='small', config={}, summary={}, log=logs.append)
    real_step = training.AdamW.step
    def step(optimizer):
        result = real_step(optimizer)
        finished.append(True)
        return result
    monkeypatch.setattr(training.time, 'perf_counter', lambda: float(len(finished)))
    monkeypatch.setattr(training, 'PROGRAM_STARTED_CLOCK', 0.)
    monkeypatch.setattr(training.AdamW, 'step', step)
    config = {**training.CONFIG, 'bpe_prefix': prefix, 'train_file': str(source), 'valid_file': str(source),
              'output_dir': str(tmp_path / 'out'), 'device': 'cpu', 'precision': 'fp32',
              'bpe_encode_workers': 1, 'd_model': 16, 'num_layers': 1, 'num_heads': 2, 'd_ff': 32,
              'context_length': 8, 'batch_size': 3, 'eval_batch_size': 3, 'reserve_seconds': 1,
              'deadline_clock': 6, 'previous_elapsed_seconds': 0}
    result = training.train_model(config, run)
    assert result['step'] == 3 and logs[-1]['train/target_tokens'] == 3 * 3 * 8
    assert run.summary['train/target_tokens'] == 72
    assert run.summary['time/train_stop_reason'] == 'reserve_plus_two_last_steps'
    assert run.summary['time/train_stop_elapsed_seconds'] == 3
    assert run.summary['time/train_stop_remaining_seconds'] == 3
    assert run.summary['time/train_stop_last_step_seconds'] == 1
    assert run.summary['time/train_stop_step_safety_seconds'] == 2
    assert run.summary['time/train_stop_reserve_seconds'] == 1
