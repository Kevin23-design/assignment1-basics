"""Resource/config/observability checks without cloud GPU use or W&B upload."""
from contextlib import contextmanager
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import launch_context_lr_experiments as launcher
import train_context_lr_experiment as runner
from transformer_main import validation_batches


@pytest.mark.parametrize('case,context,batch,lr', [('A', 512, 64, 3e-4), ('B', 512, 64, 6e-4),
                                                  ('C', 256, 128, 3e-4), ('D', 256, 128, 6e-4)])
def test_fixed_config(case, context, batch, lr, tmp_path):
    config = runner.case_config({'resume': 'stale', 'max_train_seconds': 9}, case, list(range(16)), 'study', tmp_path)
    assert (config['context_length'], config['batch_size'], config['learning_rate']) == (context, batch, lr)
    assert context * batch == 32768
    assert config['validation_context_length'] == context
    assert config['min_learning_rate'] == 3e-5
    assert config['warmup_steps'] == 500 and config['cosine_steps'] == 65000
    assert config['max_train_seconds'] == 21600 and config['reserve_seconds'] == 600
    assert config['bpe_encode_workers'] == 16 and config['bpe_worker_cache_capacity'] == 1048576
    assert config['resume'] is None and config['previous_elapsed_seconds'] == 0
    assert config['eval_batch_size'] == 16 and config['expected_validation_targets'] == 66401097


def test_cpu_allocation_and_gpu_mapping(tmp_path):
    allocation = launcher.allocate_cpus(range(10, 90))
    assert len(set().union(*map(set, allocation.values()))) == 64
    assert all(len(ids) == 16 for ids in allocation.values())
    with pytest.raises(ValueError):
        launcher.allocate_cpus(range(63))
    for gpu, case in enumerate(runner.CASES):
        cmd, env = launcher.case_command(case, allocation[case], 'study', tmp_path / case, gpu)
        assert env['CUDA_VISIBLE_DEVICES'] == str(gpu)
        assert cmd[cmd.index('--case') + 1] == case
        assert cmd[-1] == str(tmp_path / case)


def test_environment_precedes_heavy_imports(monkeypatch):
    allowed = set(range(64))
    events = []
    monkeypatch.setattr(runner.os, 'sched_getaffinity', lambda pid: allowed, raising=False)
    def set_affinity(pid, cpus):
        nonlocal allowed
        allowed = set(cpus)
        events.append(cpus)
    monkeypatch.setattr(runner.os, 'sched_setaffinity', set_affinity, raising=False)
    for key in runner.THREAD_VARS + ('CUDA_VISIBLE_DEVICES',):
        monkeypatch.setenv(key, 'before')
    runner.prepare_environment(list(range(16, 32)), '2')
    assert allowed == set(range(16, 32)) and events
    assert runner.os.environ['CUDA_VISIBLE_DEVICES'] == '2'
    assert all(runner.os.environ[key] == '16' for key in runner.THREAD_VARS)
    with pytest.raises(ValueError):
        runner.prepare_environment(list(range(15)), '0')
    with pytest.raises(ValueError):
        runner.prepare_environment(list(range(16, 32)), '0,1')


def completed_result(case='A'):
    return {'case': case, 'status': 'completed', 'source_recorded': True,
            'summary': {'source_changed': [], 'val/ppl': 40.6, 'val/loss': 3.7,
                        'val/token_count': 66401097, 'time/total_seconds': 21000., 'time/within_6h': True}}


@pytest.mark.parametrize('key,value', [('source_changed', ['changed.py']), ('val/token_count', 12),
                                      ('time/total_seconds', 21601), ('val/ppl', float('nan'))])
def test_invalid_result_exclusion(key, value):
    result = completed_result()
    assert runner.eligible_result(result)
    result['summary'][key] = value
    assert not runner.eligible_result(result)


@pytest.mark.parametrize('context', [256, 512])
def test_native_validation_targets(context):
    tokens = np.arange(context * 2 + 29)
    targets = []
    for _, y, mask in validation_batches(tokens, context, 16, 'cpu'):
        targets.extend(y[mask].tolist())
    assert targets == tokens[1:].tolist()


def test_launcher_import_is_stdlib_only():
    result = subprocess.run([sys.executable, '-c',
                             "import sys; import launch_context_lr_experiments; "
                             "assert 'torch' not in sys.modules and 'numpy' not in sys.modules"],
                            check=True, capture_output=True, text=True)
    assert result.returncode == 0


def test_live_run_capture_and_entry_time(monkeypatch, tmp_path):
    root = tmp_path
    output = root / 'outputs' / 'A'
    monkeypatch.setattr(runner, 'ROOT', root)
    monkeypatch.setattr(runner, 'prepare_environment', lambda cpus, gpu: None)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setenv('STUDENT_ID', 'student')
    monkeypatch.setattr(runner.signal, 'signal', lambda *args: None)
    events = []
    cuda = SimpleNamespace(set_device=lambda index: events.append(('device', index)),
                           get_device_properties=lambda index: SimpleNamespace(name='test-gpu', total_memory=80 << 30),
                           reset_peak_memory_stats=lambda index: events.append(('reset', index)))
    torch = SimpleNamespace(cuda=cuda, set_num_threads=lambda n: None, set_num_interop_threads=lambda n: None)
    trainer = SimpleNamespace(CONFIG={}, torch=torch)
    summary = completed_result()['summary'].copy()
    run = SimpleNamespace(id='run', summary=summary, url='https://example.com/run')
    @contextmanager
    def record(task, config, root):
        assert task == 'transformer-context-lr-exploration'
        frozen = root / 'outputs/run/source'
        frozen.mkdir(parents=True)
        (root / 'source.py').write_text('source')
        (frozen / 'source.py').write_text('source')
        yield run
        # Stand-in for an SDK that cannot read Summary after finish.
        run.summary = None
    def main():
        assert trainer.PROGRAM_STARTED_CLOCK == runner.PROGRAM_STARTED_CLOCK
        assert trainer.PROGRAM_STARTED_UNIX == runner.PROGRAM_STARTED_UNIX
        assert trainer.CONFIG['device'] == 'cuda:0'
        with trainer.experiment('ignored', trainer.CONFIG):
            events.append(('training', trainer.CONFIG['experiment_case']))
    trainer.main = main
    monkeypatch.setitem(sys.modules, 'transformer_main', trainer)
    monkeypatch.setitem(sys.modules, 'wandb_record', SimpleNamespace(experiment=record))
    runner.execute(SimpleNamespace(case='A', cpu_ids=list(range(16)), study_id='study', output_dir=str(output)))
    result = json.loads((output / 'result.json').read_text())
    assert result['eligible'] and result['status'] == 'completed'
    assert events == [('device', 0), ('reset', 0), ('training', 'A')]
    assert run.name == 'student-study-A' and run.group == 'student-study'


def test_collect_keeps_failure_isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, 'source_fingerprints', lambda: {'source': 'hash'})
    manifest = {'study_id': 'study', 'source_fingerprints': {'source': 'hash'},
                'cases': {case: {'gpu': i, 'cpu_ids': list(range(i * 16, (i + 1) * 16))}
                          for i, case in enumerate(runner.CASES)}}
    for case in runner.CASES:
        directory = tmp_path / case
        directory.mkdir()
        result = completed_result(case)
        if case == 'B':
            result['status'] = 'oom'
        (directory / 'result.json').write_text(json.dumps(result))
    report = launcher.collect(tmp_path, manifest, {'A': 0, 'B': 1, 'C': 0, 'D': 0})
    assert [r['eligible'] for r in report['results']] == [True, False, True, True]
    assert (tmp_path / 'results.csv').exists() and (tmp_path / 'report.md').exists()
    monkeypatch.setattr(launcher, 'source_fingerprints', lambda: {'source': 'changed'})
    report = launcher.collect(tmp_path, manifest, {'A': 0, 'B': 1, 'C': 0, 'D': 0})
    assert not any(r['eligible'] for r in report['results'])


def test_stop_includes_nested_encoding_sessions(monkeypatch):
    calls = []
    class Process:
        pid = 101
        stopped = False
        def poll(self):
            return 0 if self.stopped else None
        def wait(self):
            calls.append('wait')
    process = Process()
    monkeypatch.setattr(launcher, 'descendants', lambda pid: {101, 202, 203})
    monkeypatch.setattr(launcher.os, 'getpgid', lambda pid: 202 if pid == 203 else pid)
    monkeypatch.setattr(launcher.os, 'getpgrp', lambda: 999)
    def kill(group, sig):
        calls.append((group, sig))
        process.stopped = True
    monkeypatch.setattr(launcher.os, 'killpg', kill)
    launcher.stop_processes({'A': process})
    assert (101, launcher.signal.SIGTERM) in calls and (202, launcher.signal.SIGTERM) in calls
    assert 'wait' in calls and not any(isinstance(c, tuple) and c[0] == 999 for c in calls)
