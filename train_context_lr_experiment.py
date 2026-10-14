"""One fixed context/LR case; start timing before heavy imports or setup."""
import time

PROGRAM_STARTED_UNIX = time.time()
PROGRAM_STARTED_CLOCK = time.perf_counter()

# ruff: noqa: E402
import argparse
from contextlib import contextmanager
import json
import hashlib
import math
import os
from pathlib import Path
import signal

ROOT = Path(__file__).resolve().parent
CASES = {
    'A': {'context_length': 512, 'batch_size': 64, 'learning_rate': 3e-4},
    'B': {'context_length': 512, 'batch_size': 64, 'learning_rate': 6e-4},
    'C': {'context_length': 256, 'batch_size': 128, 'learning_rate': 3e-4},
    'D': {'context_length': 256, 'batch_size': 128, 'learning_rate': 6e-4},
}
THREAD_VARS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')


def save_result(path, result):
    path = Path(path)
    partial = Path(str(path) + '.partial')
    partial.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    partial.replace(path)


def prepare_environment(cpu_ids, visible_gpu):
    if len(cpu_ids) != 16 or len(set(cpu_ids)) != 16:
        raise ValueError('Each case requires exactly 16 distinct logical CPUs')
    if not hasattr(os, 'sched_getaffinity'):
        raise RuntimeError('Formal experiments require Linux CPU affinity')
    if not set(cpu_ids) <= os.sched_getaffinity(0):
        raise ValueError('Requested CPUs are outside the inherited allocation')
    if not visible_gpu or ',' in visible_gpu or visible_gpu == '-1':
        raise ValueError('Each case requires exactly one visible GPU')
    os.sched_setaffinity(0, set(cpu_ids))
    if os.sched_getaffinity(0) != set(cpu_ids):
        raise RuntimeError('CPU affinity was not applied')
    os.environ['CUDA_VISIBLE_DEVICES'] = visible_gpu
    for name in THREAD_VARS:
        os.environ[name] = '16'


def case_config(base, case, cpu_ids, study_id, out):
    config = dict(base)
    config.update(CASES[case])
    config.update({
        'experiment_case': case, 'experiment_series': study_id,
        'context_lr_cpu_ids': cpu_ids, 'output_dir': str(Path(out).resolve()),
        'train_file': '/data/assignment1-data/owt_train.txt',
        'valid_file': '/data/assignment1-data/owt_valid.txt',
        'bpe_prefix': str(ROOT / 'outputs/e9275a3b4a7a/tokenizer'),
        'device': 'cuda:0', 'precision': 'bf16', 'num_layers': 4, 'd_model': 512,
        'num_heads': 8, 'd_ff': 1408, 'rope_theta': 10000, 'seed': 42,
        'eval_batch_size': 16, 'validation_context_length': CASES[case]['context_length'],
        'expected_validation_targets': 66401097,
        'min_learning_rate': 3e-5, 'warmup_steps': 500, 'cosine_steps': 65000,
        'optimizer_betas': (0.9, 0.999), 'optimizer_eps': 1e-8,
        'weight_decay': .1, 'max_grad_norm': 1.0,
        'bpe_encode_workers': 16, 'bpe_worker_cache_capacity': 1048576,
        'bpe_cache_policy': 'clear_all', 'bpe_split_special_token': '<|endoftext|>',
        'max_train_seconds': 21600, 'reserve_seconds': 600,
        'resume': None, 'resume_from_run': None, 'previous_elapsed_seconds': 0.,
    })
    return config


def classify_error(error):
    if isinstance(error, KeyboardInterrupt):
        return 'interrupted'
    if isinstance(error, TimeoutError):
        return 'timeout'
    message = str(error).lower()
    if 'out of memory' in message:
        return 'oom'
    if '训练损失异常' in message or '非有限' in message or 'nan' in message:
        return 'nonfinite_loss'
    if '验证' in message:
        return 'validation_failed'
    return 'failed'


def eligible_result(result):
    summary = result.get('summary', {})
    ppl = summary.get('val/ppl')
    seconds = summary.get('time/total_seconds')
    return (result.get('status') == 'completed' and result.get('source_recorded', False)
            and summary.get('source_changed') == []
            and summary.get('val/token_count') == 66401097
            and isinstance(ppl, (int, float)) and math.isfinite(ppl) and ppl > 0
            and isinstance(seconds, (int, float)) and math.isfinite(seconds) and 0 <= seconds <= 21600
            and summary.get('time/within_6h') is True)


def execute(args):
    out = Path(args.output_dir).resolve()
    if not out.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Output must be inside the personal project outputs directory')
    if not args.study_id or not args.study_id.replace('-', '').replace('_', '').isalnum():
        raise ValueError('Study ID must contain letters, numbers, hyphens or underscores')
    prepare_environment(args.cpu_ids, os.environ.get('CUDA_VISIBLE_DEVICES', ''))
    out.mkdir(parents=True, exist_ok=False)
    result = {'case': args.case, 'study_id': args.study_id, 'gpu_visible': os.environ['CUDA_VISIBLE_DEVICES'],
              'cpu_ids': args.cpu_ids, 'status': 'initializing', 'summary': {}, 'source_recorded': False}
    save_result(out / 'result.json', result)

    def interrupted(signum, frame):
        raise KeyboardInterrupt('Experiment interrupted')

    signal.signal(signal.SIGTERM, interrupted)
    try:
        # Heavy imports happen after time origin, GPU visibility and CPU allocation.
        import transformer_main as trainer
        from wandb_record import experiment
        torch = trainer.torch
        trainer.PROGRAM_STARTED_CLOCK = PROGRAM_STARTED_CLOCK
        trainer.PROGRAM_STARTED_UNIX = PROGRAM_STARTED_UNIX
        trainer.CONFIG = case_config(trainer.CONFIG, args.case, args.cpu_ids, args.study_id, out)
        torch.set_num_threads(16)
        torch.set_num_interop_threads(1)
        torch.cuda.set_device(0)
        properties = torch.cuda.get_device_properties(0)
        torch.cuda.reset_peak_memory_stats(0)
        trainer.CONFIG.update({'gpu_name': properties.name, 'gpu_capacity_bytes': properties.total_memory,
                               'physical_gpu_visible': os.environ['CUDA_VISIBLE_DEVICES']})
        result['config'] = dict(trainer.CONFIG)
        save_result(out / 'result.json', result)

        @contextmanager
        def recorded_experiment(task, config):
            with experiment('transformer-context-lr-exploration', config, root=ROOT) as run:
                result['source_recorded'] = True
                run.name = f"{os.environ['STUDENT_ID']}-{args.study_id}-{args.case}"
                run.group = f"{os.environ['STUDENT_ID']}-{args.study_id}"
                result.update({'run_id': run.id, 'wandb_url': run.url, 'status': 'running'})
                save_result(out / 'result.json', result)
                try:
                    yield run
                except BaseException as error:
                    run.summary.update({'experiment/status': classify_error(error), 'experiment/error': str(error)})
                    raise
                else:
                    run.summary['experiment/status'] = 'completed'
                finally:
                    # Capture while the run is live; recorder finishes it on exit.
                    changed = []
                    frozen_root = ROOT / 'outputs' / run.id / 'source'
                    for frozen in frozen_root.rglob('*'):
                        if not frozen.is_file():
                            continue
                        relative = frozen.relative_to(frozen_root)
                        actual = ROOT / relative
                        if not actual.is_file() or hashlib.sha256(actual.read_bytes()).digest() != hashlib.sha256(
                            frozen.read_bytes()
                        ).digest():
                            changed.append(str(relative))
                    run.summary['source_changed'] = changed
                    result['summary'] = dict(run.summary)

        trainer.experiment = recorded_experiment
        trainer.main()
        result['status'] = 'completed'
    except BaseException as error:
        result.update({'status': classify_error(error), 'error': repr(error)})
        raise
    finally:
        result['eligible'] = eligible_result(result)
        save_result(out / 'result.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=list(CASES), required=True)
    parser.add_argument('--cpu-ids', type=int, nargs=16, required=True)
    parser.add_argument('--study-id', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    execute(args)


if __name__ == '__main__':
    main()
