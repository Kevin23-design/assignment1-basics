"""Cloud-only streaming cache/CPU exploration, separate from formal Transformer."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time

from parallel_bpe_encoding import (
    affinity, compare_roundtrip, compare_tokens, encode_parallel, restrict_environment, validate_tokenizer,
)
from bpe import BPE

ROOT = Path(__file__).resolve().parent
COUNTS = {'valid': 66401098, 'train': 2727120452}
SOURCE_SKIP = {'.git', '.venv', 'venv', 'outputs', 'wandb', 'data', 'datasets',
               'checkpoints', '__pycache__', '.pytest_cache'}


def save_json(path, value):
    path = Path(path)
    temporary = Path(str(path) + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def sha256(path):
    value = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def source_files():
    paths = []
    for folder, directories, names in os.walk(ROOT):
        directories[:] = sorted(d for d in directories if d not in SOURCE_SKIP)
        for name in sorted(names):
            path = Path(folder) / name
            if path.suffix in {'.py', '.sh', '.ipynb', '.cu', '.c', '.cpp', '.h', '.rs'} or name in {
                'pyproject.toml', 'uv.lock', '.python-version'
            }:
                paths.append(path)
    return paths


def fingerprint(paths):
    return {str(Path(p).resolve()): sha256(p) for p in paths}


def assert_fingerprints(expected):
    changed = [path for path, digest in expected.items() if not Path(path).is_file() or sha256(path) != digest]
    if changed:
        raise ValueError(f'Inputs/source changed: {changed}')


def shuffled_schedule(values, repeats, seed):
    rng = random.Random(seed)
    schedule = []
    for repeat in range(1, repeats + 1):
        order = list(values)
        rng.shuffle(order)
        schedule.extend({'value': value, 'repeat': repeat} for value in order)
    return schedule


def grouped_medians(rows, key):
    values = sorted({row[key] for row in rows if row.get('status') == 'ok' and not row['diagnostic']})
    return {value: statistics.median(row['total_seconds'] for row in rows
                                   if row.get('status') == 'ok' and not row['diagnostic'] and row[key] == value)
            for value in values}


def choose_smallest(medians, tolerance):
    if not medians:
        raise ValueError('No correct/available configurations')
    fastest = min(medians.values())
    return min(value for value, seconds in medians.items() if seconds <= fastest * (1 + tolerance))


def choose_workers(medians):
    choice = choose_smallest(medians, 0.05)
    if choice == 16 and 8 in medians and (medians[8] - medians[16]) / medians[8] < 0.05:
        return 8
    return choice


def summary_files(out, results):
    fields = ['stage', 'repeat', 'diagnostic', 'status', 'workers', 'cache_capacity', 'split_seconds',
              'encode_seconds', 'concatenate_seconds', 'total_seconds', 'input_bytes', 'token_count',
              'tokens_per_second', 'MiB_per_second', 'parent_peak_rss', 'worker_peak_rss_max',
              'worker_peak_rss_sum', 'id_comparison_passed', 'roundtrip_bytes_passed', 'reason']
    with (out / 'results.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    lines = ['# 流式 BPE 多进程探索',
             '正常耗时包含协调进程启动、分片、worker 启动/加载/读取/编码写盘/退出及拼接；'
             '不包含指纹、正确性核验及 W&B 操作。',
             '| 阶段 | workers | 缓存项数 | 正常次数 | 中位数(s) | 最小(s) | 最大(s) |',
             '|---|---:|---:|---:|---:|---:|---:|']
    groups = sorted({(r['stage'], r['workers'], r['cache_capacity']) for r in results if not r['diagnostic']})
    for stage, workers, capacity in groups:
        group = [r for r in results if (r['stage'], r['workers'], r['cache_capacity']) ==
                 (stage, workers, capacity) and r['status'] == 'ok' and not r['diagnostic']]
        times = [r['total_seconds'] for r in group]
        if times:
            lines.append(f'| {stage} | {workers} | {capacity} | {len(times)} | '
                         f'{statistics.median(times):.3f} | {min(times):.3f} | {max(times):.3f} |')
        else:
            lines.append(f'| {stage} | {workers} | {capacity} | 0 | 不可用 | — | — |')
    lines += ['', '## 独立缓存诊断', '| 容量 | 查询 | 命中率 | 清空 |', '|---|---:|---:|---:|']
    for row in results:
        if row['diagnostic'] and row['status'] == 'ok':
            stats = row['diagnostics']
            lines.append(f"| {row['cache_capacity']} | {stats['queries']} | {stats['hit_rate']:.2%} | {stats['clears']} |")
    lines += ['', 'worker_peak_rss_sum 是各进程独立峰值之和，不是同时发生的实际峰值。',
              '诊断不参与选优；本报告不替代正式训练记录，也不证明既有词表和核心代码来源合规。']
    if (out / 'selection.json').exists():
        selection = json.loads((out / 'selection.json').read_text())
        lines += ['', '## 推荐配置', f"缓存：{selection['cache_capacity']}；workers：{selection['selected_workers']}。",
                  '正式编码路径未修改。']
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def log_results(run, results):
    if run is None:
        return
    import wandb
    columns = ['stage', 'repeat', 'diagnostic', 'status', 'workers', 'cache_capacity',
               'split_seconds', 'encode_seconds', 'concatenate_seconds', 'total_seconds',
               'input_bytes', 'token_count', 'tokens_per_second', 'MiB_per_second',
               'parent_peak_rss', 'worker_peak_rss_max', 'worker_peak_rss_sum',
               'cpu_affinity', 'id_comparison_passed', 'roundtrip_bytes_passed', 'reason', 'diagnostics']
    data = []
    for result in results:
        data.append([json.dumps(result.get(c)) if c in {'cpu_affinity', 'diagnostics'} else result.get(c)
                     for c in columns])
    run.log({'exploration/candidates': wandb.Table(columns=columns, data=data)})


def run_trial(config, out, trial_id, stage, workers, capacity, repeat, diagnostic, cpu_pool, run):
    source_kind = 'train' if stage == 'train-confirmation' else 'valid'
    source = config[f'{source_kind}_file']
    reference = config[f'reference_{source_kind}_tokens']
    destination = out / f'{trial_id}.tokens'
    job = {'source': source, 'tokenizer_prefix': config['tokenizer_prefix'], 'destination': str(destination),
           'workers': workers, 'capacity': capacity, 'separator': config['split_special_token'],
           'cpus': cpu_pool[:workers], 'diagnostic': diagnostic}
    job_path = out / f'{trial_id}.job.json'
    result_path = out / f'{trial_id}.json'
    save_json(job_path, job)
    print(f'{trial_id}: {workers} workers, cache={capacity}; log={out / (trial_id + ".log")}', flush=True)
    affinity(cpu_pool[:workers], include_descendants=True)
    try:
        started = time.perf_counter()
        with (out / f'{trial_id}.log').open('w', encoding='utf-8') as log:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--coordinator-job', str(job_path),
                            '--coordinator-result', str(result_path)], check=True,
                           stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        total_seconds = time.perf_counter() - started
    finally:
        affinity(cpu_pool, include_descendants=True)
    result = json.loads(result_path.read_text())
    result.update({'stage': stage, 'workers': workers, 'cache_capacity': capacity,
                   'repeat': repeat, 'diagnostic': diagnostic, 'total_seconds': total_seconds,
                   'trial_id': trial_id})
    if result['status'] != 'ok':
        save_json(result_path, result)
        return result, None
    try:
        compare_tokens(destination, reference, config['expected_counts'][source_kind])
        result['id_comparison_passed'] = True
        bpe = BPE()
        bpe.load(config['tokenizer_prefix'])
        compare_roundtrip(destination, source, bpe)
        result['roundtrip_bytes_passed'] = True
        result['candidate_sha256'] = sha256(destination)
        result['tokens_per_second'] = result['token_count'] / total_seconds
        result['MiB_per_second'] = result['input_bytes'] / (1 << 20) / total_seconds
        save_json(result_path, result)
        return result, destination
    except BaseException as error:
        result.update({'status': 'failed', 'error': repr(error)})
        save_json(result_path, result)
        if run is not None:
            run.summary['failed_trial'] = result
        destination.unlink(missing_ok=True)
        raise


def resolve_config(args):
    if args.phase == 'train-confirmation':
        report = json.loads(Path(args.validation_report).read_text())
        if report.get('phase') != 'validation' or not report.get('completed'):
            raise ValueError('Requires a completed validation selection report')
        assert_fingerprints(report['input_fingerprints'])
        assert_fingerprints(report['source_fingerprints'])
        config = report['config'].copy()
        if config['development'] != args.development:
            raise ValueError('Development mode must match validation report')
        config['cache_capacity'] = report['cache_capacity']
        config['confirmation_workers'] = report['top_two_workers']
        config['validation_report_sha256'] = sha256(args.validation_report)
        return config
    config = {name: str(Path(getattr(args, name)).resolve()) for name in
              ['tokenizer_prefix', 'train_file', 'valid_file', 'reference_train_tokens', 'reference_valid_tokens']}
    config.update({'capacities': sorted(set(args.cache_capacities)), 'workers': sorted(set(args.workers)),
                   'repeats': args.repeats, 'seed': args.seed, 'diagnostics': args.diagnostics,
                   'split_special_token': args.split_special_token, 'development': args.development,
                   'expected_counts': dict(COUNTS)})
    if args.development:
        config['expected_counts'] = {kind: Path(config[f'reference_{kind}_tokens']).stat().st_size // 4
                                     for kind in ['train', 'valid']}
    return config


def preflight(config, phase, out):
    bpe = BPE()
    bpe.load(config['tokenizer_prefix'])
    validate_tokenizer(bpe)
    if config['split_special_token'] not in bpe.special_tokens:
        raise ValueError('Split token must be in the loaded tokenizer configuration')
    for kind in ['train', 'valid']:
        if Path(config[f'reference_{kind}_tokens']).stat().st_size != config['expected_counts'][kind] * 4:
            raise ValueError(f'{kind} reference size must match known token count')
        if not Path(config[f'{kind}_file']).is_file():
            raise FileNotFoundError(config[f'{kind}_file'])
    required = 33_000_000_000 if phase == 'train-confirmation' else config['expected_counts']['valid'] * 8 + (1 << 20)
    if config['development']:
        required = max(config['expected_counts'].values()) * 12 + (1 << 20)
    if shutil.disk_usage(out.parent).free < required:
        raise ValueError(f'Insufficient free space: need at least {required:,} bytes')
    return bpe


def execute(args):
    restrict_environment()
    config = resolve_config(args)
    if min(config['capacities']) <= 0 or config['repeats'] <= 0 or not config['workers']:
        raise ValueError('Capacities/repeats must be positive and workers nonempty')
    requested = config['confirmation_workers'] if args.phase == 'train-confirmation' else config['workers']
    if not all(1 <= n <= 16 for n in requested):
        raise ValueError('Workers must be between 1 and 16')
    if hasattr(os, 'sched_getaffinity'):
        allowed = sorted(os.sched_getaffinity(0))
        cpu_pool = allowed[:16]
        if max(requested) > len(cpu_pool):
            raise ValueError(f'Requested {max(requested)} CPUs, only {len(cpu_pool)} allowed; no automatic downgrade')
        affinity(cpu_pool, include_descendants=True)
    elif args.development:
        cpu_pool = []
    else:
        raise RuntimeError('Cloud experiments require Linux sched_setaffinity')
    if args.no_wandb and not args.development:
        raise ValueError('--no-wandb is restricted to --development small tests')
    out = Path(args.output_dir).resolve()
    if not args.development and not out.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Experiment output must be inside the personal project outputs directory')
    out.parent.mkdir(parents=True, exist_ok=True)
    bpe = preflight(config, args.phase, out)
    out.mkdir(exist_ok=False)
    inputs = [config['train_file'], config['valid_file'], config['reference_train_tokens'],
              config['reference_valid_tokens']] + [config['tokenizer_prefix'] + suffix
              for suffix in ['.vocab.json', '.merges.json', '.config.json']]
    manifest = {'config': config, 'phase': args.phase, 'cpu_pool': cpu_pool, 'python': sys.version,
                'started_utc': datetime.now(UTC).isoformat(),
                'input_fingerprints': fingerprint(inputs), 'source_fingerprints': fingerprint(source_files()),
                'special_tokens': bpe.special_tokens, 'special_token_ids': bpe._special_ids,
                'thread_environment': {key: value for key, value in os.environ.items()
                                       if key.endswith('NUM_THREADS') or key == 'CUDA_VISIBLE_DEVICES'},
                'worker_peak_rss_sum_scope': 'sum of independent peaks, not concurrent peak'}
    save_json(out / 'manifest.json', manifest)
    if args.no_wandb:
        context = nullcontext(None)
    else:
        from wandb_record import experiment
        context = experiment('transformer-bpe-parallel-exploration', manifest, root=ROOT)
    results, schedule = [], []
    save_json(out / 'results.json', results)
    with context as run:
        def trial(stage, workers, capacity, repeat=0, diagnostic=False):
            trial_id = f'trial-{len(schedule) + 1:03d}'
            schedule.append({'trial_id': trial_id, 'stage': stage, 'workers': workers,
                             'cache_capacity': capacity, 'repeat': repeat, 'diagnostic': diagnostic})
            save_json(out / 'schedule.json', schedule)
            result, candidate = run_trial(config, out, trial_id, stage, workers, capacity,
                                          repeat, diagnostic, cpu_pool, run)
            results.append(result)
            save_json(out / 'results.json', results)
            summary_files(out, results)
            log_results(run, results)
            return result, candidate
        try:
            if args.phase == 'validation':
                for spec in shuffled_schedule(config['capacities'], config['repeats'], config['seed']):
                    _, candidate = trial('cache', 1, spec['value'], spec['repeat'])
                    if candidate:
                        candidate.unlink()
                cache_rows = [r for r in results if r['stage'] == 'cache']
                capacity = choose_smallest(grouped_medians(cache_rows, 'cache_capacity'), 0.03)
                if config['diagnostics']:
                    for value in config['capacities']:
                        _, candidate = trial('cache-diagnostic', 1, value, diagnostic=True)
                        if candidate:
                            candidate.unlink()
                for spec in shuffled_schedule(config['workers'], config['repeats'], config['seed'] + 1):
                    _, candidate = trial('scaling', spec['value'], capacity, spec['repeat'])
                    if candidate:
                        candidate.unlink()
                scale_rows = [r for r in results if r['stage'] == 'scaling']
                # Require every repeat of a configuration to be available and correct.
                complete = [n for n in config['workers'] if sum(r['status'] == 'ok' for r in scale_rows
                                                               if r['workers'] == n) == config['repeats']]
                medians = grouped_medians([r for r in scale_rows if r['workers'] in complete], 'workers')
                selected = choose_workers(medians)
                top_two = sorted(medians, key=lambda n: (medians[n], n))[:2]
                selection = {'cache_capacity': capacity, 'selected_workers': selected,
                             'top_two_workers': top_two, 'worker_medians': medians,
                             'capacity_medians': grouped_medians(cache_rows, 'cache_capacity')}
            else:
                capacity = config['cache_capacity']
                retained = []
                for workers in config['confirmation_workers']:
                    result, candidate = trial('train-confirmation', workers, capacity)
                    if candidate:
                        retained.append((result, candidate))
                medians = grouped_medians(results, 'workers')
                selected = choose_workers(medians)
                chosen = next(path for result, path in retained if result['workers'] == selected)
                for _, candidate in retained:
                    if candidate != chosen:
                        candidate.unlink()
                chosen.replace(out / 'train.tokens')
                selection = {'cache_capacity': capacity, 'selected_workers': selected,
                             'worker_medians': medians, 'retained_tokens': str(out / 'train.tokens'),
                             'retained_sha256': sha256(out / 'train.tokens')}
            assert_fingerprints(manifest['input_fingerprints'])
            assert_fingerprints(manifest['source_fingerprints'])
            selection.update({'phase': args.phase, 'completed': True, 'config': config,
                              'input_fingerprints': manifest['input_fingerprints'],
                              'source_fingerprints': manifest['source_fingerprints']})
            save_json(out / 'selection.json', selection)
            save_json(out / 'integrity.json', {'unchanged': True})
            summary_files(out, results)
            if run is not None:
                run.summary.update({'selected_cache_capacity': capacity, 'selected_workers': selected,
                                    'completed': True, 'all_generated_candidates_correct': True})
                for name in ['manifest.json', 'results.json', 'schedule.json', 'results.csv',
                             'selection.json', 'report.md', 'integrity.json']:
                    run.save(str(out / name), base_path=str(out), policy='now')
                save_json(out / 'wandb.json', {'url': run.url})
        except BaseException as error:
            save_json(out / 'failure.json', {'error': repr(error), 'completed': False})
            if run is not None:
                run.summary.update({'completed': False, 'failure': repr(error)})
            raise
    print(f'Completed: {out / "report.md"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['validation', 'train-confirmation'], default='validation')
    parser.add_argument('--tokenizer-prefix', default='outputs/e9275a3b4a7a/tokenizer')
    parser.add_argument('--train-file', default='/data/assignment1-data/owt_train.txt')
    parser.add_argument('--valid-file', default='/data/assignment1-data/owt_valid.txt')
    parser.add_argument('--reference-train-tokens', default='outputs/6bb24d46a473/train.tokens')
    parser.add_argument('--reference-valid-tokens', default='outputs/6bb24d46a473/valid.tokens')
    parser.add_argument('--cache-capacities', nargs='+', type=int, default=[2048, 65536, 262144, 1048576])
    parser.add_argument('--workers', nargs='+', type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20261013)
    parser.add_argument('--split-special-token', default='<|endoftext|>')
    parser.add_argument('--diagnostics', action='store_true')
    parser.add_argument('--output-dir', default='outputs/parallel-bpe-exploration')
    parser.add_argument('--validation-report')
    parser.add_argument('--development', action='store_true', help='Small fixtures only; reference sizes inferred, not course evidence')
    parser.add_argument('--no-wandb', action='store_true', help='Development tests only')
    parser.add_argument('--coordinator-job', help=argparse.SUPPRESS)
    parser.add_argument('--coordinator-result', help=argparse.SUPPRESS)
    args = parser.parse_args()
    restrict_environment()
    if args.coordinator_job:
        job = json.loads(Path(args.coordinator_job).read_text())
        result = encode_parallel(**job)
        save_json(args.coordinator_result, result)
    else:
        if args.phase == 'train-confirmation' and not args.validation_report:
            parser.error('--validation-report is required for train-confirmation')
        execute(args)


if __name__ == '__main__':
    main()
