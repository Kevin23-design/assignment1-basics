"""Standard-library-only launcher for fixed independent six-hour GPU suites."""
from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from train_context_lr_experiment import ROOT, SUITES, case_parameters, case_suite, eligible_result, save_result

SKIP = {'.git', '.venv', 'venv', 'outputs', 'wandb', 'data', 'datasets',
        'checkpoints', '__pycache__', '.pytest_cache'}


def allocate_cpus(allowed, suite='context-lr'):
    allowed = sorted(set(allowed))
    if len(allowed) < 64:
        raise ValueError(f'This CPU partition requires 64 allowed logical CPUs; found {len(allowed)}')
    settings = SUITES[suite]
    offset = settings['cpu_offset']
    return {case: allowed[offset + index * 16:offset + (index + 1) * 16]
            for index, case in enumerate(settings['cases'])}


def source_fingerprints():
    result = {}
    for directory, dirs, names in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs if d not in SKIP)
        for name in sorted(names):
            path = Path(directory) / name
            if path.suffix in {'.py', '.sh', '.ipynb', '.cu', '.c', '.cpp', '.h', '.rs'} or name in {
                'pyproject.toml', 'uv.lock', '.python-version'
            }:
                result[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def case_command(case, cpus, study, out, gpu):
    environment = os.environ.copy()
    environment['CUDA_VISIBLE_DEVICES'] = str(gpu)
    command = [sys.executable, str(ROOT / 'train_context_lr_experiment.py'), '--case', case,
               '--cpu-ids', *map(str, cpus), '--study-id', study, '--suite', case_suite(case),
               '--output-dir', str(out)]
    return command, environment


def descendants(pid):
    parents = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / 'status').read_text()
            parents[int(entry.name)] = int(next(line.split()[1] for line in status.splitlines()
                                               if line.startswith('PPid:')))
        except (FileNotFoundError, ProcessLookupError, PermissionError, StopIteration):
            continue
    selected = {pid}
    while True:
        extra = {child for child, parent in parents.items() if parent in selected}
        if extra <= selected:
            return selected
        selected.update(extra)


def stop_processes(processes):
    # Encoding coordinators use their own sessions: include nested groups too.
    groups = set()
    for process in processes.values():
        if process.poll() is not None:
            continue
        for pid in descendants(process.pid):
            try:
                groups.add(os.getpgid(pid))
            except ProcessLookupError:
                pass
    groups.discard(os.getpgrp())
    for group in groups:
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    limit = time.monotonic() + 30
    while time.monotonic() < limit and any(p.poll() is None for p in processes.values()):
        time.sleep(.1)
    for group in groups:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for process in processes.values():
        process.wait()


def collect(out, manifest, exit_codes, interrupted=False):
    suite = manifest.get('suite', 'context-lr')
    results = []
    for case, values in manifest['cases'].items():
        path = out / case / 'result.json'
        if path.exists():
            try:
                result = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                result = {'status': 'unreadable_result'}
        else:
            result = {'status': 'not_started' if case not in exit_codes else 'missing_result'}
        result.update({'case': case, 'exit_code': exit_codes.get(case), 'assigned_gpu': values['gpu'],
                       'cpu_ids': values['cpu_ids']})
        result['eligible'] = eligible_result(result) and result['exit_code'] == 0
        results.append(result)
    changed = source_fingerprints() != manifest['source_fingerprints']
    if changed:
        for result in results:
            result['eligible'] = False
    aggregate = {'study_id': manifest['study_id'], 'suite': suite, 'interrupted': interrupted,
                 'source_changed': changed, 'results': results,
                 'validation_protocol': 'native context only; cross-context comparison is whole-configuration comparison'}
    save_result(out / 'results.json', aggregate)
    stop_fields = ['train_stop_reason', 'train_stop_elapsed_seconds', 'train_stop_remaining_seconds',
                   'train_stop_last_step_seconds', 'train_stop_step_safety_seconds', 'train_stop_reserve_seconds']
    phase_fields = ['encode_seconds', 'train_seconds', 'save_seconds', 'validation_seconds']
    columns = ['case', 'gpu', 'context_length', 'batch_size', 'learning_rate',
               'optimizer_beta1', 'optimizer_beta2', 'weight_decay', 'max_grad_norm',
               'tokens_per_step', 'warmup_steps', 'cosine_steps', 'warmup_target_tokens', 'cosine_target_tokens',
               'status', 'eligible',
               'run_id', 'wandb_url', 'val_ppl', 'val_loss', 'validation_targets', 'final_step',
               'target_tokens', 'tokens_per_second', 'peak_allocated_bytes', 'peak_reserved_bytes',
               'total_seconds', 'exit_code', *phase_fields, *stop_fields]
    rows = []
    for result in results:
        case = result['case']
        summary = result.get('summary', {})
        parameters = case_parameters(case)
        beta1, beta2 = parameters.pop('optimizer_betas')
        tokens_per_step = parameters['batch_size'] * parameters['context_length']
        rows.append({**parameters, 'case': case, 'gpu': result['assigned_gpu'],
                     'optimizer_beta1': beta1, 'optimizer_beta2': beta2,
                     'tokens_per_step': tokens_per_step,
                     'warmup_target_tokens': tokens_per_step * parameters['warmup_steps'],
                     'cosine_target_tokens': tokens_per_step * parameters['cosine_steps'],
                     'status': result['status'], 'eligible': result['eligible'],
                     'run_id': result.get('run_id'), 'wandb_url': result.get('wandb_url'),
                     'val_ppl': summary.get('val/ppl'), 'val_loss': summary.get('val/loss'),
                     'validation_targets': summary.get('val/token_count'),
                     'final_step': summary.get('train/final_step'),
                     'target_tokens': summary.get('train/target_tokens'),
                     'tokens_per_second': summary.get('train/tokens_per_second'),
                     'peak_allocated_bytes': summary.get('gpu/peak_allocated_bytes'),
                     'peak_reserved_bytes': summary.get('gpu/peak_reserved_bytes'),
                     'total_seconds': summary.get('time/total_seconds'), 'exit_code': result['exit_code'],
                     **{field: summary.get('time/' + field) for field in phase_fields + stop_fields}})
    with (out / 'results.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    title = '# 优化器组合 × 大 batch 性能候选' if suite == 'optimizer-batch' else '# 上下文 × 峰值学习率 2×2'
    history = ('历史 256 窗口参考：C（batch128、LR3e-4）PPL 45.18785；'
               'D（batch128、LR6e-4）PPL 41.75639。' if suite == 'optimizer-batch'
               else '历史基线完整验证 PPL：40.60899；本报告不自动修改 dev。')
    lines = [title,
             '各组使用自身上下文完成验证；跨上下文比较评价整套配置，不单独归因于训练上下文。',
             history,
             '| 组 | GPU | 验证上下文 | batch | 峰值 LR | β₂ | WD | 裁剪 | token/步 | warmup/终点步 | 状态 | 有效 | PPL | 最终 step | Run |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|---|']
    for row in rows:
        url = f"[W&B]({row['wandb_url']})" if row['wandb_url'] else '—'
        lines.append(f"| {row['case']} | {row['gpu']} | {row['context_length']} | {row['batch_size']} | "
                     f"{row['learning_rate']} | {row['optimizer_beta2']} | {row['weight_decay']} | "
                     f"{row['max_grad_norm']} | {row['tokens_per_step']:,} | "
                     f"{row['warmup_steps']}/{row['cosine_steps']} | {row['status']} | {row['eligible']} | {row['val_ppl']} | "
                     f"{row['final_step']} | {url} |")
    eligible = [r for r in rows if r['eligible']]
    if eligible:
        lines += ['', '## 有效配置按完整验证 PPL 排序',
                  ', '.join(f"{r['case']}: {r['val_ppl']:.5f}（验证窗口 {r['context_length']}）"
                            for r in sorted(eligible, key=lambda r: r['val_ppl']))]
    if changed:
        lines += ['', '启动与结束源码指纹不同，全部结果排除出有效表。']
    if suite == 'optimizer-batch':
        lines += ['', 'E 同时改变三项优化器设置；F 又改变 batch 与峰值 LR，只能评估组合效果。',
                  '按相同累计目标 token 对照区间 loss，不能按相同步数比较两组训练预算。',
                  'warmup 均对应 16,384,000 tokens；余弦终点均对应 2,129,920,000 tokens。']
    else:
        lines += ['', 'B/A、D/C 比较同上下文的学习率；C/A、D/B 比较固定 token batch 的上下文安排。']
    lines += ['', '每步目标 token：' + '，'.join(f"{r['case']}={r['tokens_per_step']:,}" for r in rows) + '。',
              '相同累计 token 不表示使用相同采样序列；吞吐结合编码耗时、GPU 型号与并发负载解释。',
              '停训诊断见 results.csv：退出原因、elapsed、剩余预算、最后一步耗时与两倍安全余量。',
              '保持原停训条件，不自动降低 batch、不做显存预检、不自动修改 dev。']
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return aggregate


def launch(args):
    settings = SUITES[args.suite]
    cases = settings['cases']
    if not hasattr(os, 'sched_getaffinity'):
        raise RuntimeError('This launcher requires Linux CPU affinity')
    if len(args.gpus) != len(cases) or len(set(args.gpus)) != len(cases) or any(g < 0 for g in args.gpus):
        raise ValueError(f'Suite {args.suite} requires {len(cases)} distinct nonnegative GPU indices')
    cpus = allocate_cpus(os.sched_getaffinity(0), args.suite)
    selected = {cpu for ids in cpus.values() for cpu in ids}
    os.sched_setaffinity(0, selected)
    out = Path(args.output_dir).resolve()
    if not out.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Output must be in the personal project outputs directory')
    if out.exists():
        raise FileExistsError(out)
    if not args.study_id or not args.study_id.replace('-', '').replace('_', '').isalnum():
        raise ValueError('Study ID must contain letters, numbers, hyphens or underscores')
    for name in ['WANDB_ENTITY', 'STUDENT_ID']:
        if not os.environ.get(name):
            raise ValueError(f'Missing {name}')
    for source in [Path('/data/assignment1-data/owt_train.txt'), Path('/data/assignment1-data/owt_valid.txt')]:
        if not source.is_file():
            raise FileNotFoundError(source)
    for suffix in ['.vocab.json', '.merges.json', '.config.json']:
        path = ROOT / ('outputs/e9275a3b4a7a/tokenizer' + suffix)
        if not path.is_file():
            raise FileNotFoundError(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out.parent).free < settings['disk_bytes']:
        raise ValueError(f'At least {settings["disk_bytes"] // 1_000_000_000} GB free disk space is required')
    out.mkdir(exist_ok=False)
    manifest = {'study_id': args.study_id, 'suite': args.suite, 'created_utc': datetime.now(UTC).isoformat(),
                'source_fingerprints': source_fingerprints(), 'cases': {
                    case: {**case_parameters(case), 'gpu': gpu, 'cpu_ids': cpus[case]}
                    for case, gpu in zip(cases, args.gpus, strict=True)}}
    save_result(out / 'manifest.json', manifest)
    processes, logs, codes = {}, {}, {}
    interrupted = False

    def interrupt(signum, frame):
        raise KeyboardInterrupt('Launcher interrupted')

    previous_handler = signal.signal(signal.SIGTERM, interrupt)
    try:
        for case, values in manifest['cases'].items():
            command, environment = case_command(case, values['cpu_ids'], args.study_id, out / case, values['gpu'])
            logs[case] = (out / (case + '.log')).open('w', encoding='utf-8')
            processes[case] = subprocess.Popen(command, env=environment, stdout=logs[case],
                                               stderr=subprocess.STDOUT, start_new_session=True, cwd=ROOT)
            print(f"Started {case}: GPU {values['gpu']}, CPUs {values['cpu_ids']}, log={out / (case + '.log')}", flush=True)
        while len(codes) < len(processes):
            for case, process in processes.items():
                code = process.poll()
                if code is not None and case not in codes:
                    codes[case] = process.wait()
                    print(f'{case} finished, exit code {code}', flush=True)
                    collect(out, manifest, codes)
            time.sleep(1)
    except BaseException:
        interrupted = True
        stop_processes(processes)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        for case, process in processes.items():
            if process.poll() is not None:
                codes[case] = process.wait()
        for handle in logs.values():
            handle.close()
        collect(out, manifest, codes, interrupted)
    print(f'Completed: {out / "report.md"}', flush=True)
    return 0 if all(code == 0 for code in codes.values()) else 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=list(SUITES), default='context-lr')
    parser.add_argument('--gpus', nargs='+', type=int)
    parser.add_argument('--output-dir')
    parser.add_argument('--study-id')
    args = parser.parse_args(argv)
    settings = SUITES[args.suite]
    if args.gpus is None:
        args.gpus = list(settings['gpus'])
    if args.output_dir is None:
        args.output_dir = 'outputs/' + settings['output_name']
    if args.study_id is None:
        args.study_id = settings['output_name'] + '-' + datetime.now(UTC).strftime('%Y%m%d-%H%M%S')
    if len(args.gpus) != len(settings['cases']) or len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0:
        parser.error(f"{args.suite} requires {len(settings['cases'])} distinct nonnegative GPU indices")
    return args


def main():
    raise SystemExit(launch(parse_args()))


if __name__ == '__main__':
    main()
