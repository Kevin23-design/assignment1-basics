"""Standard-library-only launcher for four independent six-hour GPU cases."""
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

from train_context_lr_experiment import CASES, ROOT, eligible_result, save_result

SKIP = {'.git', '.venv', 'venv', 'outputs', 'wandb', 'data', 'datasets',
        'checkpoints', '__pycache__', '.pytest_cache'}


def allocate_cpus(allowed):
    allowed = sorted(set(allowed))
    if len(allowed) < 64:
        raise ValueError(f'Four cases require 64 allowed logical CPUs; found {len(allowed)}')
    return {case: allowed[index * 16:(index + 1) * 16] for index, case in enumerate(CASES)}


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
               '--cpu-ids', *map(str, cpus), '--study-id', study, '--output-dir', str(out)]
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
    aggregate = {'study_id': manifest['study_id'], 'interrupted': interrupted,
                 'source_changed': changed, 'results': results,
                 'validation_protocol': 'native context only; cross-context comparison is whole-configuration comparison'}
    save_result(out / 'results.json', aggregate)
    columns = ['case', 'gpu', 'context_length', 'batch_size', 'learning_rate', 'status', 'eligible',
               'run_id', 'wandb_url', 'val_ppl', 'val_loss', 'validation_targets', 'final_step',
               'target_tokens', 'tokens_per_second', 'peak_allocated_bytes', 'peak_reserved_bytes',
               'total_seconds', 'exit_code']
    rows = []
    for result in results:
        case = result['case']
        summary = result.get('summary', {})
        rows.append({**CASES[case], 'case': case, 'gpu': result['assigned_gpu'],
                     'status': result['status'], 'eligible': result['eligible'],
                     'run_id': result.get('run_id'), 'wandb_url': result.get('wandb_url'),
                     'val_ppl': summary.get('val/ppl'), 'val_loss': summary.get('val/loss'),
                     'validation_targets': summary.get('val/token_count'),
                     'final_step': summary.get('train/final_step'),
                     'target_tokens': summary.get('train/target_tokens'),
                     'tokens_per_second': summary.get('train/tokens_per_second'),
                     'peak_allocated_bytes': summary.get('gpu/peak_allocated_bytes'),
                     'peak_reserved_bytes': summary.get('gpu/peak_reserved_bytes'),
                     'total_seconds': summary.get('time/total_seconds'), 'exit_code': result['exit_code']})
    with (out / 'results.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# 上下文 × 峰值学习率 2×2',
             '各组使用自身上下文完成验证；跨上下文比较评价整套配置，不单独归因于训练上下文。',
             '历史基线完整验证 PPL：40.60899；本报告不自动修改 dev。',
             '| 组 | GPU | 验证上下文 | batch | 峰值 LR | 状态 | 有效 | 完整验证 PPL | 最终 step | Run |',
             '|---|---:|---:|---:|---:|---|---|---:|---:|---|']
    for row in rows:
        url = f"[W&B]({row['wandb_url']})" if row['wandb_url'] else '—'
        lines.append(f"| {row['case']} | {row['gpu']} | {row['context_length']} | {row['batch_size']} | "
                     f"{row['learning_rate']} | {row['status']} | {row['eligible']} | {row['val_ppl']} | "
                     f"{row['final_step']} | {url} |")
    eligible = [r for r in rows if r['eligible']]
    if eligible:
        lines += ['', '## 有效配置按完整验证 PPL 排序',
                  ', '.join(f"{r['case']}: {r['val_ppl']:.5f}（验证窗口 {r['context_length']}）"
                            for r in sorted(eligible, key=lambda r: r['val_ppl']))]
    if changed:
        lines += ['', '启动与结束源码指纹不同，全部结果排除出有效表。']
    lines += ['', 'B/A、D/C 比较同上下文的学习率；C/A、D/B 比较固定 token batch 的上下文安排。',
              '每步均为 32,768 目标 token；相同 step 不表示跨上下文的采样序列相同。',
              '四 GPU 共享磁盘和主机，吞吐需结合编码耗时、GPU 型号与实际最终 token 数解释。']
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return aggregate


def launch(args):
    if not hasattr(os, 'sched_getaffinity'):
        raise RuntimeError('This launcher requires Linux CPU affinity')
    if len(args.gpus) != 4 or len(set(args.gpus)) != 4 or any(g < 0 for g in args.gpus):
        raise ValueError('Exactly four distinct nonnegative GPU indices are required')
    cpus = allocate_cpus(os.sched_getaffinity(0))
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
    if shutil.disk_usage(out.parent).free < 100_000_000_000:
        raise ValueError('At least 100 GB free disk space is required')
    out.mkdir(exist_ok=False)
    manifest = {'study_id': args.study_id, 'created_utc': datetime.now(UTC).isoformat(),
                'source_fingerprints': source_fingerprints(), 'cases': {
                    case: {**CASES[case], 'gpu': gpu, 'cpu_ids': cpus[case]}
                    for case, gpu in zip(CASES, args.gpus, strict=True)}}
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', nargs=4, type=int, default=[0, 1, 2, 3])
    parser.add_argument('--output-dir', default='outputs/context-lr-2x2')
    parser.add_argument('--study-id', default='context-lr-2x2-' + datetime.now(UTC).strftime('%Y%m%d-%H%M%S'))
    args = parser.parse_args()
    raise SystemExit(launch(args))


if __name__ == '__main__':
    main()
