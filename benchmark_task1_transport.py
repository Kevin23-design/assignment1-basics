"""Explore task-one public-API transport costs; never trains or updates dev."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import tempfile
import time

import bpe as implementation
from bpe import BPE, CACHE_CAPACITY
import benchmark_task1_efficiency as common
from parallel_bpe_encoding import affinity, restrict_environment, rss_bytes, validate_tokenizer
from task1_transport_runtime import CASES, Unavailable, selected_backend

ROOT = Path(__file__).resolve().parent
EXPECTED_TOKENS = 66401098


def validate_fixed_parameters():
    if (CACHE_CAPACITY, implementation._ENCODE_WORKERS, implementation._PARTITIONS_PER_WORKER,
        implementation._DECODE_BLOCK_SIZE) != (1048576, 16, 16, 131072):
        raise ValueError('Transport exploration requires the fixed 16-worker / 256-part / 1M / 128K configuration')


def payloads(case, input_bytes, tokens):
    result = {'units': 'logical UTF-8 text / uint32 payload; not measured pickle wire bytes',
              'caller_to_coordinator_input_bytes': input_bytes if case in ('B', 'R') else 0,
              'coordinator_to_workers_input_bytes': input_bytes if case in ('B', 'R') else 0,
              'workers_to_coordinator_token_bytes': tokens * 4 if case in ('B', 'R') else 0,
              'coordinator_to_caller_token_bytes': tokens * 4 if case in ('B', 'R') else 0,
              'caller_to_workers_input_bytes': input_bytes if case == 'D' else 0,
              'workers_to_caller_token_bytes': tokens * 4 if case == 'D' else 0,
              'shared_input_bytes': input_bytes if case == 'S' else 0,
              'shared_result_bytes': tokens * 4 if case == 'S' else 0}
    return result


def worker(job, result_path):
    restrict_environment()
    validate_fixed_parameters()
    affinity(job['cpus'])
    # Mirror the course caller's import context, without creating a W&B run in this trial.
    import wandb  # noqa: F401
    tokenizer = BPE()
    tokenizer.load(job['tokenizer_prefix'])
    validate_tokenizer(tokenizer)
    with open(job['valid_file'], encoding='utf-8', newline='') as handle:
        text = handle.read()
    result = {'case': job['case'], 'diagnostic': job['diagnostic'], 'status': 'failed',
              'cpu_affinity': job['cpus'], 'workers': job['workers'], 'parts': job['workers'] * 16,
              'cache_capacity': CACHE_CAPACITY, 'decode_block_size': 131072}
    try:
        if tokenizer._cache:
            raise RuntimeError('Each evaluation must start with an empty cache')
        metrics = {}
        if job['write_reference']:
            started = time.perf_counter()
            ids = tokenizer._encode_serial(text)
            encoded = time.perf_counter()
            decoded = tokenizer.decode(ids)
            decoded_at = time.perf_counter()
            passed = decoded == text
            ended = time.perf_counter()
        else:
            with selected_backend(job['case'], job['diagnostic'],
                                  job['workers'] if job['development'] else None) as (metrics, calls):
                started = time.perf_counter()
                ids = tokenizer.encode(text)
                encoded = time.perf_counter()
                decoded = tokenizer.decode(ids)
                decoded_at = time.perf_counter()
                passed = decoded == text
                ended = time.perf_counter()
                if calls != [job['case']]:
                    raise RuntimeError('Requested transport was bypassed or invoked more than once')
        result.update(total_seconds=ended - started, encode_seconds=encoded - started,
                      decode_seconds=decoded_at - encoded, compare_seconds=ended - decoded_at,
                      token_count=len(ids), returned_type=type(ids).__name__, roundtrip_passed=passed,
                      parent_peak_rss_bytes=rss_bytes(), transport=metrics)
        if not isinstance(ids, list) or not passed:
            raise ValueError('Must return a complete list[int] and restore the exact input')
        if job['expected_tokens'] is not None and len(ids) != job['expected_tokens']:
            raise ValueError('Token count differs from full validation reference')
        common.check_reference(ids, job['reference'], write=job['write_reference'])
        result.update(status='ok', ids_match_reference=None if job['write_reference'] else True,
                      reference_generated=job['write_reference'],
                      logical_payloads=payloads(job['case'], Path(job['valid_file']).stat().st_size, len(ids)),
                      first_200=common.first_tokens(tokenizer, ids, text))
        metadata = metrics.get('worker_metadata', metrics.get('coordinator', {}).get('worker_metadata'))
        if metadata:
            result.update(worker_peak_rss_max=max(item['peak_rss_bytes'] for item in metadata),
                          worker_peak_rss_sum=sum(item['peak_rss_bytes'] for item in metadata))
    except Unavailable as error:
        if job['case'] != 'S':
            raise
        result.update(status='unavailable', reason=str(error), transport=metrics)
    except BaseException as error:
        result['error'] = repr(error)
        common.save_json(result_path, result)
        raise
    common.save_json(result_path, result)


def fastest(rows, repeats):
    medians = {}
    if any(row['status'] == 'ok' and (not row.get('roundtrip_passed') or row.get('ids_match_reference') is not True)
           for row in rows):
        raise ValueError('A screening result failed exact correctness checks')
    for case in CASES:
        group = [row for row in rows if row['case'] == case and not row['diagnostic']]
        if len(group) == repeats and all(row['status'] == 'ok' for row in group):
            medians[case] = statistics.median(row['total_seconds'] for row in group)
    if 'B' not in medians:
        raise ValueError('Baseline did not complete every screening round')
    selected = min(medians, key=lambda case: (medians[case], case != 'B', case))
    return selected, medians


def run_stages(args, trial):
    if args.diagnostics:
        trial('diagnostic-baseline', 'B', diagnostic=True)
    group = []
    rng = random.Random(args.seed)
    for repeat in range(1, args.repeats + 1):
        order = list(CASES)
        rng.shuffle(order)
        group.extend(trial('screening', case, repeat) for case in order)
    candidate, medians = fastest(group, args.repeats)
    if args.diagnostics:
        trial('diagnostic-candidate', candidate, diagnostic=True)
    baseline_rows, candidate_rows = [], []
    rng = random.Random(args.seed + 1)
    for repeat in range(1, args.repeats + 1):
        pair = [('paired-baseline', 'B', baseline_rows), ('paired-candidate', candidate, candidate_rows)]
        rng.shuffle(pair)
        for stage, case, rows in pair:
            rows.append(trial(stage, case, repeat))
    decision = common.paired_decision(baseline_rows, candidate_rows, args.repeats)
    if candidate == 'B':
        decision.update(promote_candidate=False, reason='candidate_matches_current_configuration')
    recommended = candidate if decision['promote_candidate'] else 'B'
    confirmation = trial('confirmation', recommended)
    if confirmation['status'] != 'ok':
        raise ValueError('Independent confirmation failed')
    return {'completed': True, 'recommended': recommended, 'candidate': candidate,
            'screening_medians_seconds': medians, 'paired_confirmation': decision,
            'confirmation_trial': confirmation['trial_id'], 'confirmation_seconds': confirmation['total_seconds'],
            'submission_provenance_verified': False}


def write_report(out, rows, selection):
    columns = ['trial_id', 'stage', 'repeat', 'case', 'diagnostic', 'status', 'total_seconds', 'encode_seconds',
               'decode_seconds', 'compare_seconds', 'token_count', 'ids_match_reference', 'roundtrip_passed',
               'parent_peak_rss_bytes', 'worker_peak_rss_max', 'worker_peak_rss_sum', 'reason']
    with (out / 'results.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# 任务一正式接口接入层探索',
             '所有候选计时：BPE.encode → 完整 list[int] → BPE.decode → 原文比较。',
             '分片、状态/输入准备、进程启动、传输、组装、退出及共享内存释放均计时；诊断不排名。',
             '18.437 秒和 12.012 秒仅为历史参考，选优使用本轮配对结果。',
             '| 阶段 | 后端 | 有效次数 | 中位数(s) | 最小(s) | 最大(s) |', '|---|---|---:|---:|---:|---:|']
    groups = {}
    for row in rows:
        if not row['diagnostic']:
            groups.setdefault((row['stage'], row['case']), []).append(row)
    for (stage, case), group in groups.items():
        times = [r['total_seconds'] for r in group if r['status'] == 'ok']
        if times:
            lines.append(f'| {stage} | {case} | {len(times)} | {statistics.median(times):.3f} | '
                         f'{min(times):.3f} | {max(times):.3f} |')
        else:
            lines.append(f'| {stage} | {case} | 0 | 不可用 | — | — |')
    lines += ['', '## 推荐与配对确认', '```json', json.dumps(selection, ensure_ascii=False, indent=2), '```',
              '', '## 独立诊断与不可用原因']
    for row in rows:
        if row['diagnostic'] or row['status'] == 'unavailable':
            data = {key: row[key] for key in ['trial_id', 'case', 'status', 'reason', 'transport', 'logical_payloads']
                    if key in row}
            lines += ['', '```json', json.dumps(data, ensure_ascii=False, indent=2), '```']
    lines += ['', '跨进程时间重叠，不能相加；send/receive 包含序列化、阻塞或等待，并非纯通信时间。',
              '载荷是逻辑 UTF-8/uint32 数据量，不是实测 pickle 线缆字节数。共享内存和直接 worker 是整套实现比较。',
              'worker 峰值 RSS 之和是独立峰值之和，不是同时发生的实际峰值。',
              'B 正常测速使用未改动的正式 runtime；其 worker RSS 不可得时留空，仅独立诊断提供。',
              '诊断使用相同结构的阶段探针，不启用 cProfile；其耗时不代表正常速度。',
              '本实验不自动更新 dev，不证明词表训练来源，也不承诺恢复到历史 12 秒。']
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def log_rows(run, rows):
    if run is None:
        return
    import wandb
    columns = ['trial_id', 'stage', 'case', 'repeat', 'diagnostic', 'status', 'workers', 'parts',
               'cache_capacity', 'decode_block_size', 'total_seconds', 'encode_seconds', 'decode_seconds',
               'compare_seconds', 'token_count', 'ids_match_reference', 'roundtrip_passed', 'cpu_affinity',
               'parent_peak_rss_bytes', 'worker_peak_rss_max', 'worker_peak_rss_sum']
    run.log({'transport/trials': wandb.Table(columns=columns, data=[[r.get(k) for k in columns] for r in rows])})


def stop_trial(process):
    """The trial's coordinator/direct workers have their own sessions: reclaim that owned tree too."""
    import psutil
    from task1_transport_runtime import stop_processes
    try:
        root = psutil.Process(process.pid)
        descendants = root.children(recursive=True)
    except psutil.NoSuchProcess:
        stop_processes([process], groups=True)
        return
    owned = [root, *descendants]
    groups = set()
    for member in owned:
        try:
            if member.is_running():
                group = os.getpgid(member.pid)
                if group != os.getpgrp():
                    groups.add(group)
        except ProcessLookupError:
            pass
    for group in groups:
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    _, alive = psutil.wait_procs(descendants, timeout=2)
    for member in [root, *alive]:
        try:
            # psutil checks the original creation time, avoiding signals to reused PIDs.
            if member.is_running() and member.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
                group = os.getpgid(member.pid)
                if group in groups:
                    os.killpg(group, signal.SIGKILL)
        except (psutil.NoSuchProcess, ProcessLookupError):
            pass
    process.wait()


def launch_job(job, job_path, result_path, log_path):
    common.save_json(job_path, job)
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker-job', str(job_path),
                                    '--worker-result', str(result_path)], env=os.environ.copy(), cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            if process.wait():
                raise RuntimeError(f'Trial failed; inspect {log_path}')
        except BaseException:
            stop_trial(process)
            raise
    return json.loads(result_path.read_text(encoding='utf-8'))


def execute(args):
    restrict_environment()
    validate_fixed_parameters()
    if args.repeats < (1 if args.development else 5):
        raise ValueError('Cloud runs require at least five repeats')
    if args.no_wandb and not args.development:
        raise ValueError('--no-wandb is restricted to development fixtures')
    workers = args.development_workers if args.development else 16
    if not 1 <= workers <= 16:
        raise ValueError('Development workers must be 1..16')
    if hasattr(os, 'sched_getaffinity'):
        allowed = sorted(os.sched_getaffinity(0))
        cpus = sorted(args.cpu_ids) if args.cpu_ids is not None else allowed[:workers]
        if len(cpus) != workers or len(set(cpus)) != workers or not set(cpus) <= set(allowed):
            raise ValueError(f'Requires exactly {workers} distinct allowed CPUs; no automatic downgrade')
        affinity(cpus, include_descendants=True)
    elif args.development and args.cpu_ids is None:
        cpus = []
    else:
        raise RuntimeError('Cloud runs require Linux CPU affinity')
    out = Path(args.output_dir).resolve()
    if not args.development and not out.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Cloud output must be inside personal project outputs')
    if out.exists():
        raise FileExistsError(out)
    prefix, valid = str(Path(args.tokenizer_prefix).resolve()), str(Path(args.valid_file).resolve())
    inputs = [valid] + [prefix + suffix for suffix in ('.vocab.json', '.merges.json', '.config.json')]
    reference = str(Path(args.reference_tokens).resolve()) if args.reference_tokens else None
    if reference:
        size = Path(reference).stat().st_size
        if size % 4 or (not args.development and size != EXPECTED_TOKENS * 4):
            raise ValueError('Reference must have the exact full-validation uint32 size')
        inputs.append(reference)
    tokenizer = BPE()
    tokenizer.load(prefix)
    validate_tokenizer(tokenizer)
    manifest = {'arguments': vars(args), 'python': sys.version, 'started_utc': datetime.now(UTC).isoformat(),
                'input_sha256': {path: common.sha256(path) for path in inputs},
                'source_sha256': common.source_fingerprints(), 'cpu_affinity': cpus,
                'workers': workers, 'parts': workers * 16, 'cache_capacity': CACHE_CAPACITY,
                'cache_policy': 'clear_all', 'decode_block_size': 131072,
                'special_tokens': tokenizer.special_tokens, 'diagnostics_ranked': False,
                'backend_imports_preloaded': True,
                'timing': 'complete public encode/list/decode/exact compare, including transport cleanup'}
    out.mkdir(parents=True, exist_ok=False)
    common.save_json(out / 'manifest.json', manifest)
    if args.no_wandb:
        context = nullcontext(None)
    else:
        from wandb_record import experiment
        context = experiment('bpe-task1-transport-exploration', manifest, root=ROOT)
    rows, order = [], []
    expected = None if args.development else EXPECTED_TOKENS
    selection = {}
    with context as run, tempfile.TemporaryDirectory(prefix='reference-', dir=out) as temporary:
        if reference is None:
            reference = str(Path(temporary) / 'reference.u32le')
        def trial(stage, case, repeat=0, diagnostic=False, write_reference=False):
            name = f'trial-{len(order) + 1:03d}'
            record = {'trial_id': name, 'stage': stage, 'case': case, 'repeat': repeat, 'diagnostic': diagnostic}
            order.append(record)
            common.save_json(out / 'schedule.json', order)
            job = {**record, 'tokenizer_prefix': prefix, 'valid_file': valid, 'cpus': cpus, 'workers': workers,
                   'reference': reference, 'expected_tokens': expected, 'write_reference': write_reference,
                   'development': args.development}
            print(f'{name}: {stage} {case}; log={out / (name + ".log")}', flush=True)
            row = launch_job(job, out / (name + '.job.json'), out / (name + '.json'), out / (name + '.log'))
            row.update(record)
            common.save_json(out / (name + '.json'), row)
            rows.append(row)
            common.save_json(out / 'results.json', rows)
            write_report(out, rows, selection)
            log_rows(run, rows)
            return row
        try:
            if args.reference_tokens:
                expected = Path(reference).stat().st_size // 4
            else:
                generated = trial('reference', 'serial', write_reference=True)
                expected = generated['token_count']
            common.save_json(out / 'reference.json', {'sha256': common.sha256(reference), 'token_count': expected,
                                                     'provided': bool(args.reference_tokens)})
            selection = run_stages(args, trial)
            changed = [path for path, digest in manifest['input_sha256'].items() if common.sha256(path) != digest]
            source_changed = common.source_fingerprints() != manifest['source_sha256']
            common.save_json(out / 'integrity.json', {'changed_inputs': changed, 'source_changed': source_changed})
            if changed or source_changed:
                raise ValueError('Source/input changed during experiment; results ineligible')
            common.save_json(out / 'selection.json', selection)
            write_report(out, rows, selection)
            if run is not None:
                import wandb
                confirmation = next(row for row in rows if row['trial_id'] == selection['confirmation_trial'])
                sample = confirmation['first_200']
                print('First 200 IDs:', sample['ids'], flush=True)
                print('First 200 decoded:', repr(sample['decoded']), flush=True)
                run.log({'bpe/first_200_tokens': wandb.Table(columns=sample['columns'], data=sample['rows'])})
                run.summary.update({'completed': True, 'recommended': selection['recommended'],
                                    'bpe/valid_seconds': selection['confirmation_seconds'],
                                    'bpe/valid_token_count': confirmation['token_count'], 'bpe/roundtrip_passed': True,
                                    'paired_confirmation': selection['paired_confirmation'],
                                    'submission_provenance_verified': False})
                for name in ['manifest.json', 'results.json', 'results.csv', 'schedule.json', 'reference.json',
                             'selection.json', 'integrity.json', 'report.md']:
                    run.save(str(out / name), base_path=str(out), policy='now')
                for row in rows:
                    run.save(str(out / (row['trial_id'] + '.json')), base_path=str(out), policy='now')
                common.save_json(out / 'wandb.json', {'url': run.url})
        except BaseException as error:
            common.save_json(out / 'failure.json', {'completed': False, 'error': repr(error)})
            if run is not None:
                run.summary.update({'completed': False, 'error': str(error)})
            raise
    print(f'Completed: {out / "report.md"}', flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer-prefix', default='outputs/8b68b68e0cbe/tokenizer')
    parser.add_argument('--valid-file', default='/data/assignment1-data/owt_valid.txt')
    parser.add_argument('--reference-tokens')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261015)
    parser.add_argument('--cpu-ids', nargs='+', type=int)
    parser.add_argument('--diagnostics', action='store_true')
    parser.add_argument('--output-dir', default='outputs/task1-transport-exploration')
    parser.add_argument('--development', action='store_true')
    parser.add_argument('--development-workers', type=int, default=2)
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--worker-job', help=argparse.SUPPRESS)
    parser.add_argument('--worker-result', help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    restrict_environment()
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Transport exploration interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    if args.worker_job:
        worker(json.loads(Path(args.worker_job).read_text(encoding='utf-8')), args.worker_result)
    else:
        execute(args)


if __name__ == '__main__':
    main()
