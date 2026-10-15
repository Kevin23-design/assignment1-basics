"""Task-one full encode/decode/compare exploration, separate from token-file benchmarks."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import cProfile
import csv
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import signal
import statistics
import subprocess
import sys
import tempfile
import time

from bpe import BPE, CACHE_CAPACITY
from parallel_bpe_encoding import (
    DiagnosticCache, affinity, read_uint32, restrict_environment, rss_bytes, validate_tokenizer, write_uint32,
)
from task1_memory_encoding import InsufficientBoundaries, decode_blocked, encode_parallel_memory

ROOT = Path(__file__).resolve().parent
EXPECTED_TOKENS = 66401098
BLOCK = 262144
SKIP = {'.git', '.venv', 'venv', 'outputs', 'wandb', 'data', 'datasets',
        'checkpoints', '__pycache__', '.pytest_cache'}


def save_json(path, value):
    path = Path(path)
    partial = Path(str(path) + '.partial')
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    partial.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprints():
    result = {}
    for folder, directories, names in os.walk(ROOT):
        directories[:] = sorted(d for d in directories if d not in SKIP)
        for name in sorted(names):
            path = Path(folder) / name
            if path.suffix in {'.py', '.sh', '.ipynb', '.cu', '.c', '.cpp', '.h', '.rs'} or name in {
                'pyproject.toml', 'uv.lock', '.python-version'
            }:
                result[str(path.relative_to(ROOT))] = sha256(path)
    return result


def check_reference(ids, path, write=False):
    if write:
        with open(path, 'wb') as handle:
            for start in range(0, len(ids), BLOCK):
                write_uint32(handle, ids[start:start + BLOCK])
        return
    if Path(path).stat().st_size != len(ids) * 4:
        raise ValueError('Reference token count differs')
    with open(path, 'rb') as handle:
        start = 0
        for values in read_uint32(handle):
            if any(token_id != ids[start + i] for i, token_id in enumerate(values)):
                raise ValueError(f'Full token ID comparison failed in block starting at {start}')
            start += len(values)


def measure(text, tokenizer, spec, prefix, separator, cpus, diagnostic=False, profile_dir=None):
    if tokenizer._cache:
        raise RuntimeError('Each timed evaluation must start with an empty tokenizer cache')
    if diagnostic:
        tokenizer._cache = DiagnosticCache()
    profiler = cProfile.Profile() if profile_dir else None
    details = {}
    if profiler:
        profiler.enable()
    try:
        started = time.perf_counter()
        if spec['method'] == 'single':
            ids = tokenizer.encode(text)
        else:
            ids, details = encode_parallel_memory(
                text, tokenizer, prefix, spec['workers'], spec['partitions_per_worker'], separator,
                cpus, diagnostic=diagnostic, profile_dir=profile_dir)
        encoded_at = time.perf_counter()
        if spec['decode_block_size']:
            decode_diagnostics = {} if diagnostic else None
            decoded = decode_blocked(tokenizer, ids, spec['decode_block_size'], diagnostics=decode_diagnostics)
            if diagnostic:
                details['decode_diagnostics'] = decode_diagnostics
        else:
            decoded = tokenizer.decode(ids)
        decoded_at = time.perf_counter()
        passed = decoded == text
        ended_at = time.perf_counter()
    finally:
        if profiler:
            profiler.disable()
            profiler.dump_stats(str(Path(profile_dir) / 'parent.prof'))
    result = {**spec, **details, 'encode_seconds': encoded_at - started,
              'decode_seconds': decoded_at - encoded_at, 'compare_seconds': ended_at - decoded_at,
              'total_seconds': ended_at - started, 'roundtrip_passed': passed, 'token_count': len(ids),
              'parent_peak_rss_bytes': rss_bytes(), 'cpu_affinity': cpus,
              'returned_type': type(ids).__name__, 'cache_capacity': CACHE_CAPACITY,
              'cache_policy': 'clear_all', 'diagnostic': diagnostic}
    if diagnostic and spec['method'] == 'single':
        stats = dict(tokenizer._cache.stats)
        stats['hit_rate'] = stats['hits'] / stats['queries'] if stats['queries'] else 0.
        result['cache_diagnostics'] = stats
    return ids, result


def first_tokens(tokenizer, ids, text):
    raw = text.encode('utf-8')
    rows, offset = [], 0
    for index, token_id in enumerate(ids[:200]):
        piece = tokenizer.vocab[token_id]
        end = offset + len(piece)
        try:
            piece.decode('utf-8', errors='strict')
            standalone_valid = True
        except UnicodeDecodeError:
            standalone_valid = False
        rows.append([index, token_id, piece.hex(), repr(piece), repr(tokenizer.decode([token_id])),
                     standalone_valid, offset, end, piece == raw[offset:end]])
        offset = end
    return {'columns': ['position', 'token_id', 'token_hex', 'token_bytes', 'decoded_token',
                        'standalone_utf8_valid', 'byte_start', 'byte_end', 'matches_input_bytes'],
            'rows': rows, 'ids': ids[:200], 'decoded': tokenizer.decode(ids[:200])}


def worker(job, result_path):
    restrict_environment()
    affinity(job['cpus'])
    tokenizer = BPE()
    tokenizer.load(job['tokenizer_prefix'])
    validate_tokenizer(tokenizer)
    if CACHE_CAPACITY != 1048576:
        raise ValueError('This experiment fixes the production cache capacity at 1M')
    with open(job['valid_file'], encoding='utf-8', newline='') as handle:
        text = handle.read()
    if job.get('profile_dir'):
        Path(job['profile_dir']).mkdir(exist_ok=False)
    try:
        ids, result = measure(text, tokenizer, job['spec'], job['tokenizer_prefix'], job['separator'],
                              job['cpus'], diagnostic=job['diagnostic'], profile_dir=job.get('profile_dir'))
    except InsufficientBoundaries as error:
        save_json(result_path, {**job['spec'], 'status': 'unavailable', 'reason': str(error),
                                'diagnostic': job['diagnostic']})
        return
    result['status'] = 'ok'
    try:
        if not isinstance(ids, list) or not result['roundtrip_passed']:
            raise ValueError('Must return a complete list[int] and restore the exact input')
        if job['expected_tokens'] is not None and len(ids) != job['expected_tokens']:
            raise ValueError(f'Expected {job["expected_tokens"]} tokens, got {len(ids)}')
        check_reference(ids, job['reference'], write=job['write_reference'])
        result['ids_match_reference'] = None if job['write_reference'] else True
        result['reference_generated'] = job['write_reference']
        result['first_200'] = first_tokens(tokenizer, ids, text)
        result['special_tokens'] = tokenizer.special_tokens
        result['special_token_ids'] = tokenizer._special_ids
    except BaseException as error:
        result.update({'status': 'failed', 'error': repr(error)})
        save_json(result_path, result)
        raise
    save_json(result_path, result)


def config_key(spec):
    return spec['method'], spec['workers'], spec['partitions_per_worker'], spec['decode_block_size']


def choose(rows, repeats, tolerance=.03):
    groups = {}
    for row in rows:
        if not row.get('diagnostic'):
            groups.setdefault(config_key(row), []).append(row)
    candidates = []
    for key, group in groups.items():
        if len(group) != repeats or any(row['status'] != 'ok' for row in group):
            continue
        median = statistics.median(row['total_seconds'] for row in group)
        candidates.append((key, median))
    if not candidates:
        raise ValueError('No available configuration passed every repeat')
    fastest = min(seconds for _, seconds in candidates)
    near = [(key, seconds) for key, seconds in candidates if seconds <= fastest * (1 + tolerance)]
    selected, median = min(near, key=lambda item: (
        item[0][1], item[0][0] != 'single', item[0][2], item[0][3] != 0, item[1]))
    method, workers, parts, decoder = selected
    return {'method': method, 'workers': workers, 'partitions_per_worker': parts,
            'decode_block_size': decoder}, median


def schedule(specs, repeats, seed):
    rng = random.Random(seed)
    result = []
    for repeat in range(1, repeats + 1):
        order = list(specs)
        rng.shuffle(order)
        result.extend((spec, repeat) for spec in order)
    return result


def paired_decision(baseline_rows, candidate_rows, repeats, minimum_gain=.03):
    """Require complete matched rounds, >=3% median improvement and >=80% paired wins."""
    if any(len(rows) != repeats for rows in (baseline_rows, candidate_rows)):
        raise ValueError('Paired confirmation needs every baseline and candidate repeat')
    groups = []
    for rows in (baseline_rows, candidate_rows):
        if any(row['status'] != 'ok' or row.get('diagnostic') or not row.get('roundtrip_passed')
               or row.get('ids_match_reference') is not True for row in rows):
            raise ValueError('Paired confirmation requires normal, exact, successful trials')
        group = {row['repeat']: row for row in rows}
        if set(group) != set(range(1, repeats + 1)):
            raise ValueError('Paired confirmation has missing or duplicate rounds')
        groups.append(group)
    baseline, candidate = groups
    before = statistics.median(row['total_seconds'] for row in baseline_rows)
    after = statistics.median(row['total_seconds'] for row in candidate_rows)
    wins = sum(candidate[index]['total_seconds'] < baseline[index]['total_seconds'] for index in baseline)
    required_wins = (4 * repeats + 4) // 5
    promote = after <= before * (1 - minimum_gain) and wins >= required_wins
    return {'promote_candidate': promote, 'baseline_median_seconds': before,
            'candidate_median_seconds': after, 'median_improvement_fraction': 1 - after / before,
            'paired_wins': wins, 'required_paired_wins': required_wins,
            'minimum_improvement_fraction': minimum_gain,
            'paired_seconds': [{'repeat': index, 'baseline': baseline[index]['total_seconds'],
                               'candidate': candidate[index]['total_seconds']} for index in sorted(baseline)],
            'reason': 'threshold_and_consistency_passed' if promote else 'retain_current_configuration'}


def run_refinement(args, trial):
    baseline = {'method': 'parallel', 'workers': max(args.workers), 'partitions_per_worker': 4,
                'decode_block_size': 262144}
    selections = {'baseline': baseline}
    variants = [{**baseline, 'decode_block_size': size} for size in sorted(set(args.decode_block_sizes))]
    group = [trial('refine-decoder', spec, repeat)
             for spec, repeat in schedule(variants, args.repeats, args.seed)]
    selected, median = choose(group, args.repeats, tolerance=0.)
    selections['decoder'] = {**selected, 'median_seconds': median}
    variants = [{**selected, 'partitions_per_worker': size} for size in sorted(set(args.partition_multipliers))]
    group = [trial('refine-partitions', spec, repeat)
             for spec, repeat in schedule(variants, args.repeats, args.seed + 1)]
    candidate, median = choose(group, args.repeats, tolerance=0.)
    selections['partitions'] = {**candidate, 'median_seconds': median}
    if args.diagnostics:
        trial('diagnostic-baseline', baseline, diagnostic=True)
        trial('diagnostic-candidate', candidate, diagnostic=True)
    baseline_rows, candidate_rows = [], []
    rng = random.Random(args.seed + 2)
    for repeat in range(1, args.repeats + 1):
        pair = [('baseline', baseline, baseline_rows), ('candidate', candidate, candidate_rows)]
        rng.shuffle(pair)
        for label, spec, rows in pair:
            rows.append(trial('paired-' + label, spec, repeat))
    decision = paired_decision(baseline_rows, candidate_rows, args.repeats)
    if config_key(candidate) == config_key(baseline):
        decision.update(promote_candidate=False, reason='candidate_matches_current_configuration')
    selections['paired_confirmation'] = decision
    promoted = decision['promote_candidate']
    selected = candidate if promoted else baseline
    # Keep one actual independent run as the reported task-one score; never report the best trial.
    confirmation = (candidate_rows if promoted else baseline_rows)[-1]
    return selected, selections, confirmation


def write_report(out, rows, selections):
    fields = ['stage', 'repeat', 'method', 'workers', 'partitions_per_worker', 'decode_block_size',
              'status', 'diagnostic', 'total_seconds', 'encode_seconds', 'decode_seconds', 'compare_seconds',
              'split_seconds', 'startup_seconds', 'worker_encode_ipc_join_seconds', 'assembly_seconds',
              'token_count', 'roundtrip_passed', 'ids_match_reference', 'parent_peak_rss_bytes',
              'worker_peak_rss_max', 'worker_peak_rss_sum', 'reason']
    with (out / 'results.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# 任务一完整验证效率探索',
             '计时为完整内存文本 encode → 完整 list[int] → 全文 decode → 原文比较。',
             '并行分片、worker 启动/加载/退出、文本/uint32 IPC、Python 列表组装全部计入。',
             '读取主输入和加载父分词器在计时前；ID 精确对照、前 200 token 和上传在计时后。',
             '| 阶段 | 方案 | workers | 每 worker 任务数 | 解码块 | 有效次数 | 中位数(s) | 最小(s) | 最大(s) |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    groups = {}
    for row in rows:
        if not row['diagnostic']:
            groups.setdefault((row['stage'], *config_key(row)), []).append(row)
    for (stage, method, workers, parts, decoder), group in groups.items():
        times = [r['total_seconds'] for r in group if r['status'] == 'ok']
        if times:
            lines.append(f'| {stage} | {method} | {workers} | {parts} | {decoder} | {len(times)} | '
                         f'{statistics.median(times):.3f} | {min(times):.3f} | {max(times):.3f} |')
        else:
            lines.append(f'| {stage} | {method} | {workers} | {parts} | {decoder} | 0 | 不可用 | — | — |')
    if selections:
        lines += ['', '## 已选配置', '```json', json.dumps(selections, ensure_ascii=False, indent=2), '```']
    diagnostics = [r for r in rows if r['diagnostic'] and r['status'] == 'ok']
    if diagnostics:
        lines += ['', '## 独立诊断（不参与排名）', '| 方案 | workers | 查询 | 命中率 | 清空 |', '|---|---:|---:|---:|---:|']
        for row in diagnostics:
            stats = row.get('cache_diagnostics', {})
            lines.append(f"| {row['method']} | {row['workers']} | {stats.get('queries')} | "
                         f"{stats.get('hit_rate', 0):.2%} | {stats.get('clears')} |")
        for row in diagnostics:
            lines += ['', f"### {row.get('trial_id', row['stage'])}：分项诊断", '```json',
                      json.dumps({key: row[key] for key in ('pipeline_diagnostics', 'decode_diagnostics')
                                  if key in row}, ensure_ascii=False, indent=2), '```']
            if row.get('worker_metadata'):
                lines += ['', '| worker | 任务 | 字符数 | token 数 | 编码(s) | 打包(s) | send(s) | 取任务等待(s) |',
                          '|---:|---:|---:|---:|---:|---:|---:|---:|']
                for item in row['worker_metadata']:
                    lines.append(f"| {item['worker_id']} | {item['tasks']} | {item.get('characters', '—')} | "
                                 f"{item.get('tokens', '—')} | {item.get('encode_seconds', 0):.3f} | "
                                 f"{item.get('pack_seconds', 0):.3f} | {item.get('send_seconds', 0):.3f} | "
                                 f"{item.get('job_wait_seconds', 0):.3f} |")
    lines += ['', 'worker_encode_ipc_join_seconds 是重叠阶段墙钟时间，不是纯合并计算时间。',
              '局部精调只有配对中位数改善至少 3% 且至少 80% 配对获胜时才推荐候选；否则保留当前配置。',
              '诊断的 send 包含序列化及阻塞，receive 包含等待及反序列化；各进程耗时不能相加。',
              '诊断解码为分离查找和拼接而额外物化字节引用，分项仅用于定位，不代表正常解码速度。',
              'worker_peak_rss_sum 是独立进程峰值之和，不是同时发生的实际峰值。',
              'CPU 数量不同的方案比较代表最多 16 核预算下的整套实现收益，不能解释为单核算法提速。',
              '本工具不证明 tokenizer 的完整训练集来源，不自动修改正式实现；不能用任务二写盘成绩代替。']
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def log_rows(run, rows):
    if run is None:
        return
    import wandb
    columns = ['stage', 'repeat', 'method', 'workers', 'partitions_per_worker', 'decode_block_size',
               'diagnostic', 'status', 'total_seconds', 'encode_seconds', 'decode_seconds', 'compare_seconds',
               'token_count', 'ids_match_reference', 'roundtrip_passed', 'parent_peak_rss_bytes',
               'worker_peak_rss_max', 'worker_peak_rss_sum']
    run.log({'exploration/trials': wandb.Table(columns=columns, data=[[r.get(k) for k in columns] for r in rows])})


def launch_job(job, job_path, result_path, log_path):
    save_json(job_path, job)
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                    '--worker-job', str(job_path), '--worker-result', str(result_path)],
                                   env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, cwd=ROOT)
        try:
            code = process.wait()
            if code:
                raise RuntimeError(f'Trial failed (exit {code}); inspect {log_path}')
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    return json.loads(result_path.read_text(encoding='utf-8'))


def execute(args):
    restrict_environment()
    if not args.development and args.no_wandb:
        raise ValueError('--no-wandb is restricted to --development small fixtures')
    if args.repeats <= 0 or not args.workers or not all(1 <= n <= 16 for n in args.workers):
        raise ValueError('Repeats must be positive; worker counts must be 1..16')
    if min(args.partition_multipliers) <= 0 or min(args.decode_block_sizes) <= 0:
        raise ValueError('Partition multipliers and decode blocks must be positive')
    if args.suite == 'local-refinement' and not args.development:
        if args.workers != [16] or args.repeats < 5:
            raise ValueError('Cloud refinement requires 16 workers and at least five repeats')
        if 262144 not in args.decode_block_sizes or 4 not in args.partition_multipliers:
            raise ValueError('Refinement must include the current decoder and partition controls')
    if args.cpu_ids is not None and (len(set(args.cpu_ids)) != len(args.cpu_ids) or not 1 <= len(args.cpu_ids) <= 16):
        raise ValueError('--cpu-ids must contain 1..16 distinct logical CPU indices')
    max_workers = max(args.workers)
    if hasattr(os, 'sched_getaffinity'):
        allowed = sorted(os.sched_getaffinity(0))
        requested = args.cpu_ids if args.cpu_ids is not None else allowed[:max_workers]
        if len(requested) < max_workers or not set(requested) <= set(allowed):
            raise ValueError('Requested workers exceed the allowed CPU allocation; no automatic downgrade')
        cpus = requested[:max_workers]
        affinity(cpus, include_descendants=True)
    elif args.development and args.cpu_ids is None:
        cpus = []
    else:
        raise RuntimeError('Cloud experiments require Linux CPU affinity')
    out = Path(args.output_dir).resolve()
    if not args.development and not out.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Cloud outputs must be inside the personal project outputs directory')
    if out.exists():
        raise FileExistsError(out)
    prefix = str(Path(args.tokenizer_prefix).resolve())
    valid = str(Path(args.valid_file).resolve())
    inputs = [valid] + [prefix + suffix for suffix in ['.vocab.json', '.merges.json', '.config.json']]
    reference = str(Path(args.reference_tokens).resolve()) if args.reference_tokens else None
    if reference:
        inputs.append(reference)
        if Path(reference).stat().st_size % 4:
            raise ValueError('Reference must be a little-endian uint32 token file')
        if not args.development and Path(reference).stat().st_size != EXPECTED_TOKENS * 4:
            raise ValueError('Reference size differs from full OWT validation token count')
    tokenizer = BPE()
    tokenizer.load(prefix)
    validate_tokenizer(tokenizer)
    if args.separator not in tokenizer.special_tokens:
        raise ValueError('Configured split token is absent from tokenizer')
    expected = None if args.development else EXPECTED_TOKENS
    manifest = {'arguments': vars(args), 'python': sys.version, 'platform': platform.platform(),
                'started_utc': datetime.now(UTC).isoformat(), 'cpu_pool': cpus, 'cpu_limit': 16,
                'input_sha256': {str(path): sha256(path) for path in inputs},
                'source_sha256': source_fingerprints(), 'special_tokens': tokenizer.special_tokens,
                'special_token_ids': tokenizer._special_ids, 'cache_capacity': 1048576,
                'cache_policy': 'clear_all', 'max_piece_characters': 128,
                'timing': 'split/start/load/encode/IPC/list assembly/join/decode/exact text compare'}
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / 'manifest.json', manifest)
    if args.no_wandb:
        context = nullcontext(None)
    else:
        from wandb_record import experiment
        context = experiment('bpe-task1-efficiency-' + ('refinement' if args.suite == 'local-refinement'
                                                     else 'exploration'), manifest, root=ROOT)
    rows, execution_order, selections = [], [], {}
    original = {'method': 'single', 'workers': 1, 'partitions_per_worker': 1, 'decode_block_size': 0}
    save_json(out / 'results.json', rows)
    with context as run, tempfile.TemporaryDirectory(prefix='reference-', dir=out) as temporary:
        if reference is None:
            reference = str(Path(temporary) / 'reference.u32le')
        def trial(stage, spec, repeat=0, diagnostic=False, write_reference=False):
            number = len(execution_order) + 1
            name = f'trial-{number:03d}'
            record = {'trial_id': name, 'stage': stage, 'repeat': repeat, 'diagnostic': diagnostic, **spec}
            execution_order.append(record)
            save_json(out / 'schedule.json', execution_order)
            job = {'spec': spec, 'tokenizer_prefix': prefix, 'valid_file': valid, 'separator': args.separator,
                   'cpus': cpus[:spec['workers']], 'reference': reference, 'expected_tokens': expected,
                   'diagnostic': diagnostic, 'write_reference': write_reference,
                   'profile_dir': str(out / (name + '-profiles')) if diagnostic else None}
            print(f'{name}: {stage} {spec}; log={out / (name + ".log")}', flush=True)
            affinity(job['cpus'], include_descendants=True)
            try:
                result = launch_job(job, out / (name + '.job.json'), out / (name + '.json'), out / (name + '.log'))
            finally:
                affinity(cpus, include_descendants=True)
            result.update(record)
            save_json(out / (name + '.json'), result)
            rows.append(result)
            save_json(out / 'results.json', rows)
            write_report(out, rows, selections)
            log_rows(run, rows)
            return result
        try:
            if not args.reference_tokens:
                reference_result = trial('reference', original, write_reference=True)
                if args.development:
                    expected = reference_result['token_count']
            elif args.development:
                expected = Path(reference).stat().st_size // 4
            save_json(out / 'reference.json', {'sha256': sha256(reference), 'token_count': expected,
                                               'provided': bool(args.reference_tokens)})
            if args.suite == 'local-refinement':
                selected, selections, confirmation = run_refinement(args, trial)
            else:
                selected, selections, confirmation = run_broad_search(args, trial, original)
            if confirmation['status'] != 'ok':
                raise ValueError('Independent confirmation did not complete successfully')
            changed = [path for path, digest in manifest['input_sha256'].items() if sha256(path) != digest]
            source_changed = source_fingerprints() != manifest['source_sha256']
            save_json(out / 'integrity.json', {'changed_inputs': changed, 'source_changed': source_changed})
            if changed or source_changed:
                raise ValueError('Source or input changed during exploration; results are not eligible')
            final = {'completed': True, 'suite': args.suite, 'selected': selected, 'selections': selections,
                     'confirmation_seconds': confirmation['total_seconds'],
                     'confirmation_trial': confirmation['trial_id'], 'submission_provenance_verified': False}
            save_json(out / 'selection.json', final)
            write_report(out, rows, final)
            if run is not None:
                import wandb
                sample = confirmation['first_200']
                run.log({'bpe/first_200_tokens': wandb.Table(columns=sample['columns'], data=sample['rows'])})
                print('First 200 IDs:', sample['ids'], flush=True)
                print('First 200 decoded:', repr(sample['decoded']), flush=True)
                run.summary.update({'completed': True, 'bpe/valid_seconds': confirmation['total_seconds'],
                                    'bpe/valid_token_count': confirmation['token_count'],
                                    'bpe/roundtrip_passed': confirmation['roundtrip_passed'],
                                    'ids_match_reference': confirmation['ids_match_reference'],
                                    'selected_method': selected, 'submission_provenance_verified': False})
                if args.suite == 'local-refinement':
                    run.summary['paired_confirmation'] = selections['paired_confirmation']
                for name in ['manifest.json', 'reference.json', 'results.json', 'results.csv', 'schedule.json',
                             'selection.json', 'integrity.json', 'report.md']:
                    run.save(str(out / name), base_path=str(out), policy='now')
                if args.diagnostics:
                    for path in sorted(out.glob('trial-*-profiles/*.prof')):
                        run.save(str(path), base_path=str(out), policy='now')
                    for row in rows:
                        if row['diagnostic']:
                            run.save(str(out / (row['trial_id'] + '.json')), base_path=str(out), policy='now')
                save_json(out / 'wandb.json', {'url': run.url})
        except BaseException as error:
            save_json(out / 'failure.json', {'completed': False, 'error': repr(error)})
            if run is not None:
                run.summary.update({'completed': False, 'error': str(error)})
            raise
    print(f'Completed: {out / "report.md"}', flush=True)


def run_broad_search(args, trial, original):
    selections = {}
    stage1 = [original] + [{'method': 'parallel', 'workers': count, 'partitions_per_worker': 1,
                            'decode_block_size': 0} for count in sorted(set(args.workers))]
    group = [trial('workers', spec, repeat) for spec, repeat in schedule(stage1, args.repeats, args.seed)]
    selected, median = choose(group, args.repeats)
    selections['workers'] = {**selected, 'median_seconds': median}
    if selected['method'] == 'parallel':
        variants = [{**selected, 'partitions_per_worker': multiplier}
                    for multiplier in sorted(set([1, *args.partition_multipliers]))]
        group = [trial('partitions', spec, repeat) for spec, repeat in schedule(variants, args.repeats, args.seed + 1)]
        selected, median = choose(group, args.repeats, tolerance=0.)
        selections['partitions'] = {**selected, 'median_seconds': median}
    variants = [{**selected, 'decode_block_size': size} for size in [0, *sorted(set(args.decode_block_sizes))]]
    group = [trial('decoder', spec, repeat) for spec, repeat in schedule(variants, args.repeats, args.seed + 2)]
    selected, median = choose(group, args.repeats, tolerance=0.)
    selections['decoder'] = {**selected, 'median_seconds': median}
    if args.diagnostics:
        trial('diagnostic-original', original, diagnostic=True)
        if config_key(selected) != config_key(original):
            trial('diagnostic-selected', selected, diagnostic=True)
    confirmation = trial('confirmation', selected)
    return selected, selections, confirmation


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['broad', 'local-refinement'], default='broad')
    parser.add_argument('--tokenizer-prefix', default='outputs/e9275a3b4a7a/tokenizer')
    parser.add_argument('--valid-file', default='/data/assignment1-data/owt_valid.txt')
    parser.add_argument('--reference-tokens')
    parser.add_argument('--workers', nargs='+', type=int)
    parser.add_argument('--repeats', type=int)
    parser.add_argument('--partition-multipliers', nargs='+', type=int)
    parser.add_argument('--decode-block-sizes', nargs='+', type=int)
    parser.add_argument('--separator', default='<|endoftext|>')
    parser.add_argument('--seed', type=int, default=20261015)
    parser.add_argument('--cpu-ids', nargs='+', type=int)
    parser.add_argument('--diagnostics', action='store_true')
    parser.add_argument('--output-dir')
    parser.add_argument('--development', action='store_true', help='Small fixtures only; not course submission evidence')
    parser.add_argument('--no-wandb', action='store_true', help='Development tests only')
    parser.add_argument('--worker-job', help=argparse.SUPPRESS)
    parser.add_argument('--worker-result', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    refinement = args.suite == 'local-refinement'
    if args.output_dir is None:
        args.output_dir = 'outputs/task1-refinement' if refinement else 'outputs/task1-efficiency'
    if args.workers is None:
        args.workers = [16] if refinement else [1, 2, 4, 8, 16]
    if args.repeats is None:
        args.repeats = 5 if refinement else 3
    if args.partition_multipliers is None:
        args.partition_multipliers = [2, 4, 8, 16] if refinement else [1, 2, 4]
    if args.decode_block_sizes is None:
        args.decode_block_sizes = [65536, 131072, 262144, 524288] if refinement else [65536, 262144, 1048576]
    return args


def main():
    args = parse_args()
    restrict_environment()
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Task-one exploration interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    if args.worker_job:
        worker(json.loads(Path(args.worker_job).read_text(encoding='utf-8')), args.worker_result)
    else:
        execute(args)


if __name__ == '__main__':
    main()
