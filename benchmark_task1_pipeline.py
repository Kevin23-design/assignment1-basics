"""Cold public BPE startup/decode ablations; loads existing tokenizer, never trains."""
from __future__ import annotations

import argparse
from array import array
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

from bpe import BPE
import benchmark_task1_efficiency as common
import benchmark_task1_transport as previous
from parallel_bpe_encoding import affinity, restrict_environment, rss_bytes, validate_tokenizer
from task1_pipeline_runtime import CASES, DECODERS, selected_decoder, selected_pipeline

ROOT = Path(__file__).resolve().parent
EXPECTED_TOKENS = 66401098


def read_ids(path):
    ids = []
    with open(path, 'rb') as handle:
        while raw := handle.read(262144 * 4):
            if len(raw) % 4:
                raise ValueError('Reference is not uint32 little-endian')
            block = array('I')
            if block.itemsize != 4:
                raise RuntimeError('Requires 32-bit unsigned arrays')
            block.frombytes(raw)
            if sys.byteorder != 'little':
                block.byteswap()
            ids.extend(block)
    return ids


def worker(job, result_path):
    if job['stage'] == 'reference':
        return previous.worker(job, result_path)
    restrict_environment()
    previous.validate_fixed_parameters()
    affinity(job['cpus'])
    import wandb  # noqa: F401 -- mirror the course caller, without a run in each subprocess
    tokenizer = BPE()
    tokenizer.load(job['tokenizer_prefix'])
    validate_tokenizer(tokenizer)
    with open(job['valid_file'], encoding='utf-8', newline='') as handle:
        text = handle.read()
    result = {'case': job['case'], 'decoder': job['decoder'], 'diagnostic': job['diagnostic'],
              'status': 'failed', 'cpu_affinity': job['cpus'], 'workers': job['workers'],
              'parts': job['workers'] * 16, 'cache_capacity': 1048576, 'decode_block_size': 131072,
              'timing_scope': 'decoder-only' if job['stage'] == 'decoder-screening' else 'full-public-interface'}
    try:
        if tokenizer._cache:
            raise RuntimeError('Trial must start with an empty BPE cache')
        if job['stage'] == 'decoder-screening':
            ids = read_ids(job['reference'])
            with selected_decoder(job['decoder']):
                started = encoded = time.perf_counter()
                decoded = tokenizer.decode(ids)
                decoded_at = time.perf_counter()
                passed = decoded == text
                ended = time.perf_counter()
            metrics = {}
            measured = decoded_at - encoded  # Selection uses decode only, not the comparison.
        else:
            with selected_pipeline(job['case'], job['decoder'], job['diagnostic'],
                                   job['workers'] if job['development'] else None) as (metrics, calls):
                started = time.perf_counter()
                ids = tokenizer.encode(text)
                encoded = time.perf_counter()
                decoded = tokenizer.decode(ids)
                decoded_at = time.perf_counter()
                passed = decoded == text
                ended = time.perf_counter()
                if calls != ['D']:
                    raise RuntimeError('Public encode bypassed the requested backend')
            measured = ended - started
        result.update(total_seconds=ended - started, measured_seconds=measured,
                      encode_seconds=encoded - started, decode_seconds=decoded_at - encoded,
                      compare_seconds=ended - decoded_at, token_count=len(ids), returned_type=type(ids).__name__,
                      roundtrip_passed=passed, parent_peak_rss_bytes=rss_bytes(), transport=metrics)
        if not isinstance(ids, list) or not passed or len(ids) != job['expected_tokens']:
            raise ValueError('Incomplete IDs, wrong count or failed exact roundtrip')
        common.check_reference(ids, job['reference'])
        result.update(status='ok', ids_match_reference=True, first_200=common.first_tokens(tokenizer, ids, text))
        metadata = metrics.get('worker_metadata', [])
        if metadata:
            result.update(worker_peak_rss_max=max(w['peak_rss_bytes'] for w in metadata),
                          worker_peak_rss_sum=sum(w['peak_rss_bytes'] for w in metadata))
    except BaseException as error:
        result['error'] = repr(error)
        common.save_json(result_path, result)
        raise
    common.save_json(result_path, result)


def check_row(row):
    if (row.get('status') != 'ok' or row.get('ids_match_reference') is not True
            or row.get('roundtrip_passed') is not True):
        raise ValueError('Candidate failed exact correctness checks')


def medians(rows, cases, repeats, key):
    for row in rows:
        check_row(row)
    result = {}
    for case in cases:
        group = [r for r in rows if r['case'] == case]
        if len(group) != repeats:
            raise ValueError('Incomplete screening rounds')
        result[case] = statistics.median(r[key] for r in group)
    return result


def run_stages(args, trial):
    rng, rows = random.Random(args.seed), []
    for repeat in range(1, args.decoder_repeats + 1):
        order = list(DECODERS)
        rng.shuffle(order)
        rows.extend(trial('decoder-screening', case, repeat, decoder=case) for case in order)
    decode_medians = medians(rows, DECODERS, args.decoder_repeats, 'decode_seconds')
    nominee = min(DECODERS[1:], key=lambda c: (decode_medians[c], c))
    if args.diagnostics:
        trial('diagnostic-baseline', 'D0', decoder='C0', diagnostic=True)
    rows = []
    for repeat in range(1, args.repeats + 1):
        order = list(CASES)
        rng.shuffle(order)
        rows.extend(trial('screening', case, repeat, decoder=nominee if case in ('D2', 'D3') else 'C0')
                    for case in order)
    full_medians = medians(rows, CASES, args.repeats, 'total_seconds')
    candidate = min(CASES, key=lambda c: (full_medians[c], c))
    if args.diagnostics and candidate != 'D0':
        trial('diagnostic-candidate', candidate, decoder=nominee if candidate in ('D2', 'D3') else 'C0',
              diagnostic=True)
    decision = {'promote_candidate': False, 'reason': 'candidate_matches_current_configuration'}
    if candidate != 'D0':
        baseline_rows, candidate_rows = [], []
        rng = random.Random(args.seed + 1)
        for repeat in range(1, args.repeats + 1):
            pair = [('paired-baseline', 'D0', baseline_rows), ('paired-candidate', candidate, candidate_rows)]
            rng.shuffle(pair)
            for stage, case, target in pair:
                target.append(trial(stage, case, repeat, decoder=nominee if case in ('D2', 'D3') else 'C0'))
        decision = common.paired_decision(baseline_rows, candidate_rows, args.repeats)
    recommended = candidate if decision['promote_candidate'] else 'D0'
    confirmed = trial('confirmation', recommended, decoder=nominee if recommended in ('D2', 'D3') else 'C0')
    check_row(confirmed)
    return {'completed': True, 'decoder_nominee': nominee, 'decoder_medians_seconds': decode_medians,
            'screening_medians_seconds': full_medians, 'candidate': candidate, 'recommended': recommended,
            'recommended_decoder': confirmed['decoder'], 'paired_confirmation': decision,
            'confirmation_trial': confirmed['trial_id'], 'confirmation_seconds': confirmed['total_seconds'],
            'submission_provenance_verified': False}


def historical_inputs(directory, current):
    directory = Path(directory).resolve()
    manifest_path, selection_path = directory / 'manifest.json', directory / 'selection.json'
    prior = json.loads(manifest_path.read_text())
    selection = json.loads(selection_path.read_text())
    if not selection.get('completed') or selection.get('recommended') != 'D':
        raise ValueError('Previous results must be completed and recommend D')
    fixed = {'workers': 16, 'parts': 256, 'cache_capacity': 1048576, 'decode_block_size': 131072}
    if any(prior.get(key) != value for key, value in fixed.items()):
        raise ValueError('Previous D resource/configuration differs from fixed baseline')
    old = prior['input_sha256']
    names = [Path(prior['arguments']['valid_file']).name]
    names.extend(Path(prior['arguments']['tokenizer_prefix']).name + suffix
                 for suffix in ('.vocab.json', '.merges.json', '.config.json'))
    for name, (_, digest) in zip(names, current.items(), strict=True):
        candidates = [value for path, value in old.items() if Path(path).name == name]
        if len(candidates) != 1 or candidates[0] != digest:
            raise ValueError(f'Input/tokenizer fingerprint differs from previous experiment: {name}')
    return {'directory': str(directory), 'manifest_sha256': common.sha256(manifest_path),
            'selection_sha256': common.sha256(selection_path), 'confirmation_seconds': selection['confirmation_seconds'],
            'note': 'Historical score only; new D0 pairing decides promotion'}


def write_report(out, rows, selection):
    columns = ['trial_id', 'stage', 'case', 'decoder', 'repeat', 'diagnostic', 'timing_scope', 'status',
               'total_seconds', 'measured_seconds', 'encode_seconds', 'decode_seconds', 'compare_seconds',
               'token_count', 'ids_match_reference', 'roundtrip_passed', 'parent_peak_rss_bytes',
               'worker_peak_rss_max', 'worker_peak_rss_sum']
    with (out / 'results.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# 任务一：启动并发化 × 解码优化',
             '只加载已有分词器，不重新训练；历史 D 约 15.622 秒只作参考，选优使用本轮 D0 配对。',
             'C0 字典/map；C1 稠密表/map；C2 字典/itemgetter；C3 稠密表/itemgetter。',
             'D0 当前D；D1仅优化启动；D2仅优化解码；D3两项组合。',
             '查找表构造、索引准备、启动、传输、组装和退出全部计入对应接口耗时。',
             '', '## 解码专项（不属于完整成绩）', '',
             '| 解码 | 次数 | 解码中位数(s) | 最小(s) | 最大(s) |', '|---|---:|---:|---:|---:|']
    for case in DECODERS:
        values = [r['decode_seconds'] for r in rows if r['stage'] == 'decoder-screening' and r['case'] == case]
        if values:
            lines.append(f'| {case} | {len(values)} | {statistics.median(values):.3f} | {min(values):.3f} | {max(values):.3f} |')
    lines += ['', '## 完整 encode/list/decode/原文比较', '',
              '| 阶段 | 方案 | 解码 | 次数 | 总中位数(s) | 最小(s) | 最大(s) |', '|---|---|---|---:|---:|---:|---:|']
    groups = {}
    for row in rows:
        if row['case'] in CASES and not row['diagnostic']:
            groups.setdefault((row['stage'], row['case'], row['decoder']), []).append(row['total_seconds'])
    for (stage, case, decoder), values in groups.items():
        lines.append(f'| {stage} | {case} | {decoder} | {len(values)} | {statistics.median(values):.3f} | '
                     f'{min(values):.3f} | {max(values):.3f} |')
    if all(any(stage == 'screening' and candidate == case for stage, candidate, _ in groups) for case in CASES):
        times = {c: statistics.median(r['total_seconds'] for r in rows if r['stage'] == 'screening' and r['case'] == c)
                 for c in CASES}
        lines += ['', '筛选阶段消融（描述性比较，最终替换仍以配对复测为准）：',
                  f"仅启动 D1/D0 耗时比 {times['D1']/times['D0']:.3f}；仅解码 D2/D0 {times['D2']/times['D0']:.3f}；"
                  f"组合 D3/D0 {times['D3']/times['D0']:.3f}。"]
    lines += ['', '## 推荐与配对', '', '```json', json.dumps(selection, ensure_ascii=False, indent=2), '```',
              '', '## 独立诊断（不排名）']
    for row in rows:
        if row['diagnostic']:
            lines += ['', '```json', json.dumps({'trial_id': row['trial_id'], 'case': row['case'],
                                                'transport': row.get('transport')}, ensure_ascii=False, indent=2), '```']
    lines += ['', '跨进程时间重叠，不能相加；send/receive 包含序列化、阻塞和等待，不是纯通信成本。',
              '解码诊断为拆分查找与拼接而物化临时元组，与正常分配模式不同，不作速度成绩。',
              'worker 峰值 RSS 之和是独立峰值之和，不是同时发生的实际峰值。',
              '解码专项用同一参考ID装载方式筛选，完整实验再验证实际encode产物上的效果。',
              '不自动更新 dev，不替代课程入口正式成绩，不承诺达到5秒；不证明既有词表训练来源。']
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


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
            previous.stop_trial(process)
            raise
    return json.loads(result_path.read_text())


def execute(args):
    restrict_environment()
    previous.validate_fixed_parameters()
    if args.repeats < (1 if args.development else 5) or args.decoder_repeats < (1 if args.development else 3):
        raise ValueError('Cloud runs require at least five full and three decoder repeats')
    if args.no_wandb and not args.development:
        raise ValueError('--no-wandb is limited to development')
    workers = args.development_workers if args.development else 16
    if not 1 <= workers <= 16:
        raise ValueError('Workers must be 1..16')
    if hasattr(os, 'sched_getaffinity'):
        allowed = sorted(os.sched_getaffinity(0))
        cpus = sorted(args.cpu_ids) if args.cpu_ids is not None else allowed[:workers]
        if len(cpus) != workers or len(set(cpus)) != workers or not set(cpus) <= set(allowed):
            raise ValueError(f'Requires exactly {workers} distinct allowed CPUs; no automatic downgrade')
        affinity(cpus, include_descendants=True)
    elif args.development and args.cpu_ids is None:
        cpus = []
    else:
        raise RuntimeError('Cloud experiment requires Linux CPU affinity')
    out = Path(args.output_dir).resolve()
    if not args.development and not out.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Output must be in personal project outputs')
    if out.exists():
        raise FileExistsError(out)
    prefix, valid = str(Path(args.tokenizer_prefix).resolve()), str(Path(args.valid_file).resolve())
    inputs = [valid] + [prefix + s for s in ('.vocab.json', '.merges.json', '.config.json')]
    hashes = {path: common.sha256(path) for path in inputs}
    history = historical_inputs(args.previous_results_dir, hashes)
    for name in ('manifest.json', 'selection.json'):
        path = str((Path(args.previous_results_dir) / name).resolve())
        hashes[path] = common.sha256(path)
    reference = str(Path(args.reference_tokens).resolve()) if args.reference_tokens else None
    if reference:
        size = Path(reference).stat().st_size
        if size % 4 or (not args.development and size != EXPECTED_TOKENS * 4):
            raise ValueError('Reference must have exact uint32 full-validation size')
        hashes[reference] = common.sha256(reference)
    tokenizer = BPE()
    tokenizer.load(prefix)
    validate_tokenizer(tokenizer)
    manifest = {'arguments': vars(args), 'python': sys.version, 'started_utc': datetime.now(UTC).isoformat(),
                'input_sha256': hashes, 'source_sha256': common.source_fingerprints(), 'history': history,
                'cpu_affinity': cpus, 'workers': workers, 'parts': workers * 16, 'cache_capacity': 1048576,
                'decode_block_size': 131072, 'special_tokens': tokenizer.special_tokens,
                'timing': 'cold public encode/list/decode/compare including cleanup; decoder-only separately marked',
                'training': False, 'diagnostics_ranked': False}
    out.mkdir(parents=True, exist_ok=False)
    common.save_json(out / 'manifest.json', manifest)
    if args.no_wandb:
        context = nullcontext(None)
    else:
        from wandb_record import experiment
        context = experiment('bpe-task1-pipeline-exploration', manifest, root=ROOT)
    rows, order, selection = [], [], {}
    expected = None if args.development else EXPECTED_TOKENS
    with context as run, tempfile.TemporaryDirectory(prefix='reference-', dir=out) as temporary:
        reference = reference or str(Path(temporary) / 'reference.u32le')
        def trial(stage, case, repeat=0, decoder='C0', diagnostic=False, write_reference=False):
            name = f'trial-{len(order) + 1:03d}'
            record = {'trial_id': name, 'stage': stage, 'case': case, 'decoder': decoder,
                      'repeat': repeat, 'diagnostic': diagnostic}
            order.append(record)
            common.save_json(out / 'schedule.json', order)
            job = {**record, 'tokenizer_prefix': prefix, 'valid_file': valid, 'cpus': cpus, 'workers': workers,
                   'reference': reference, 'expected_tokens': expected, 'write_reference': write_reference,
                   'development': args.development}
            print(f'{name}: {stage} {case}/{decoder}', flush=True)
            row = launch_job(job, out / (name + '.job.json'), out / (name + '.json'), out / (name + '.log'))
            if stage != 'reference':
                check_row(row)
            row.update(record)
            common.save_json(out / (name + '.json'), row)
            rows.append(row)
            common.save_json(out / 'results.json', rows)
            write_report(out, rows, selection)
            return row
        try:
            if args.reference_tokens:
                expected = Path(reference).stat().st_size // 4
            else:
                generated = trial('reference', 'serial', write_reference=True)
                expected = generated['token_count']
            reference_digest = common.sha256(reference)
            common.save_json(out / 'reference.json', {'sha256': reference_digest, 'token_count': expected,
                                                     'provided': bool(args.reference_tokens)})
            selection = run_stages(args, trial)
            changed = [path for path, digest in hashes.items() if common.sha256(path) != digest]
            if common.sha256(reference) != reference_digest and reference not in changed:
                changed.append(reference)
            source_changed = common.source_fingerprints() != manifest['source_sha256']
            common.save_json(out / 'integrity.json', {'changed_inputs': changed, 'source_changed': source_changed})
            if changed or source_changed:
                raise ValueError('Source/input changed; all results ineligible')
            common.save_json(out / 'selection.json', selection)
            write_report(out, rows, selection)
            if run is not None:
                import wandb
                columns = ['trial_id', 'stage', 'case', 'decoder', 'repeat', 'diagnostic', 'timing_scope',
                           'status', 'total_seconds', 'encode_seconds', 'decode_seconds', 'compare_seconds',
                           'token_count', 'ids_match_reference', 'roundtrip_passed', 'parent_peak_rss_bytes',
                           'worker_peak_rss_max', 'worker_peak_rss_sum', 'cpu_affinity']
                run.log({'pipeline/trials': wandb.Table(columns=columns, data=[[r.get(k) for k in columns] for r in rows])})
                confirm = next(r for r in rows if r['trial_id'] == selection['confirmation_trial'])
                sample = confirm['first_200']
                run.log({'bpe/first_200_tokens': wandb.Table(columns=sample['columns'], data=sample['rows'])})
                run.summary.update({'completed': True, 'recommended': selection['recommended'],
                                    'recommended_decoder': selection['recommended_decoder'],
                                    'bpe/valid_seconds': selection['confirmation_seconds'],
                                    'bpe/valid_token_count': confirm['token_count'], 'bpe/roundtrip_passed': True,
                                    'paired_confirmation': selection['paired_confirmation'],
                                    'submission_provenance_verified': False})
                for path in out.glob('*'):
                    if path.suffix in ('.json', '.csv', '.md'):
                        run.save(str(path), base_path=str(out), policy='now')
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
    parser.add_argument('--previous-results-dir', default='outputs/task1-transport-exploration')
    parser.add_argument('--reference-tokens')
    parser.add_argument('--decoder-repeats', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261019)
    parser.add_argument('--cpu-ids', nargs='+', type=int)
    parser.add_argument('--diagnostics', action='store_true')
    parser.add_argument('--output-dir', default='outputs/task1-pipeline-exploration')
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
        raise KeyboardInterrupt('Pipeline exploration interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    if args.worker_job:
        worker(json.loads(Path(args.worker_job).read_text()), args.worker_result)
    else:
        execute(args)


if __name__ == '__main__':
    main()
