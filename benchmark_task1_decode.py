"""Cold decode exploration and unpatched ordinary-BPE confirmation; never trains."""
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
import time

from bpe import BPE
from bpe_decode_runtime import METHODS
import benchmark_task1_efficiency as common
import benchmark_task1_pipeline as pipeline
import benchmark_task1_transport as transport
from parallel_bpe_encoding import affinity, restrict_environment, rss_bytes, validate_tokenizer
from task1_decode_experiment import (
    integration_decision, make_snapshot, previous_inputs, selected_decoder, verify_snapshot,
)

ROOT = Path(__file__).resolve().parent
EXPECTED_TOKENS = 66401098


def worker(job, result_path):
    if job['stage'] == 'reference':
        return transport.worker(job, result_path)
    restrict_environment()
    transport.validate_fixed_parameters()
    affinity(job['cpus'])
    import wandb  # noqa: F401 -- same caller import context, without a per-trial run
    tokenizer = BPE()
    tokenizer.load(job['tokenizer_prefix'])
    validate_tokenizer(tokenizer)
    with open(job['valid_file'], encoding='utf-8', newline='') as handle:
        text = handle.read()
    result = {'status': 'failed', 'method': job['method'], 'diagnostic': job['diagnostic'],
              'cpu_affinity': job['cpus'], 'patching': True,
              'timing_scope': 'decoder-only' if job['stage'] == 'decoder-screening' else 'full-public-interface',
              'encode_workers': job['workers'], 'parts': job['workers'] * 16,
              'cache_capacity': 1048576, 'numpy_loaded_before': 'numpy' in sys.modules}
    try:
        if tokenizer._cache:
            raise RuntimeError('Every trial must start with an empty BPE cache')
        ids = pipeline.read_ids(job['reference']) if job['stage'] == 'decoder-screening' else None
        modules_before = set(sys.modules)
        with selected_decoder(job['method'], job['diagnostic'],
                              job['workers'] if job['development'] else None) as (stats, calls):
            began = time.perf_counter()
            if ids is None:
                ids = tokenizer.encode(text)
                if calls != ['D']:
                    raise RuntimeError('Encoder bypassed fixed D3 path')
            encoded = time.perf_counter()
            decoded = tokenizer.decode(ids)
            decoded_at = time.perf_counter()
            passed = decoded == text
            ended = time.perf_counter()
        if not isinstance(ids, list) or len(ids) != job['expected_tokens'] or not passed:
            raise ValueError('Wrong token count, output type or roundtrip')
        common.check_reference(ids, job['reference'])
        result.update(status='ok', total_seconds=ended - began, encode_seconds=encoded - began,
                      decode_seconds=decoded_at - encoded, compare_seconds=ended - decoded_at,
                      token_count=len(ids), returned_type=type(ids).__name__, roundtrip_passed=True,
                      ids_match_reference=True, transport=stats, parent_peak_rss_bytes=rss_bytes(),
                      modules_loaded_during=sorted(set(sys.modules) - modules_before),
                      first_200=common.first_tokens(tokenizer, ids, text))
        for phase, metadata in [('encode', stats.get('worker_metadata', [])),
                                ('decode', stats.get('decoder', {}).get('worker_metadata', []))]:
            if metadata:
                result[phase + '_worker_peak_rss_max'] = max(w['peak_rss_bytes'] for w in metadata)
                result[phase + '_worker_peak_rss_sum'] = sum(w['peak_rss_bytes'] for w in metadata)
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        common.save_json(result_path, result)


def run_stages(args, trial):
    rng, rows = random.Random(args.seed), []
    for repeat in range(1, args.decoder_repeats + 1):
        order = list(METHODS)
        rng.shuffle(order)
        rows.extend(trial('decoder-screening', method, repeat) for method in order)
    decoder_times = pipeline.medians([{**r, 'case': r['method']} for r in rows], METHODS,
                                     args.decoder_repeats, 'decode_seconds')
    nominees = ('R0', min(('N128', 'N1M'), key=lambda m: decoder_times[m]),
                min(('P4', 'P8', 'P16'), key=lambda m: decoder_times[m]))
    if args.diagnostics:
        trial('diagnostic-baseline', 'R0', diagnostic=True)
    rows = []
    for repeat in range(1, args.repeats + 1):
        order = list(nominees)
        rng.shuffle(order)
        rows.extend(trial('screening', method, repeat) for method in order)
    full_times = pipeline.medians([{**r, 'case': r['method']} for r in rows], nominees,
                                  args.repeats, 'total_seconds')
    candidate = min(nominees, key=lambda m: (full_times[m], m != 'R0', m))
    if args.diagnostics and candidate != 'R0':
        trial('diagnostic-candidate', candidate, diagnostic=True)
    decision = {'promote_candidate': False, 'reason': 'candidate_matches_baseline'}
    if candidate != 'R0':
        before, after = [], []
        for repeat in range(1, args.repeats + 1):
            pair = [('paired-baseline', 'R0', before), ('paired-candidate', candidate, after)]
            rng.shuffle(pair)
            for stage, method, target in pair:
                target.append(trial(stage, method, repeat))
        decision = common.paired_decision(before, after, args.repeats)
    recommended = candidate if decision['promote_candidate'] else 'R0'
    confirmation = trial('confirmation', recommended)
    pipeline.check_row(confirmation)
    return {'completed': True, 'recommended': recommended, 'candidate': candidate, 'nominees': nominees,
            'decoder_medians_seconds': decoder_times, 'screening_medians_seconds': full_times,
            'paired_confirmation': decision, 'confirmation_seconds': confirmation['total_seconds'],
            'confirmation_trial': confirmation['trial_id'], 'submission_provenance_verified': False}


def launch_job(job, job_path, result_path, log_path):
    common.save_json(job_path, job)
    if job.get('plain'):
        command = [sys.executable, str(Path(job['snapshot']['directory']) / 'verify_task1_plain.py'),
                   '--job', str(job_path), '--result', str(result_path)]
    else:
        command = [sys.executable, str(Path(__file__).resolve()), '--worker-job', str(job_path),
                   '--worker-result', str(result_path)]
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT,
                                   cwd=ROOT, start_new_session=True)
        try:
            if process.wait():
                raise RuntimeError(f'Trial failed; inspect {log_path}')
        except BaseException:
            transport.stop_trial(process)
            raise
    return json.loads(result_path.read_text())


def record_outputs(out, rows, selection):
    columns = ['trial_id', 'stage', 'method', 'repeat', 'diagnostic', 'patching', 'timing_scope', 'status',
               'encode_workers', 'total_seconds', 'encode_seconds', 'decode_seconds', 'compare_seconds',
               'token_count', 'ids_match_reference', 'roundtrip_passed', 'parent_peak_rss_bytes',
               'encode_worker_peak_rss_max', 'encode_worker_peak_rss_sum',
               'decode_worker_peak_rss_max', 'decode_worker_peak_rss_sum', 'numpy_loaded_before']
    common.save_json(out / 'results.json', rows)
    with (out / 'results.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# 任务一：批量／并行解码及普通 BPE 接入',
             '只加载旧词表，不训练；历史约11.167秒只作参考。',
             '编码固定并发启动16 workers、256安全分片、每worker1M清空缓存。',
             'R0=C2；N128/N1M=object词表NumPy索引；P4/P8/P16=新启动的bytes-only解码worker。']
    for title, stages, key in [('解码专项（不属于完整成绩）', {'decoder-screening'}, 'decode_seconds'),
                               ('完整接口（包含启动、转换、通信、退出与比较）',
                                {'screening', 'paired-baseline', 'paired-candidate', 'confirmation'}, 'total_seconds'),
                               ('普通 BPE 与实验路径（接入验收）', {'integration-experimental', 'integration-plain'},
                                'total_seconds')]:
        lines += ['', '## ' + title, '', '| 阶段 | 方法 | 次数 | 中位数(s) | 最小(s) | 最大(s) |',
                  '|---|---|---:|---:|---:|---:|']
        groups = {}
        for row in rows:
            if row['stage'] in stages and not row['diagnostic'] and row['status'] == 'ok':
                groups.setdefault((row['stage'], row['method']), []).append(row[key])
        for (stage, method), values in groups.items():
            lines.append(f'| {stage} | {method} | {len(values)} | {statistics.median(values):.3f} | '
                         f'{min(values):.3f} | {max(values):.3f} |')
    lines += ['', '## 选优与接入门槛', '', '```json', json.dumps(selection, ensure_ascii=False, indent=2), '```',
              '', '接入门槛：普通路径中位数≤实验路径103%，且至少4/5对≤对应实验路径105%。',
              '普通路径直接导入独立源码快照的BPE，无后端替换，无新增协调层；首次导入成本照常计时。',
              '活跃W&B独立确认须另行运行，加载词表的探索不是课程训练入口正式成绩。',
              '源码、词表、输入和参考指纹变化时结果无效；不会自动修改dev，不承诺5秒。',
              '跨进程诊断时间有重叠，不能相加；worker峰值RSS之和不代表同时发生的实际峰值。']
    for row in rows:
        if row['diagnostic']:
            lines += ['', '## 独立诊断 ' + row['trial_id'], '', '```json',
                      json.dumps(row.get('transport'), ensure_ascii=False, indent=2), '```']
    (out / 'report.md').write_text('\n'.join(lines) + '\n')


def publish(run, out, rows, selection):
    if run is None:
        return
    import wandb
    columns = ['trial_id', 'stage', 'method', 'repeat', 'diagnostic', 'patching', 'timing_scope', 'status',
               'total_seconds', 'encode_seconds', 'decode_seconds', 'compare_seconds', 'token_count',
               'ids_match_reference', 'roundtrip_passed', 'parent_peak_rss_bytes', 'cpu_affinity']
    run.log({'decode/trials': wandb.Table(columns=columns, data=[[r.get(k) for k in columns] for r in rows])})
    run.summary.update({'completed': bool(selection.get('completed')), 'training': False,
                        'recommended': selection.get('recommended'),
                        'integration_passed': selection.get('integration', {}).get('passed', False),
                        'submission_provenance_verified': False})
    if selection.get('snapshot'):
        artifact = wandb.Artifact(f'plain-candidate-{run.id}', type='source-code')
        for path in Path(selection['snapshot']['directory']).iterdir():
            if path.is_file():
                artifact.add_file(str(path), name=path.name)
        run.log_artifact(artifact)
    for path in out.iterdir():
        if path.is_file() and path.suffix in ('.json', '.csv', '.md'):
            run.save(str(path), base_path=str(out), policy='now')


def choose_cpus(args, workers):
    if hasattr(os, 'sched_getaffinity'):
        allowed = sorted(os.sched_getaffinity(0))
        cpus = sorted(args.cpu_ids) if args.cpu_ids is not None else allowed[:workers]
        if len(cpus) != workers or len(set(cpus)) != workers or not set(cpus) <= set(allowed):
            raise ValueError(f'Requires exactly {workers} distinct allowed CPUs; no downgrade')
        affinity(cpus, include_descendants=True)
        return cpus
    if args.development and args.cpu_ids is None:
        return []
    raise RuntimeError('Formal exploration requires Linux affinity')


def exploration(args):
    transport.validate_fixed_parameters()
    if args.repeats < (1 if args.development else 5) or args.decoder_repeats < (1 if args.development else 3):
        raise ValueError('Need five full/integration repeats and three decoder repeats')
    if args.no_wandb and not args.development:
        raise ValueError('--no-wandb is development-only')
    workers = args.development_workers if args.development else 16
    if not 1 <= workers <= 16:
        raise ValueError('Worker cap must be 1..16')
    cpus = choose_cpus(args, workers)
    out = Path(args.output_dir).resolve()
    if not args.development and not out.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Output must be in personal outputs')
    if out.exists():
        raise FileExistsError(out)
    prefix, valid = str(Path(args.tokenizer_prefix).resolve()), str(Path(args.valid_file).resolve())
    inputs = [valid] + [prefix + s for s in ('.vocab.json', '.merges.json', '.config.json')]
    hashes = {p: common.sha256(p) for p in inputs}
    history = previous_inputs(args.previous_results_dir, hashes)
    for name in ('manifest.json', 'selection.json'):
        path = str((Path(args.previous_results_dir) / name).resolve())
        hashes[path] = common.sha256(path)
    reference = str(Path(args.reference_tokens).resolve()) if args.reference_tokens else str(out / 'reference.u32le')
    if args.reference_tokens:
        size = Path(reference).stat().st_size
        if size % 4 or (not args.development and size != EXPECTED_TOKENS * 4):
            raise ValueError('Reference must be exact full-validation uint32 size')
        hashes[reference] = common.sha256(reference)
    tokenizer = BPE()
    tokenizer.load(prefix)
    validate_tokenizer(tokenizer)
    manifest = {'arguments': vars(args), 'python': sys.version, 'started_utc': datetime.now(UTC).isoformat(),
                'input_sha256': hashes, 'source_sha256': common.source_fingerprints(), 'history': history,
                'cpu_affinity': cpus, 'workers': workers, 'parts': workers * 16, 'cache_capacity': 1048576,
                'training': False, 'timing': 'cold public encode/list/decode/compare; decoder-only separate'}
    out.mkdir(parents=True, exist_ok=False)
    common.save_json(out / 'manifest.json', manifest)
    if args.no_wandb:
        context = nullcontext(None)
    else:
        from wandb_record import experiment
        context = experiment('bpe-task1-decode-exploration', manifest, root=ROOT)
    rows, order, selection = [], [], {}
    expected = None if args.development else EXPECTED_TOKENS
    with context as run:
        def trial(stage, method, repeat=0, diagnostic=False, snapshot=None):
            name = f'trial-{len(order) + 1:03d}'
            record = {'trial_id': name, 'stage': stage, 'method': method, 'repeat': repeat, 'diagnostic': diagnostic}
            order.append(record)
            common.save_json(out / 'schedule.json', order)
            plain = stage == 'integration-plain'
            job = {**record, 'plain': plain, 'tokenizer_prefix': prefix, 'valid_file': valid,
                   'reference': reference, 'expected_tokens': expected, 'cpus': cpus, 'workers': workers,
                   'development': args.development, 'write_reference': stage == 'reference',
                   'case': 'serial' if stage == 'reference' else 'D3', 'snapshot': snapshot,
                   'input_sha256': hashes, 'active_wandb': False}
            print(f'{name}: {stage} {method}', flush=True)
            row = launch_job(job, out / (name + '.job.json'), out / (name + '.json'), out / (name + '.log'))
            if stage != 'reference':
                pipeline.check_row(row)
            row.update(record)
            rows.append(row)
            common.save_json(out / (name + '.json'), row)
            record_outputs(out, rows, selection)
            return row
        try:
            if args.reference_tokens:
                expected = Path(reference).stat().st_size // 4
            else:
                expected = trial('reference', 'R0')['token_count']
            hashes[reference] = common.sha256(reference)
            common.save_json(out / 'reference.json', {'sha256': hashes[reference], 'token_count': expected,
                                                     'provided': bool(args.reference_tokens),
                                                     'retained_for_plain_confirmation': True})
            common.save_json(out / 'manifest.json', manifest)
            selection.update(run_stages(args, trial))
            snapshot = make_snapshot(out / 'plain-source', selection['recommended'], workers, args.development)
            selection['snapshot'] = snapshot
            original, ordinary = [], []
            rng = random.Random(args.seed + 2)
            for repeat in range(1, args.repeats + 1):
                pair = [('integration-experimental', original), ('integration-plain', ordinary)]
                rng.shuffle(pair)
                for stage, target in pair:
                    target.append(trial(stage, selection['recommended'], repeat, snapshot=snapshot))
            selection['integration'] = integration_decision(original, ordinary, args.repeats)
            selection.update(ready_for_plain_confirmation=selection['integration']['passed'],
                             tokenizer_prefix=prefix, valid_file=valid, reference=reference,
                             expected_tokens=expected, input_sha256=hashes, cpu_affinity=cpus,
                             development=args.development)
            verify_snapshot(snapshot)
            changed = [p for p, digest in hashes.items() if common.sha256(p) != digest]
            source_changed = common.source_fingerprints() != manifest['source_sha256']
            common.save_json(out / 'integrity.json', {'changed_inputs': changed, 'source_changed': source_changed})
            if changed or source_changed:
                raise ValueError('Source/input changed; no valid recommendation')
            common.save_json(out / 'selection.json', selection)
            record_outputs(out, rows, selection)
            publish(run, out, rows, selection)
        except BaseException as error:
            common.save_json(out / 'failure.json', {'error': repr(error), 'completed': False})
            if run is not None:
                run.summary.update({'completed': False, 'error': repr(error)})
            raise
    print(f'Completed: {out / "report.md"}', flush=True)


def plain_confirmation(args):
    report = Path(args.selection_report).resolve()
    selection = json.loads(report.read_text())
    if not selection.get('completed') or not selection.get('ready_for_plain_confirmation'):
        raise ValueError('Plain integration gate did not pass; inspect exploration report')
    if bool(selection['development']) != bool(args.development):
        raise ValueError('Development provenance cannot be used as a formal result')
    if args.no_wandb and not args.development:
        raise ValueError('Formal plain confirmation requires an active W&B run')
    snapshot = selection['snapshot']
    verify_snapshot(snapshot)
    cpus = selection['cpu_affinity']
    if args.cpu_ids is not None and sorted(args.cpu_ids) != cpus:
        raise ValueError('Confirmation must use the same CPUs as exploration')
    if cpus:
        if not hasattr(os, 'sched_getaffinity') or not set(cpus) <= os.sched_getaffinity(0):
            raise ValueError('Exploration CPUs unavailable; no resource change allowed')
        affinity(cpus, include_descendants=True)
    elif not args.development:
        raise RuntimeError('Formal confirmation requires Linux affinity')
    hashes = {**selection['input_sha256'], str(report): common.sha256(report)}
    if any(common.sha256(p) != digest for p, digest in hashes.items()):
        raise ValueError('Confirmation inputs changed')
    out = Path(args.output_dir).resolve()
    if not args.development and not out.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Output must be in personal outputs')
    out.mkdir(parents=True, exist_ok=False)
    job = {key: selection[key] for key in ('tokenizer_prefix', 'valid_file', 'reference', 'expected_tokens')}
    job.update(plain=True, snapshot=snapshot, input_sha256=hashes, cpus=cpus,
               development=args.development, active_wandb=not args.no_wandb)
    try:
        result = launch_job(job, out / 'job.json', out / 'result.json', out / 'confirmation.log')
        pipeline.check_row(result)
        verify_snapshot(snapshot)
        if any(common.sha256(p) != digest for p, digest in hashes.items()):
            raise ValueError('Confirmation inputs changed during run')
        ratio = result['total_seconds'] / selection['integration']['plain_median_seconds']
        result.update(context_ratio_to_plain_median=ratio, context_regression_flag=ratio > 1.05,
                      submission_provenance_verified=False)
        common.save_json(out / 'confirmation.json', result)
        (out / 'report.md').write_text(
            '# 普通 BPE 活跃 W&B 独立确认\n\n'
            f"完整耗时 {result['total_seconds']:.3f}s；相对接入中位数 {ratio:.3f}。\n\n"
            f"活跃W&B：{result['active_wandb']}；超过5%需定位上下文差异：{result['context_regression_flag']}。\n\n"
            '仅加载旧词表、不训练、不执行后端替换；单次测量不是稳定性证明，不替代课程入口成绩。\n')
    except BaseException as error:
        common.save_json(out / 'failure.json', {'completed': False, 'error': repr(error)})
        raise
    print(f'Completed: {out / "report.md"}', flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('exploration', 'plain-confirmation'), default='exploration')
    parser.add_argument('--tokenizer-prefix', default='outputs/8b68b68e0cbe/tokenizer')
    parser.add_argument('--valid-file', default='/data/assignment1-data/owt_valid.txt')
    parser.add_argument('--previous-results-dir', default='outputs/task1-pipeline-exploration')
    parser.add_argument('--selection-report', default='outputs/task1-decode-exploration/selection.json')
    parser.add_argument('--reference-tokens')
    parser.add_argument('--decoder-repeats', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261019)
    parser.add_argument('--cpu-ids', nargs='+', type=int)
    parser.add_argument('--diagnostics', action='store_true')
    parser.add_argument('--output-dir')
    parser.add_argument('--development', action='store_true')
    parser.add_argument('--development-workers', type=int, default=2)
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--worker-job', help=argparse.SUPPRESS)
    parser.add_argument('--worker-result', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = 'outputs/task1-decode-exploration' if args.phase == 'exploration' else 'outputs/task1-plain-confirmation'
    return args


def main():
    args = parse_args()
    restrict_environment()
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Decode exploration interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    if args.worker_job:
        worker(json.loads(Path(args.worker_job).read_text()), args.worker_result)
    elif args.phase == 'exploration':
        exploration(args)
    else:
        plain_confirmation(args)


if __name__ == '__main__':
    main()
