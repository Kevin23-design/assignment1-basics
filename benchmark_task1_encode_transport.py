"""T0--T3 cold public-interface exploration using the verified R0 source snapshot."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import random
import shutil
import signal
import statistics
import subprocess
import sys
import time

# Trial interpreters must select the accepted snapshot before any BPE import.
if __name__ == '__main__' and '--worker-job' in sys.argv:
    _job = json.loads(Path(sys.argv[sys.argv.index('--worker-job') + 1]).read_text())
    sys.path.insert(0, _job['baseline_snapshot']['directory'])

import benchmark_task1_decode as previous
from task1_decode_experiment import integration_decision, replace_function, source_hashes, verify_snapshot
import task1_encode_transport_runtime as runtime

common, pipeline, transport = previous.common, previous.pipeline, previous.transport
ROOT = Path(__file__).resolve().parent


def load_history(args):
    selection_path, confirmation_path = Path(args.previous_selection).resolve(), Path(args.plain_confirmation).resolve()
    selection = json.loads(selection_path.read_text())
    confirmation = json.loads(confirmation_path.read_text())
    snapshot = selection['snapshot']
    if (not selection.get('completed') or not selection.get('ready_for_plain_confirmation')
            or not selection.get('integration', {}).get('passed') or selection.get('recommended') != 'R0'
            or snapshot['method'] != 'R0' or snapshot.get('patching') is not False):
        raise ValueError('Previous results must contain an accepted ordinary R0 snapshot')
    if bool(selection['development']) != bool(args.development):
        raise ValueError('Development provenance cannot become a formal result')
    verify_snapshot(snapshot)
    pipeline.check_row(confirmation)
    if (confirmation.get('patching') is not False or confirmation.get('source_changed') is not False
            or confirmation.get('method') != 'R0' or confirmation.get('token_count') != selection['expected_tokens']
            or confirmation.get('cpu_affinity') != selection['cpu_affinity']
            or Path(confirmation['bpe_source']).resolve() != Path(snapshot['directory']) / 'bpe.py'
            or (not args.development and not confirmation.get('active_wandb'))):
        raise ValueError('Plain confirmation does not match the accepted R0 snapshot/resources')
    if not args.development and (snapshot['workers'] != 16 or selection['expected_tokens'] != previous.EXPECTED_TOKENS):
        raise ValueError('Formal exploration requires 16 workers and the full validation token count')
    hashes = dict(selection['input_sha256'])
    # Ensure the reference is fingerprinted even for malformed/incomplete reports.
    required = [selection['valid_file'], selection['reference']] + [selection['tokenizer_prefix'] + suffix
               for suffix in ('.vocab.json', '.merges.json', '.config.json')]
    if any(path not in hashes for path in required):
        raise ValueError('Previous report is missing required input fingerprints')
    if Path(selection['reference']).stat().st_size != selection['expected_tokens'] * 4:
        raise ValueError('Reference must have the exact uint32 token size')
    if any(common.sha256(path) != digest for path, digest in hashes.items()):
        raise ValueError('Previous input/reference fingerprint changed')
    for path in (selection_path, confirmation_path):
        hashes[str(path)] = common.sha256(path)
    cpus = selection['cpu_affinity']
    if args.cpu_ids is not None and sorted(args.cpu_ids) != cpus:
        raise ValueError('Must use the same CPU set as the previous experiment')
    if cpus:
        if len(cpus) != snapshot['workers'] or len(set(cpus)) != len(cpus):
            raise ValueError('CPU set must match the worker count')
        if not hasattr(os, 'sched_getaffinity') or not set(cpus) <= os.sched_getaffinity(0):
            raise ValueError('Previous CPUs unavailable; no silent resource change')
        previous.affinity(cpus, include_descendants=True)
    elif not args.development:
        raise ValueError('Formal exploration requires Linux CPU affinity')
    return selection, hashes


def make_snapshot(directory, baseline, method):
    if method not in runtime.METHODS:
        raise ValueError('Unknown transport snapshot method')
    verify_snapshot(baseline)
    directory = Path(directory).resolve()
    shutil.copytree(baseline['directory'], directory, ignore=shutil.ignore_patterns('__pycache__', 'snapshot.json'))
    shutil.copyfile(ROOT / 'task1_encode_transport_runtime.py', directory / 'task1_encode_transport_runtime.py')
    if method != 'T0':
        path = directory / 'bpe_memory_runtime.py'
        path.write_text(replace_function(path.read_text(), '_launch', f'''def _launch(tokenizer, text, ranges, cpus, workers):
    from task1_encode_transport_runtime import launch
    return launch(tokenizer, text, ranges, cpus, workers, {method!r})
'''))
    snapshot = {**baseline, 'directory': str(directory), 'method': method,
                'source_sha256': source_hashes(directory)}
    common.save_json(directory / 'snapshot.json', snapshot)
    return snapshot


def worker(job, result_path):
    previous.restrict_environment()
    previous.affinity(job['cpus'])
    verify_snapshot(job['baseline_snapshot'])
    import bpe
    if Path(bpe.__file__).resolve() != Path(job['baseline_snapshot']['directory']) / 'bpe.py':
        raise RuntimeError('Trial imported BPE outside the accepted snapshot')
    import wandb  # noqa: F401 -- caller import context, no per-trial run
    tokenizer = bpe.BPE()
    tokenizer.load(job['tokenizer_prefix'])
    previous.validate_tokenizer(tokenizer)
    with open(job['valid_file'], encoding='utf-8', newline='') as handle:
        text = handle.read()
    if tokenizer._cache:
        raise RuntimeError('Expected a fresh empty cache')
    result = {'status': 'failed', 'method': job['method'], 'diagnostic': job['diagnostic'],
              'patching': True, 'timing_scope': 'full-public-interface', 'cpu_affinity': job['cpus'],
              'encode_workers': job['workers'], 'parts': job['workers'] * 16, 'cache_capacity': 1048576,
              'bpe_source': str(Path(bpe.__file__).resolve())}
    stats = {}
    try:
        with runtime.selected_method(job['method'], job['diagnostic']) as (stats, calls):
            began = time.perf_counter()
            ids = tokenizer.encode(text)
            encoded = time.perf_counter()
            decoded = tokenizer.decode(ids)
            decoded_at = time.perf_counter()
            passed = decoded == text
            ended = time.perf_counter()
        if calls != [job['method']] or not isinstance(ids, list) or len(ids) != job['expected_tokens'] or not passed:
            raise ValueError('Wrong execution path, token count, output type or roundtrip')
        common.check_reference(ids, job['reference'])
        verify_snapshot(job['baseline_snapshot'])
        result.update(status='ok', total_seconds=ended - began, encode_seconds=encoded - began,
                      decode_seconds=decoded_at - encoded, compare_seconds=ended - decoded_at,
                      token_count=len(ids), returned_type='list', roundtrip_passed=True, ids_match_reference=True,
                      parent_peak_rss_bytes=previous.rss_bytes(), first_200=common.first_tokens(tokenizer, ids, text))
        metadata = stats.get('worker_metadata', [])
        if metadata:
            result.update(encode_worker_peak_rss_max=max(w['peak_rss_bytes'] for w in metadata),
                          encode_worker_peak_rss_sum=sum(w['peak_rss_bytes'] for w in metadata))
    except transport.Unavailable as error:
        result.update(status='unavailable', error=str(error))
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        result['transport'] = stats
        common.save_json(result_path, result)


def launch_job(job, job_path, result_path, log_path):
    if job.get('plain'):
        return previous.launch_job(job, job_path, result_path, log_path)
    common.save_json(job_path, job)
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


def run_stages(args, trial):
    rng, rows, unavailable = random.Random(args.seed), [], {}
    def measure(stage, method, repeat=0, diagnostic=False):
        row = trial(stage, method, repeat, diagnostic)
        if row['status'] == 'unavailable':
            if method not in ('T2', 'T3'):
                raise RuntimeError('Baseline/pipe-only candidate cannot be unavailable')
            unavailable[method] = row['error']
        else:
            pipeline.check_row(row)
        return row
    if args.diagnostics:
        measure('diagnostic-baseline', 'T0', diagnostic=True)
    for repeat in range(1, args.repeats + 1):
        order = list(runtime.METHODS)
        rng.shuffle(order)
        for method in order:
            if method not in unavailable:
                rows.append(measure('screening', method, repeat))
    times = {}
    for method in runtime.METHODS:
        if method not in unavailable:
            values = [r['total_seconds'] for r in rows if r['method'] == method]
            if len(values) != args.repeats:
                raise ValueError('Incomplete screening')
            times[method] = statistics.median(values)
    candidate = min(times, key=lambda m: (times[m], m != 'T0', m))
    decision = {'promote_candidate': False, 'reason': 'candidate_matches_baseline'}
    if args.diagnostics and candidate != 'T0':
        measure('diagnostic-candidate', candidate, diagnostic=True)
    if candidate != 'T0' and candidate not in unavailable:
        before, after = [], []
        for repeat in range(1, args.repeats + 1):
            pair = [('paired-baseline', 'T0', before), ('paired-candidate', candidate, after)]
            rng.shuffle(pair)
            for stage, method, target in pair:
                target.append(measure(stage, method, repeat))
            if candidate in unavailable:
                break
        if candidate not in unavailable:
            decision = common.paired_decision(before, after, args.repeats)
    if candidate in unavailable:
        decision = {'promote_candidate': False, 'reason': 'candidate_became_unavailable'}
    recommended = candidate if decision['promote_candidate'] else 'T0'
    confirmation = measure('confirmation', recommended)
    pipeline.check_row(confirmation)
    return {'completed': True, 'recommended': recommended, 'candidate': candidate,
            'screening_medians_seconds': times, 'unavailable': unavailable, 'paired_confirmation': decision,
            'confirmation_seconds': confirmation['total_seconds'], 'confirmation_trial': confirmation['trial_id'],
            'submission_provenance_verified': False}


def record_outputs(out, rows, selection):
    common.save_json(out / 'results.json', rows)
    columns = ['trial_id', 'stage', 'method', 'repeat', 'diagnostic', 'patching', 'timing_scope', 'status',
               'encode_workers', 'parts', 'cache_capacity', 'total_seconds', 'encode_seconds', 'decode_seconds',
               'compare_seconds', 'token_count', 'ids_match_reference', 'roundtrip_passed', 'cpu_affinity',
               'parent_peak_rss_bytes', 'encode_worker_peak_rss_max', 'encode_worker_peak_rss_sum', 'error']
    with (out / 'results.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# 任务一：编码传输探索', '', 'T0=验收通过的R0；T1=状态只序列化一次；'
             'T2=T1+共享输出；T3=T2+共享输入。解码始终使用C2。', '',
             '| 阶段 | 方法 | 次数 | 完整耗时中位数(s) | 最小(s) | 最大(s) |',
             '|---|---|---:|---:|---:|---:|']
    groups = {}
    for row in rows:
        if row['status'] == 'ok' and not row['diagnostic']:
            groups.setdefault((row['stage'], row['method']), []).append(row['total_seconds'])
    for (stage, method), values in groups.items():
        lines.append(f'| {stage} | {method} | {len(values)} | {statistics.median(values):.3f} | '
                     f'{min(values):.3f} | {max(values):.3f} |')
    lines += ['', '## 选优与接入', '', '```json', json.dumps(selection, ensure_ascii=False, indent=2), '```', '',
              '诊断不参与排名；跨进程阶段存在重叠，不能相加解释总耗时。',
              'worker峰值RSS之和并非同时发生的实际峰值。共享内存不可用时不回退冒充测速。',
              '不训练、不自动更新dev；普通接口加载词表实验不替代课程入口成绩。']
    for row in rows:
        if row['diagnostic']:
            lines += ['', '## 诊断 ' + row['trial_id'], '', '```json',
                      json.dumps(row['transport'], ensure_ascii=False, indent=2), '```']
    (out / 'report.md').write_text('\n'.join(lines) + '\n')


def publish(run, out, rows, selection):
    if run is None:
        return
    import wandb
    columns = ['trial_id', 'stage', 'method', 'repeat', 'diagnostic', 'status', 'total_seconds',
               'encode_seconds', 'decode_seconds', 'compare_seconds', 'token_count', 'ids_match_reference',
               'roundtrip_passed', 'cpu_affinity', 'parent_peak_rss_bytes', 'encode_worker_peak_rss_max',
               'encode_worker_peak_rss_sum', 'error']
    run.log({'encode_transport/trials': wandb.Table(columns=columns, data=[[r.get(k) for k in columns] for r in rows])})
    run.summary.update({'completed': True, 'training': False, 'recommended': selection['recommended'],
                        'integration_passed': selection['integration']['passed'], 'submission_provenance_verified': False})
    baseline = json.loads((out / 'manifest.json').read_text())['baseline_snapshot']
    for label, snapshot in [('accepted-baseline', baseline), ('plain-candidate', selection['snapshot'])]:
        artifact = wandb.Artifact(f'{label}-{run.id}', type='source-code')
        for path in Path(snapshot['directory']).iterdir():
            if path.is_file():
                artifact.add_file(str(path), name=path.name)
        run.log_artifact(artifact)
    for path in out.iterdir():
        if path.is_file() and path.suffix in ('.json', '.csv', '.md'):
            run.save(str(path), base_path=str(out), policy='now')


def exploration(args):
    if args.repeats < (1 if args.development else 5):
        raise ValueError('Formal exploration needs at least five repeats')
    if args.no_wandb and not args.development:
        raise ValueError('--no-wandb is development-only')
    history, hashes = load_history(args)
    out = Path(args.output_dir).resolve()
    if not args.development and not out.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Output must be in personal outputs')
    if out.exists():
        raise FileExistsError(out)
    manifest = {'arguments': vars(args), 'python': sys.version, 'started_utc': datetime.now(UTC).isoformat(),
                'input_sha256': hashes, 'source_sha256': common.source_fingerprints(),
                'baseline_snapshot': history['snapshot'], 'workers': history['snapshot']['workers'],
                'cpu_affinity': history['cpu_affinity'], 'training': False, 'decode': 'R0/C2',
                'parts': history['snapshot']['workers'] * 16, 'cache_capacity': 1048576,
                'decode_block_size': 131072, 'timing': 'cold public encode/list/decode/compare including cleanup'}
    out.mkdir(parents=True, exist_ok=False)
    common.save_json(out / 'manifest.json', manifest)
    if args.no_wandb:
        context = nullcontext(None)
    else:
        from wandb_record import experiment
        context = experiment('bpe-task1-encode-transport-exploration', manifest, root=ROOT)
    rows, order, selection = [], [], {}
    with context as run:
        def trial(stage, method, repeat=0, diagnostic=False, snapshot=None):
            name = f'trial-{len(order) + 1:03d}'
            record = {'trial_id': name, 'stage': stage, 'method': method, 'repeat': repeat, 'diagnostic': diagnostic}
            order.append(record)
            common.save_json(out / 'schedule.json', order)
            job = {**record, **{k: history[k] for k in ('tokenizer_prefix', 'valid_file', 'reference', 'expected_tokens')},
                   'baseline_snapshot': history['snapshot'], 'workers': history['snapshot']['workers'],
                   'cpus': history['cpu_affinity'], 'development': args.development, 'input_sha256': hashes,
                   'plain': stage == 'integration-plain', 'snapshot': snapshot, 'active_wandb': False}
            print(f'{name}: {stage} {method}', flush=True)
            row = launch_job(job, out / (name + '.job.json'), out / (name + '.json'), out / (name + '.log'))
            if row['status'] != 'unavailable':
                pipeline.check_row(row)
            row.update(record)
            rows.append(row)
            common.save_json(out / (name + '.json'), row)
            record_outputs(out, rows, selection)
            return row
        try:
            selection.update(run_stages(args, trial))
            snapshot = make_snapshot(out / 'plain-source', history['snapshot'], selection['recommended'])
            selection['snapshot'] = snapshot
            original, ordinary = [], []
            rng = random.Random(args.seed + 2)
            for repeat in range(1, args.repeats + 1):
                pair = [('integration-experimental', original), ('integration-plain', ordinary)]
                rng.shuffle(pair)
                for stage, target in pair:
                    row = trial(stage, selection['recommended'], repeat, snapshot=snapshot)
                    pipeline.check_row(row)
                    target.append(row)
            selection['integration'] = integration_decision(original, ordinary, args.repeats)
            selection.update(ready_for_plain_confirmation=selection['integration']['passed'],
                             **{k: history[k] for k in ('tokenizer_prefix', 'valid_file', 'reference', 'expected_tokens',
                                                       'cpu_affinity', 'development')}, input_sha256=hashes)
            verify_snapshot(snapshot)
            verify_snapshot(history['snapshot'])
            changed = [p for p, digest in hashes.items() if common.sha256(p) != digest]
            source_changed = common.source_fingerprints() != manifest['source_sha256']
            common.save_json(out / 'integrity.json', {'changed_inputs': changed, 'source_changed': source_changed})
            if changed or source_changed:
                raise ValueError('Input/source changed; results invalid')
            common.save_json(out / 'selection.json', selection)
            record_outputs(out, rows, selection)
            # Same complete candidate table and source artifact, under this suite's run.
            publish(run, out, rows, selection)
        except BaseException as error:
            selection.update(completed=False, ready_for_plain_confirmation=False, error=repr(error))
            common.save_json(out / 'selection.json', selection)
            record_outputs(out, rows, selection)
            common.save_json(out / 'failure.json', {'completed': False, 'error': repr(error)})
            if run is not None:
                run.summary.update({'completed': False, 'error': repr(error)})
            raise
    print(f'Completed: {out / "report.md"}', flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('exploration', 'plain-confirmation'), default='exploration')
    parser.add_argument('--previous-selection', default='outputs/task1-decode-exploration/selection.json')
    parser.add_argument('--plain-confirmation', default='outputs/task1-plain-confirmation/confirmation.json')
    parser.add_argument('--selection-report', default='outputs/task1-encode-transport/selection.json')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261019)
    parser.add_argument('--diagnostics', action='store_true')
    parser.add_argument('--cpu-ids', type=int, nargs='+')
    parser.add_argument('--output-dir')
    parser.add_argument('--development', action='store_true')
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--worker-job', help=argparse.SUPPRESS)
    parser.add_argument('--worker-result', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = ('outputs/task1-encode-transport' if args.phase == 'exploration'
                           else 'outputs/task1-encode-transport-plain-confirmation')
    return args


def main():
    args = parse_args()
    previous.restrict_environment()
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Encoding transport exploration interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    if args.worker_job:
        worker(json.loads(Path(args.worker_job).read_text()), args.worker_result)
    elif args.phase == 'plain-confirmation':
        previous.plain_confirmation(args)
    else:
        exploration(args)


if __name__ == '__main__':
    main()
