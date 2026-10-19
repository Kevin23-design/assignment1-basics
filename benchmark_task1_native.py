"""Staged native decode/encode exploration; no training and no runtime compilation."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
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

# Trial interpreters must select the accepted snapshot before any BPE import.
if __name__ == '__main__' and '--worker-job' in sys.argv:
    _job = json.loads(Path(sys.argv[sys.argv.index('--worker-job') + 1]).read_text())
    sys.path.insert(0, _job['snapshot']['directory'])

import benchmark_task1_decode as previous
from task1_decode_experiment import integration_decision, verify_snapshot
import benchmark_task1_encode_transport as accepted
from task1_native_experiment import METHODS, make_snapshot
from task1_native_runtime import verify_build

common, pipeline, transport = previous.common, previous.pipeline, previous.transport
ROOT = Path(__file__).resolve().parent


def load_history(args):
    return accepted.load_history(args)




@contextmanager
def diagnostic_launch(enabled, stats):
    import bpe_memory_runtime as baseline
    original, split = baseline._launch, baseline._partition_ranges
    if enabled:
        from task1_pipeline_runtime import launch_all_first
        def launch(tokenizer, text, ranges, cpus, workers):
            result = launch_all_first(tokenizer, text, ranges, cpus, workers, True, stats)
            tokenizer._task1_worker_metrics = stats.get('worker_metadata', [])
            return result
        def partition(*args):
            began = time.perf_counter()
            result = split(*args)
            stats['split_seconds'] = time.perf_counter() - began
            return result
        baseline._launch, baseline._partition_ranges = launch, partition
    try:
        yield
    finally:
        baseline._launch, baseline._partition_ranges = original, split


def worker(job, result_path):
    previous.restrict_environment()
    previous.affinity(job['cpus'])
    verify_snapshot(job['snapshot'])
    import bpe
    if Path(bpe.__file__).resolve() != Path(job['snapshot']['directory']) / 'bpe.py':
        raise RuntimeError('Trial imported BPE outside its immutable candidate snapshot')
    import wandb  # noqa: F401 -- no per-trial run
    tokenizer = bpe.BPE()
    tokenizer.load(job['tokenizer_prefix'])
    previous.validate_tokenizer(tokenizer)
    with open(job['valid_file'], encoding='utf-8', newline='') as handle:
        text = handle.read()
    if 'preflight_parts' in job:
        from bpe_memory_runtime import _partition_ranges
        availability = {str(parts): _partition_ranges(tokenizer, text, parts) is not None
                        for parts in job['preflight_parts']}
        common.save_json(result_path, {'availability': availability})
        return
    if tokenizer._cache:
        raise RuntimeError('Expected an empty cache')
    result = {'status': 'failed', 'method': job['method'], 'diagnostic': job['diagnostic'],
              'patching': bool(job['diagnostic']), 'timing_scope': 'full-public-interface',
              'cpu_affinity': job['cpus'], 'encode_workers': job['workers'],
              'parts': job['snapshot'].get('parts', job['workers'] * 16),
              'cache_capacity': 1048576, 'bpe_source': str(Path(bpe.__file__).resolve())}
    stats = {}
    try:
        with diagnostic_launch(job['diagnostic'], stats):
            began = time.perf_counter()
            ids = tokenizer.encode(text)
            encoded = time.perf_counter()
            decoded = tokenizer.decode(ids)
            decoded_at = time.perf_counter()
            passed = decoded == text
            ended = time.perf_counter()
        if not isinstance(ids, list) or len(ids) != job['expected_tokens'] or not passed:
            raise ValueError('Wrong token count, output type or roundtrip')
        metadata = getattr(tokenizer, '_task1_worker_metrics', [])
        if not metadata:
            raise ValueError('Encoder bypassed the required direct-worker path')
        if job['snapshot'].get('decode_access_case') and sum(w['tasks'] for w in metadata) != result['parts']:
            raise ValueError('Encoder used a different number of partitions')
        native_metrics = getattr(tokenizer, '_native_decode_metrics', None)
        if job['method'] != 'B0' and (not native_metrics or native_metrics.get('backend') != 'native'):
            raise ValueError('Candidate silently bypassed native decoding')
        if job['method'] == 'CY-ED' and any(not w.get('native_encoding') or
                w['native_encoding']['fallback_calls'] or not w['native_encoding']['native_calls'] for w in metadata):
            raise ValueError('Candidate silently bypassed native encoding')
        if job['snapshot'].get('assembly_build'):
            assembly = getattr(tokenizer, '_task1_assembly_metrics', None)
            if (not assembly or assembly.get('method') != job['snapshot']['assembly_method']
                    or assembly.get('converted_ids') != len(ids)):
                raise ValueError('Candidate bypassed its assembly backend')
            result['assembly'] = assembly
            if assembly['method'] != 'M0':
                expected = Path(job['snapshot']['directory']) / job['snapshot']['assembly_build']['binary']
                if Path(assembly['extension_path']).resolve() != expected.resolve():
                    raise ValueError('Assembly extension loaded from another snapshot')
            if not job['diagnostic']:
                build = job['snapshot'].get('decode_access_build', job['snapshot']['native_build'])
                expected = Path(job['snapshot']['directory']) / build['binary']
                if Path(native_metrics['extension_path']).resolve() != expected.resolve():
                    raise ValueError('Normal trial replaced the original native decoder')
        if job['snapshot'].get('decode_access_build'):
            from task1_decode_access_runtime import check_result
            check_result(job['snapshot'], {'native_decode': native_metrics, 'token_count': len(ids)})
        common.check_reference(ids, job['reference'])
        verify_snapshot(job['snapshot'])
        result.update(status='ok', total_seconds=ended - began, encode_seconds=encoded - began,
                      decode_seconds=decoded_at - encoded, compare_seconds=ended - decoded_at,
                      token_count=len(ids), returned_type='list', roundtrip_passed=True, ids_match_reference=True,
                      parent_peak_rss_bytes=previous.rss_bytes(), native_decode=native_metrics,
                      worker_metadata=metadata, encode_worker_peak_rss_max=max(w['peak_rss_bytes'] for w in metadata),
                      encode_worker_peak_rss_sum=sum(w['peak_rss_bytes'] for w in metadata),
                      first_200=common.first_tokens(tokenizer, ids, text))
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
    rng, decisions = random.Random(args.seed), {}
    if args.diagnostics:
        trial('diagnostic-baseline', 'B0', diagnostic=True)
    def comparison(baseline, candidate):
        rows = {baseline: [], candidate: []}
        label = 'decode' if candidate == 'CY-D' else 'encode'
        for repeat in range(1, args.repeats + 1):
            order = [baseline, candidate]
            rng.shuffle(order)
            for method in order:
                row = trial(label + '-screening', method, repeat)
                pipeline.check_row(row)
                rows[method].append(row)
        if args.diagnostics:
            trial('diagnostic-' + label, candidate, diagnostic=True)
        before, after = [], []
        for repeat in range(1, args.repeats + 1):
            pair = [('paired-' + label + '-baseline', baseline, before),
                    ('paired-' + label + '-candidate', candidate, after)]
            rng.shuffle(pair)
            for stage, method, target in pair:
                target.append(trial(stage, method, repeat))
        decision = common.paired_decision(before, after, args.repeats)
        decision['screening_medians_seconds'] = {k: statistics.median(r['total_seconds'] for r in v)
                                                for k, v in rows.items()}
        return decision
    decisions['decode'] = comparison('B0', 'CY-D')
    recommended = 'B0'
    if decisions['decode']['promote_candidate']:
        recommended = 'CY-D'
        decisions['encode'] = comparison('CY-D', 'CY-ED')
        if decisions['encode']['promote_candidate']:
            recommended = 'CY-ED'
    else:
        decisions['encode'] = {'skipped': True, 'reason': 'native_decode_did_not_pass_gate'}
    confirmation = trial('confirmation', recommended)
    pipeline.check_row(confirmation)
    return {'completed': True, 'recommended': recommended, 'decisions': decisions,
            'confirmation_seconds': confirmation['total_seconds'], 'confirmation_trial': confirmation['trial_id'],
            'submission_provenance_verified': False}


def record_outputs(out, rows, selection):
    common.save_json(out / 'results.json', rows)
    columns = ['trial_id', 'stage', 'method', 'repeat', 'diagnostic', 'patching', 'timing_scope', 'status',
               'encode_workers', 'parts', 'cache_capacity', 'total_seconds', 'encode_seconds', 'decode_seconds',
               'compare_seconds', 'token_count', 'ids_match_reference', 'roundtrip_passed', 'cpu_affinity',
               'parent_peak_rss_bytes', 'encode_worker_peak_rss_max', 'encode_worker_peak_rss_sum', 'native_decode', 'error']
    with (out / 'results.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# 任务一：Cython 解码与编码热点探索', '', 'B0=验收通过的R0；CY-D=原生批量解码；CY-ED=原生未命中核心+原生解码。', '',
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
              'worker峰值RSS之和并非同时发生的实际峰值。原生候选不得静默回退冒充测速。',
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
               'encode_worker_peak_rss_sum', 'native_decode', 'error']
    run.log({'native/trials': wandb.Table(columns=columns, data=[[r.get(k) for k in columns] for r in rows])})
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


@contextmanager
def recording_context(context, out, selection):
    yielded = False
    try:
        with context as run:
            yielded = True
            yield run
    except Exception as error:
        if not yielded or not selection.get('completed'):
            raise
        # A finish/upload error after verified computation must not relabel IDs as wrong.
        selection.update(upload_status='failed', upload_error=repr(error))
        common.save_json(out / 'upload_failure.json', {'error': repr(error), 'measurements_valid': True})
        common.save_json(out / 'selection.json', selection)


def exploration(args):
    if args.repeats < (1 if args.development else 5):
        raise ValueError('Formal exploration needs at least five repeats')
    if args.no_wandb and not args.development:
        raise ValueError('--no-wandb is development-only')
    history, hashes = load_history(args)
    build = verify_build(args.build_dir)
    if build['pyx_sha256'] != common.sha256(ROOT / 'task1_native.pyx'):
        raise ValueError('Native source changed since compilation; rebuild before benchmarking')
    build_path = Path(args.build_dir).resolve()
    for name in build['files']:
        hashes[str(build_path / name)] = common.sha256(build_path / name)
    hashes[str(build_path / 'build.json')] = common.sha256(build_path / 'build.json')
    out = Path(args.output_dir).resolve()
    if not args.development and not out.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Output must be in personal outputs')
    if out.exists():
        raise FileExistsError(out)
    manifest = {'arguments': vars(args), 'python': sys.version, 'started_utc': datetime.now(UTC).isoformat(),
                'input_sha256': hashes, 'source_sha256': common.source_fingerprints(),
                'native_build': build, 'baseline_snapshot': history['snapshot'], 'workers': history['snapshot']['workers'],
                'cpu_affinity': history['cpu_affinity'], 'training': False, 'decode': 'C2 or two-pass native',
                'parts': history['snapshot']['workers'] * 16, 'cache_capacity': 1048576,
                'decode_block_size': 131072, 'timing': 'cold public encode/list/decode/compare including cleanup'}
    out.mkdir(parents=True, exist_ok=False)
    common.save_json(out / 'manifest.json', manifest)
    snapshots = {method: make_snapshot(out / ('candidate-' + method), history['snapshot'], method, args.build_dir)
                 for method in METHODS}
    if args.no_wandb:
        context = nullcontext(None)
    else:
        from wandb_record import experiment
        context = experiment('bpe-task1-native-exploration', manifest, root=ROOT)
    rows, order, selection = [], [], {}
    with recording_context(context, out, selection) as run:
        def trial(stage, method, repeat=0, diagnostic=False, snapshot=None):
            name = f'trial-{len(order) + 1:03d}'
            record = {'trial_id': name, 'stage': stage, 'method': method, 'repeat': repeat, 'diagnostic': diagnostic}
            order.append(record)
            common.save_json(out / 'schedule.json', order)
            job = {**record, **{k: history[k] for k in ('tokenizer_prefix', 'valid_file', 'reference', 'expected_tokens')},
                   'native_build': build, 'baseline_snapshot': history['snapshot'], 'workers': history['snapshot']['workers'],
                   'cpus': history['cpu_affinity'], 'development': args.development, 'input_sha256': hashes,
                   'plain': stage == 'integration-plain', 'snapshot': snapshot or snapshots[method], 'active_wandb': False}
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
            snapshot = snapshots[selection['recommended']]
            selection['snapshot'] = snapshot
            selection['candidate_snapshots'] = snapshots
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
            selection.update(measurements_valid=True, ready_for_plain_confirmation=selection['integration']['passed'],
                             **{k: history[k] for k in ('tokenizer_prefix', 'valid_file', 'reference', 'expected_tokens',
                                                       'cpu_affinity', 'development')}, input_sha256=hashes)
            verify_snapshot(snapshot)
            verify_snapshot(history['snapshot'])
            for candidate_snapshot in snapshots.values():
                verify_snapshot(candidate_snapshot)
            changed = [p for p, digest in hashes.items() if common.sha256(p) != digest]
            source_changed = common.source_fingerprints() != manifest['source_sha256']
            common.save_json(out / 'integrity.json', {'changed_inputs': changed, 'source_changed': source_changed})
            if changed or source_changed:
                raise ValueError('Input/source changed; results invalid')
            common.save_json(out / 'selection.json', selection)
            record_outputs(out, rows, selection)
            # Same complete candidate table and source artifact, under this suite's run.
            try:
                publish(run, out, rows, selection)
            except Exception as upload_error:
                selection['upload_status'] = 'failed'
                selection['upload_error'] = repr(upload_error)
                common.save_json(out / 'upload_failure.json', {'error': repr(upload_error), 'measurements_valid': True})
            else:
                selection['upload_status'] = 'disabled' if run is None else 'submitted'
            common.save_json(out / 'selection.json', selection)
            record_outputs(out, rows, selection)
        except BaseException as error:
            selection.update(completed=False, measurements_valid=False, ready_for_plain_confirmation=False, error=repr(error))
            common.save_json(out / 'selection.json', selection)
            record_outputs(out, rows, selection)
            common.save_json(out / 'failure.json', {'completed': False, 'error': repr(error)})
            if run is not None:
                run.summary.update({'completed': False, 'error': repr(error)})
            raise
    record_outputs(out, rows, selection)
    print(f'Completed: {out / "report.md"}', flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('exploration', 'plain-confirmation'), default='exploration')
    parser.add_argument('--previous-selection', default='outputs/task1-decode-exploration/selection.json')
    parser.add_argument('--plain-confirmation', default='outputs/task1-plain-confirmation/confirmation.json')
    parser.add_argument('--selection-report', default='outputs/task1-native/selection.json')
    parser.add_argument('--build-dir', default='outputs/task1-native-build')
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
        args.output_dir = ('outputs/task1-native' if args.phase == 'exploration'
                           else 'outputs/task1-native-plain-confirmation')
    return args


def main():
    args = parse_args()
    previous.restrict_environment()
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Native BPE exploration interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    if args.worker_job:
        worker(json.loads(Path(args.worker_job).read_text()), args.worker_result)
    elif args.phase == 'plain-confirmation':
        report = json.loads(Path(args.selection_report).read_text())
        verify_build(report['snapshot']['directory'])
        previous.plain_confirmation(args)
    else:
        exploration(args)


if __name__ == '__main__':
    main()
