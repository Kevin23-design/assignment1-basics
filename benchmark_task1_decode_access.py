"""Borrowed-reference decoding, then safe-partition sweep; never trains or runtime-builds."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import random
import signal
import statistics
import sys

import benchmark_task1_assembly as assembly
from task1_assembly_runtime import verify_build as verify_assembly
from task1_decode_access_experiment import CASES, PARTS, make_snapshot
from task1_decode_access_runtime import verify_build as verify_access
from task1_decode_experiment import integration_decision, verify_snapshot
from task1_native_adapter_experiment import assert_identical_native
from task1_native_runtime import verify_build

native, common, pipeline = assembly.native, assembly.common, assembly.pipeline
ROOT = Path(__file__).resolve().parent
system_load = assembly.system_load


def load_history(args):
    report, confirmation_path = Path(args.previous_selection).resolve(), Path(args.plain_confirmation).resolve()
    history, confirmation = json.loads(report.read_text()), json.loads(confirmation_path.read_text())
    if (not history.get('completed') or not history.get('measurements_valid')
            or not history.get('ready_for_plain_confirmation') or not history.get('integration', {}).get('passed')
            or history.get('recommended') != 'M2-A1'):
        raise ValueError('Requires completed, valid, accepted M2-A1 assembly results')
    if bool(history['development']) != bool(args.development):
        raise ValueError('Development provenance cannot be used as a formal result')
    origin = history['snapshot']
    if (origin != history['candidate_snapshots']['M2-A1'] or origin['method'] != 'CY-ED'
            or origin.get('assembly_case') != 'M2-A1' or origin.get('assembly_method') != 'M2'
            or origin.get('adapter_case') != 'A1' or origin.get('assembly_diagnostic')
            or origin.get('diagnostic_instrumentation') or origin.get('decode_access_case')
            or origin['development'] != history['development']):
        raise ValueError('Historical source is not the normal accepted M2-A1 snapshot')
    verify_snapshot(origin)
    verify_build(origin['directory'])
    verify_assembly(origin['directory'])
    pipeline.check_row(confirmation)
    if (confirmation.get('patching') is not False or confirmation.get('source_changed') is not False
            or confirmation.get('method') != 'CY-ED' or confirmation.get('assembly_case') != 'M2-A1'
            or confirmation.get('token_count') != history['expected_tokens']
            or confirmation.get('cpu_affinity') != history['cpu_affinity']
            or Path(confirmation['bpe_source']).resolve() != Path(origin['directory']) / 'bpe.py'
            or confirmation.get('native_decode', {}).get('backend') != 'native'
            or confirmation.get('assembly', {}).get('method') != 'M2'
            or confirmation['assembly'].get('converted_ids') != history['expected_tokens']
            or (not args.development and not confirmation.get('active_wandb'))):
        raise ValueError('Plain confirmation does not match accepted M2-A1')
    for metric, build in [('native_decode', 'native_build'), ('assembly', 'assembly_build')]:
        expected = Path(origin['directory']) / origin[build]['binary']
        if Path(confirmation[metric]['extension_path']).resolve() != expected.resolve():
            raise ValueError('Plain confirmation used another extension binary')
    workers = origin['workers']
    if not 1 <= workers <= 16 or (not args.development and
            (workers != 16 or history['expected_tokens'] != native.previous.EXPECTED_TOKENS)):
        raise ValueError('Formal experiment requires 16 workers and the complete validation count')
    hashes = dict(history['input_sha256'])
    required = [history['valid_file'], history['reference']] + [history['tokenizer_prefix'] + suffix
               for suffix in ('.vocab.json', '.merges.json', '.config.json')]
    if any(path not in hashes for path in required):
        raise ValueError('Missing input/tokenizer/reference fingerprints')
    if Path(history['reference']).stat().st_size != history['expected_tokens'] * 4:
        raise ValueError('Reference uint32 size differs')
    if any(common.sha256(path) != digest for path, digest in hashes.items()):
        raise ValueError('Historical input fingerprint changed')
    for path in (report, confirmation_path):
        hashes[str(path)] = common.sha256(path)
    cpus = history['cpu_affinity']
    if args.cpu_ids is not None and sorted(args.cpu_ids) != cpus:
        raise ValueError('Must inherit the exact same CPU set')
    if cpus:
        if len(cpus) != workers or len(set(cpus)) != workers or not hasattr(os, 'sched_getaffinity') \
                or not set(cpus) <= os.sched_getaffinity(0):
            raise ValueError('Historical CPU allocation unavailable; no downgrade')
        native.previous.affinity(cpus, include_descendants=True)
    elif not args.development:
        raise ValueError('Formal experiment requires Linux affinity')
    context = {key: confirmation.get(key) for key in
               ('total_seconds', 'context_ratio_to_plain_median', 'context_regression_flag')}
    return history, origin, hashes, assert_identical_native([origin]), context


def case_key(decoder, parts):
    return decoder if parts == 256 else f'{decoder}-P{parts}'


def run_stages(args, trial, availability):
    rng = random.Random(args.seed)
    def screen(cases, stage):
        rows = {case: [] for case in cases}
        for repeat in range(1, args.repeats + 1):
            order = list(cases)
            rng.shuffle(order)
            for case in order:
                row = trial(stage, case, repeat)
                pipeline.check_row(row)
                rows[case].append(row)
        return {case: statistics.median(r['total_seconds'] for r in values) for case, values in rows.items()}
    def paired(baseline, candidate, stage):
        before, after = [], []
        for repeat in range(1, args.repeats + 1):
            pair = [(stage + '-baseline', baseline, before), (stage + '-candidate', candidate, after)]
            rng.shuffle(pair)
            for label, case, target in pair:
                target.append(trial(label, case, repeat))
        return common.paired_decision(before, after, args.repeats)
    if not availability.get('256'):
        raise ValueError('Baseline cannot produce 256 nonempty safe partitions')
    if args.diagnostics:
        for case in CASES:
            pipeline.check_row(trial('diagnostic-decode', case, diagnostic=True))
    medians = screen(CASES, 'decode-screening')
    candidate = min(medians, key=lambda case: (medians[case], case != 'N0', case))
    selected = 'N0'
    decision = {'promote_candidate': False, 'reason': 'baseline_screened_fastest'}
    if candidate != 'N0':
        decision = paired('N0', candidate, 'decode-paired')
        if decision['promote_candidate']:
            selected = candidate
    mechanism = {'skipped': True, 'reason': 'N2_not_promoted'}
    if selected == 'N2':
        mechanism = paired('N1', 'N2', 'index-buffer-mechanism')
        mechanism['ranking_effect'] = 'none; explains buffer contribution only'
    cases = [case_key(selected, parts) for parts in PARTS if availability.get(str(parts))]
    partition_times = screen(cases, 'partition-screening')
    best = min(partition_times, key=lambda case: (partition_times[case], case != selected, case))
    partition_decision = {'promote_candidate': False, 'reason': 'baseline_screened_fastest'}
    recommended = selected
    if best != selected:
        partition_decision = paired(selected, best, 'partition-paired')
        if partition_decision['promote_candidate']:
            recommended = best
    if args.diagnostics:
        for case in cases:
            if case != selected:
                pipeline.check_row(trial('diagnostic-partition', case, diagnostic=True))
    confirmation = trial('confirmation', recommended)
    pipeline.check_row(confirmation)
    return {'completed': True, 'recommended': recommended, 'selected_decoder': selected,
            'decode_screening_medians_seconds': medians, 'decode_decision': decision,
            'index_buffer_mechanism': mechanism, 'partition_screening_medians_seconds': partition_times,
            'partition_decision': partition_decision,
            'unavailable_partitions': {key: 'Cannot produce requested nonempty safe partitions'
                                       for key, available in availability.items() if not available},
            'confirmation_seconds': confirmation['total_seconds'], 'confirmation_trial': confirmation['trial_id'],
            'submission_provenance_verified': False}


def record_outputs(out, rows, selection):
    assembly.record_outputs(out, rows, selection)
    path = out / 'report.md'
    text = path.read_text().replace('任务一：原生列表组装与整数复用探索', '任务一：解码对象访问与安全分片探索')
    text = text.replace('M0=当前array扩展；M1=原生整数转换；M2=原生整数池。-A1表示固定组装后组合修复的原生编码。',
                        'N0=原解码；N1=借用引用两遍读取；N2=借用引用与调用内uint32索引。P后数字为分片数，默认256。')
    text = text.replace('正常解码保留原二进制；诊断分阶段解码是独立实现，不进入排名。',
                        '编码和整数池二进制不变；诊断不进入排名。N2检查/累计时间包含索引写入，不单独估算其成本。')
    path.write_text(text)


def publish(run, out, rows, selection):
    if run is None:
        return
    import wandb
    columns = ['trial_id', 'stage', 'method', 'parts', 'diagnostic', 'status', 'total_seconds', 'encode_seconds',
               'decode_seconds', 'compare_seconds', 'token_count', 'ids_match_reference', 'roundtrip_passed',
               'cpu_affinity', 'parent_peak_rss_bytes', 'native_decode', 'system_load_before', 'system_load_after']
    run.log({'decode_access/trials': wandb.Table(columns=columns, data=[[r.get(k) for k in columns] for r in rows])})
    run.summary.update({'completed': True, 'training': False, 'recommended': selection['recommended'],
                        'integration_passed': selection['integration']['passed'], 'submission_provenance_verified': False})
    manifest = json.loads((out / 'manifest.json').read_text())
    snapshots = {**manifest['origin_snapshots'], **selection['candidate_snapshots']}
    snapshots.update({'diagnostic-' + key: value for key, value in selection.get('diagnostic_snapshots', {}).items()})
    for label, snapshot in snapshots.items():
        artifact = wandb.Artifact(f'decode-access-{label.lower()}-{run.id}', type='source-code')
        for name in [*snapshot['source_sha256'], 'snapshot.json']:
            artifact.add_file(str(Path(snapshot['directory']) / name), name=name)
        run.log_artifact(artifact)
    for path in out.iterdir():
        if path.is_file() and path.suffix in ('.json', '.csv', '.md'):
            run.save(str(path), base_path=str(out), policy='now')


def audit(out, manifest, hashes, snapshots):
    assembly.audit(out, manifest, hashes, snapshots)
    for snapshot in snapshots:
        if snapshot.get('decode_access_build'):
            verify_access(snapshot['directory'])


def exploration(args):
    if args.repeats < (1 if args.development else 5) or (args.no_wandb and not args.development):
        raise ValueError('Formal exploration requires five repeats and W&B')
    history, origin, hashes, identity, prior_context = load_history(args)
    build_dir = Path(args.build_dir).resolve()
    build = verify_access(build_dir)
    if build['pyx_sha256'] != common.sha256(ROOT / 'task1_decode_access.pyx'):
        raise ValueError('Decode source changed since compilation; rebuild explicitly')
    for name, digest in build['files'].items():
        hashes[str(build_dir / name)] = digest
    hashes[str(build_dir / 'build.json')] = common.sha256(build_dir / 'build.json')
    out = Path(args.output_dir).resolve()
    if not args.development and not out.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Output must be in personal outputs')
    out.mkdir(parents=True, exist_ok=False)
    snapshots, diagnostics = {}, {}
    def get_snapshot(case, diagnostic=False):
        collection = diagnostics if diagnostic else snapshots
        if case not in collection:
            decoder, _, part = case.partition('-P')
            parts = int(part) if part else 256
            collection[case] = make_snapshot(out / (('diagnostic-' if diagnostic else 'candidate-') + case),
                                              origin, decoder, parts, build_dir, diagnostic)
        return collection[case]
    baseline = get_snapshot('N0')
    manifest = {'arguments': vars(args), 'python': sys.version, 'executable': sys.executable,
                'started_utc': datetime.now(UTC).isoformat(), 'source_sha256': common.source_fingerprints(),
                'input_sha256': hashes, 'origin_snapshots': {'M2-A1': origin}, 'native_identity': identity,
                'decode_access_build': build, 'cpu_affinity': history['cpu_affinity'], 'workers': origin['workers'],
                'partition_candidates': list(PARTS), 'cache_capacity': 1048576, 'history_context': prior_context,
                'training': False, 'compilation': False,
                'timing': 'cold public encode/list/decode/compare including first import, index buffer and cleanup'}
    common.save_json(out / 'manifest.json', manifest)
    job_base = {**{k: history[k] for k in ('tokenizer_prefix', 'valid_file', 'reference', 'expected_tokens')},
                'method': origin['method'], 'workers': origin['workers'], 'cpus': history['cpu_affinity'],
                'development': args.development, 'input_sha256': hashes, 'active_wandb': False}
    rows, order, selection = [], [], {}
    try:
        # Read-only preflight uses the exact frozen splitter in a separate
        # interpreter. Every timed encode independently recomputes its ranges.
        availability = native.launch_job({**job_base, 'snapshot': baseline, 'diagnostic': False,
                                          'preflight_parts': list(PARTS)}, out / 'preflight.job.json',
                                         out / 'preflight.json', out / 'preflight.log')['availability']
        if args.no_wandb:
            context = nullcontext(None)
        else:
            from wandb_record import experiment
            context = experiment('bpe-task1-decode-access-exploration', manifest, root=ROOT)
        with native.recording_context(context, out, selection) as run:
            def trial(stage, case, repeat=0, diagnostic=False, plain=False):
                name = f'trial-{len(order) + 1:03d}'
                snapshot = get_snapshot(case, diagnostic)
                record = {'trial_id': name, 'stage': stage, 'method': case, 'repeat': repeat, 'diagnostic': diagnostic,
                          'parts': snapshot['parts']}
                order.append(record)
                common.save_json(out / 'schedule.json', order)
                job = {**job_base, **record, 'method': snapshot['method'], 'snapshot': snapshot, 'plain': plain}
                print(f'{name}: {stage} {case}', flush=True)
                before = system_load()
                row = native.launch_job(job, out / (name + '.job.json'), out / (name + '.json'), out / (name + '.log'))
                after = system_load()
                pipeline.check_row(row)
                row.update(record, native_method=snapshot['method'], system_load_before=before, system_load_after=after)
                rows.append(row)
                common.save_json(out / (name + '.json'), row)
                record_outputs(out, rows, selection)
                return row
            selection.update(run_stages(args, trial, availability))
            original, ordinary = [], []
            rng = random.Random(args.seed + 2)
            for repeat in range(1, args.repeats + 1):
                pair = [('integration-experimental', False, original), ('integration-plain', True, ordinary)]
                rng.shuffle(pair)
                for stage, plain, target in pair:
                    target.append(trial(stage, selection['recommended'], repeat, plain=plain))
            selection.update(integration=integration_decision(original, ordinary, args.repeats),
                             snapshot=snapshots[selection['recommended']], candidate_snapshots=snapshots,
                             diagnostic_snapshots=diagnostics, native_identity=identity, decode_access_build=build,
                             history_context=prior_context,
                             **{k: history[k] for k in ('tokenizer_prefix', 'valid_file', 'reference', 'expected_tokens',
                                                       'cpu_affinity', 'development')}, input_sha256=hashes)
            all_snapshots = [origin, *snapshots.values(), *diagnostics.values()]
            assert_identical_native(all_snapshots)
            audit(out, manifest, hashes, all_snapshots)
            selection.update(measurements_valid=True, ready_for_plain_confirmation=selection['integration']['passed'])
            common.save_json(out / 'selection.json', selection)
            record_outputs(out, rows, selection)
            try:
                publish(run, out, rows, selection)
            except Exception as error:
                selection.update(upload_status='failed', upload_error=repr(error))
                common.save_json(out / 'upload_failure.json', {'error': repr(error), 'measurements_valid': True})
            else:
                selection['upload_status'] = 'disabled' if run is None else 'submitted'
            common.save_json(out / 'selection.json', selection)
        audit(out, manifest, hashes, all_snapshots)
    except BaseException as error:
        selection.update(completed=False, measurements_valid=False, ready_for_plain_confirmation=False, error=repr(error))
        common.save_json(out / 'selection.json', selection)
        common.save_json(out / 'failure.json', {'completed': False, 'error': repr(error)})
        record_outputs(out, rows, selection)
        raise
    record_outputs(out, rows, selection)
    print(f'Completed: {out / "report.md"}', flush=True)


def plain_confirmation(args):
    report = json.loads(Path(args.selection_report).read_text())
    snapshot = report['snapshot']
    if snapshot.get('decode_access_diagnostic'):
        raise ValueError('Cannot confirm diagnostic decoder')
    verify_build(snapshot['directory'])
    verify_assembly(snapshot['directory'])
    if snapshot.get('decode_access_build'):
        verify_access(snapshot['directory'])
    native.previous.plain_confirmation(args)
    path = Path(args.output_dir).resolve() / 'confirmation.json'
    result = json.loads(path.read_text())
    result['decode_access_recommended'] = report['recommended']
    common.save_json(path, result)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('exploration', 'plain-confirmation'), default='exploration')
    parser.add_argument('--previous-selection', default='outputs/task1-assembly/selection.json')
    parser.add_argument('--plain-confirmation', default='outputs/task1-assembly-plain-confirmation/confirmation.json')
    parser.add_argument('--selection-report', default='outputs/task1-decode-access/selection.json')
    parser.add_argument('--build-dir', default='outputs/task1-decode-access-build')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261019)
    parser.add_argument('--diagnostics', action='store_true')
    parser.add_argument('--cpu-ids', type=int, nargs='+')
    parser.add_argument('--output-dir')
    parser.add_argument('--development', action='store_true')
    parser.add_argument('--no-wandb', action='store_true')
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = ('outputs/task1-decode-access' if args.phase == 'exploration'
                           else 'outputs/task1-decode-access-plain-confirmation')
    return args


def main():
    args = parse_args()
    native.previous.restrict_environment()
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Decode access exploration interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    if args.phase == 'plain-confirmation':
        plain_confirmation(args)
    else:
        exploration(args)


if __name__ == '__main__':
    main()
