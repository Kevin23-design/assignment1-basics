"""Parent dispatch and pooled final-list assembly exploration; no training or runtime build."""
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
from task1_parent_pipeline_experiment import CASES, make_snapshot
from task1_benchmark_reporting import table_rows
from task1_decode_access_runtime import verify_build as verify_decode
from task1_parent_pipeline_runtime import verify_build as verify_parent
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
            or history.get('recommended') != 'N2'):
        raise ValueError('Requires completed, valid, accepted N2/256 results')
    if bool(history['development']) != bool(args.development):
        raise ValueError('Development provenance cannot be used as a formal result')
    origin = history['snapshot']
    if (origin != history['candidate_snapshots']['N2'] or origin['method'] != 'CY-ED'
            or origin.get('decode_access_case') != 'N2' or origin.get('parts') != 256
            or origin.get('decode_access_diagnostic') or origin.get('assembly_diagnostic')
            or origin.get('diagnostic_instrumentation') or origin.get('parent_pipeline_case')
            or origin.get('assembly_method') != 'M2' or origin.get('adapter_case') != 'A1'
            or origin['development'] != history['development']):
        raise ValueError('Historical source is not normal N2/256')
    verify_snapshot(origin)
    verify_build(origin['directory'])
    verify_assembly(origin['directory'])
    verify_decode(origin['directory'])
    pipeline.check_row(confirmation)
    if (confirmation.get('patching') is not False or confirmation.get('source_changed') is not False
            or confirmation.get('diagnostic') or confirmation.get('method') != 'CY-ED'
            or confirmation.get('decode_access_case') != 'N2' or confirmation.get('parts') != 256
            or confirmation.get('token_count') != history['expected_tokens']
            or confirmation.get('cpu_affinity') != history['cpu_affinity']
            or Path(confirmation['bpe_source']).resolve() != Path(origin['directory']) / 'bpe.py'
            or confirmation.get('native_decode', {}).get('backend') != 'native'
            or confirmation['native_decode'].get('decoder_case') != 'N2'
            or confirmation['native_decode'].get('index_buffer_bytes') != history['expected_tokens'] * 4
            or confirmation.get('assembly', {}).get('method') != 'M2'
            or confirmation['assembly'].get('converted_ids') != history['expected_tokens']
            or (not args.development and not confirmation.get('active_wandb'))):
        raise ValueError('Plain confirmation does not match accepted N2/256')
    for metric, build in [('native_decode', 'decode_access_build'), ('assembly', 'assembly_build')]:
        expected = Path(origin['directory']) / origin[build]['binary']
        if Path(confirmation[metric]['extension_path']).resolve() != expected.resolve():
            raise ValueError('Plain confirmation used another extension binary')
    workers = origin['workers']
    if not 1 <= workers <= 16 or (not args.development and
            (workers != 16 or history['expected_tokens'] != native.previous.EXPECTED_TOKENS)):
        raise ValueError('Formal experiment requires 16 workers and complete validation')
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
        raise ValueError('Must inherit exact CPU allocation')
    if cpus:
        if len(cpus) != workers or len(set(cpus)) != workers or not hasattr(os, 'sched_getaffinity') \
                or not set(cpus) <= os.sched_getaffinity(0):
            raise ValueError('Historical CPU allocation unavailable; no downgrade')
        native.previous.affinity(cpus, include_descendants=True)
    elif not args.development:
        raise ValueError('Formal experiment requires Linux affinity')
    context = {key: confirmation.get(key) for key in
               ('total_seconds', 'context_ratio_to_plain_median', 'context_regression_flag')}
    context.update(upload_status=history.get('upload_status'), upload_error=history.get('upload_error'),
                   note='Historical upload failure is preserved separately from measurement validity')
    return history, origin, hashes, assert_identical_native([origin]), context


def run_stages(args, trial):
    rng = random.Random(args.seed)
    def paired(baseline, candidate, stage):
        before, after = [], []
        for repeat in range(1, args.repeats + 1):
            pair = [(stage + '-baseline', baseline, before), (stage + '-candidate', candidate, after)]
            rng.shuffle(pair)
            for label, case, target in pair:
                target.append(trial(label, case, repeat))
        return common.paired_decision(before, after, args.repeats)
    if args.diagnostics:
        for case in CASES:
            pipeline.check_row(trial('diagnostic', case, diagnostic=True))
    rows = {case: [] for case in CASES}
    for repeat in range(1, args.repeats + 1):
        order = list(CASES)
        rng.shuffle(order)
        for case in order:
            row = trial('screening', case, repeat)
            pipeline.check_row(row)
            rows[case].append(row)
    medians = {case: statistics.median(r['total_seconds'] for r in values) for case, values in rows.items()}
    best = min(medians, key=lambda case: (medians[case], case != 'Q0', case))
    recommended = 'Q0'
    decision = {'promote_candidate': False, 'reason': 'baseline_screened_fastest'}
    if best != 'Q0':
        decision = paired('Q0', best, 'paired')
        if decision['promote_candidate']:
            recommended = best
    mechanism = {'skipped': True, 'reason': 'no_promoted_native_assembly'}
    if recommended in ('Q2', 'Q3'):
        mechanism = paired('Q1', recommended, 'assembly-mechanism')
        mechanism['ranking_effect'] = 'none; dispatch held fixed, explanatory only'
    confirmation = trial('confirmation', recommended)
    pipeline.check_row(confirmation)
    return {'completed': True, 'recommended': recommended, 'screening_medians_seconds': medians,
            'decision': decision, 'assembly_mechanism': mechanism,
            'confirmation_seconds': confirmation['total_seconds'], 'confirmation_trial': confirmation['trial_id'],
            'submission_provenance_verified': False}


def record_outputs(out, rows, selection):
    assembly.record_outputs(out, rows, selection)
    path = out / 'report.md'
    text = path.read_text().replace('任务一：原生列表组装与整数复用探索', '任务一：父进程调度与最终列表组装探索')
    text = text.replace('M0=当前array扩展；M1=原生整数转换；M2=原生整数池。-A1表示固定组装后组合修复的原生编码。',
                        'Q0=已验收N2；Q1=先派发再组装；Q2=Q1加直接原生append；Q3=Q1加收齐实际结果后精确分配列表。')
    text = text.replace('正常解码保留原二进制；诊断分阶段解码是独立实现，不进入排名。',
                        '编码、原组装及N2解码二进制保留。Q3最终组装另记finish_assembly_seconds，仍计入完整encode。')
    text += '\n结构化Table单元格使用JSON文本，完整原始结构保存在逐次JSON；历史上传失败不改写。\n'
    path.write_text(text)


def publish(run, out, rows, selection):
    if run is None:
        return
    import wandb
    columns = ['trial_id', 'stage', 'method', 'parts', 'diagnostic', 'status', 'total_seconds', 'encode_seconds',
               'decode_seconds', 'compare_seconds', 'token_count', 'ids_match_reference', 'roundtrip_passed',
               'cpu_affinity', 'parent_peak_rss_bytes', 'native_decode', 'assembly', 'system_load_before', 'system_load_after']
    run.log({'parent_pipeline/trials': wandb.Table(columns=columns, data=table_rows(columns, rows))})
    run.summary.update({'completed': True, 'training': False, 'recommended': selection['recommended'],
                        'integration_passed': selection['integration']['passed'], 'submission_provenance_verified': False})
    manifest = json.loads((out / 'manifest.json').read_text())
    snapshots = {**manifest['origin_snapshots'], **selection['candidate_snapshots']}
    snapshots.update({'diagnostic-' + key: value for key, value in selection.get('diagnostic_snapshots', {}).items()})
    for label, snapshot in snapshots.items():
        artifact = wandb.Artifact(f'parent-pipeline-{label.lower()}-{run.id}', type='source-code')
        for name in [*snapshot['source_sha256'], 'snapshot.json']:
            artifact.add_file(str(Path(snapshot['directory']) / name), name=name)
        run.log_artifact(artifact)
    for path in out.iterdir():
        if path.is_file() and path.suffix in ('.json', '.csv', '.md'):
            run.save(str(path), base_path=str(out), policy='now')


def audit(out, manifest, hashes, snapshots):
    assembly.audit(out, manifest, hashes, snapshots)
    for snapshot in snapshots:
        verify_decode(snapshot['directory'])
        if snapshot.get('parent_assembly_build'):
            verify_parent(snapshot['directory'])


def exploration(args):
    if args.repeats < (1 if args.development else 5) or (args.no_wandb and not args.development):
        raise ValueError('Formal exploration requires five repeats and W&B')
    history, origin, hashes, identity, prior_context = load_history(args)
    build_dir = Path(args.build_dir).resolve()
    build = verify_parent(build_dir)
    if build['pyx_sha256'] != common.sha256(ROOT / 'task1_parent_assembly.pyx'):
        raise ValueError('Parent assembly source changed since compilation; rebuild explicitly')
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
            collection[case] = make_snapshot(out / (('diagnostic-' if diagnostic else 'candidate-') + case),
                                              origin, case, build_dir, diagnostic)
        return collection[case]
    baseline = get_snapshot('Q0')
    manifest = {'arguments': vars(args), 'python': sys.version, 'executable': sys.executable,
                'started_utc': datetime.now(UTC).isoformat(), 'source_sha256': common.source_fingerprints(),
                'input_sha256': hashes, 'origin_snapshots': {'N2': origin}, 'native_identity': identity,
                'parent_assembly_build': build, 'cpu_affinity': history['cpu_affinity'], 'workers': origin['workers'],
                'parts': 256, 'cache_capacity': 1048576, 'history_context': prior_context,
                'training': False, 'compilation': False,
                'timing': 'cold public encode/list/decode/compare including first import, assembly and cleanup'}
    common.save_json(out / 'manifest.json', manifest)
    job_base = {**{k: history[k] for k in ('tokenizer_prefix', 'valid_file', 'reference', 'expected_tokens')},
                'method': origin['method'], 'workers': origin['workers'], 'cpus': history['cpu_affinity'],
                'development': args.development, 'input_sha256': hashes, 'active_wandb': False}
    rows, order, selection = [], [], {}
    try:
        # Read-only preflight uses the exact frozen splitter in a separate
        # interpreter. Every timed encode independently recomputes its ranges.
        preflight = native.launch_job({**job_base, 'snapshot': baseline, 'diagnostic': False,
                                          'preflight_parts': [256]}, out / 'preflight.job.json',
                                         out / 'preflight.json', out / 'preflight.log')['availability']
        if not preflight.get('256'):
            raise ValueError('Cannot produce 256 safe nonempty partitions')
        if args.no_wandb:
            context = nullcontext(None)
        else:
            from wandb_record import experiment
            context = experiment('bpe-task1-parent-pipeline-exploration', manifest, root=ROOT)
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
            selection.update(run_stages(args, trial))
            original, ordinary = [], []
            rng = random.Random(args.seed + 2)
            for repeat in range(1, args.repeats + 1):
                pair = [('integration-experimental', False, original), ('integration-plain', True, ordinary)]
                rng.shuffle(pair)
                for stage, plain, target in pair:
                    target.append(trial(stage, selection['recommended'], repeat, plain=plain))
            selection.update(integration=integration_decision(original, ordinary, args.repeats),
                             snapshot=snapshots[selection['recommended']], candidate_snapshots=snapshots,
                             diagnostic_snapshots=diagnostics, native_identity=identity, parent_assembly_build=build,
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
    if snapshot.get('parent_pipeline_diagnostic'):
        raise ValueError('Cannot confirm diagnostic decoder')
    verify_build(snapshot['directory'])
    verify_assembly(snapshot['directory'])
    verify_decode(snapshot['directory'])
    if snapshot.get('parent_assembly_build'):
        verify_parent(snapshot['directory'])
    native.previous.plain_confirmation(args)
    path = Path(args.output_dir).resolve() / 'confirmation.json'
    result = json.loads(path.read_text())
    result['parent_pipeline_recommended'] = report['recommended']
    common.save_json(path, result)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('exploration', 'plain-confirmation'), default='exploration')
    parser.add_argument('--previous-selection', default='outputs/task1-decode-access/selection.json')
    parser.add_argument('--plain-confirmation', default='outputs/task1-decode-access-plain-confirmation/confirmation.json')
    parser.add_argument('--selection-report', default='outputs/task1-parent-pipeline/selection.json')
    parser.add_argument('--build-dir', default='outputs/task1-parent-assembly-build')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261019)
    parser.add_argument('--diagnostics', action='store_true')
    parser.add_argument('--cpu-ids', type=int, nargs='+')
    parser.add_argument('--output-dir')
    parser.add_argument('--development', action='store_true')
    parser.add_argument('--no-wandb', action='store_true')
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = ('outputs/task1-parent-pipeline' if args.phase == 'exploration'
                           else 'outputs/task1-parent-pipeline-plain-confirmation')
    return args


def main():
    args = parse_args()
    native.previous.restrict_environment()
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Parent pipeline exploration interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    if args.phase == 'plain-confirmation':
        plain_confirmation(args)
    else:
        exploration(args)


if __name__ == '__main__':
    main()
