"""Native ID-list assembly and integer-pool exploration; no training or runtime build."""
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
import sys

import benchmark_task1_native as native
from task1_decode_experiment import integration_decision, verify_snapshot
from task1_native_adapter_experiment import assert_identical_native
from task1_assembly_experiment import CASES, make_snapshot
from task1_assembly_runtime import verify_build as verify_assembly_build
from task1_native_runtime import verify_build

common, pipeline = native.common, native.pipeline
ROOT = Path(__file__).resolve().parent


def load_history(args):
    report, confirmation_path = Path(args.previous_selection).resolve(), Path(args.plain_confirmation).resolve()
    history, confirmation = json.loads(report.read_text()), json.loads(confirmation_path.read_text())
    if (not history.get('completed') or not history.get('measurements_valid')
            or not history.get('ready_for_plain_confirmation') or not history.get('integration', {}).get('passed')
            or history.get('recommended') != 'A0'):
        raise ValueError('Requires completed, valid, accepted A0/CY-D adapter results')
    if bool(history['development']) != bool(args.development):
        raise ValueError('Development provenance cannot be used as a formal result')
    origins = {key: history['candidate_snapshots'][key] for key in ('A0', 'A1')}
    if history['snapshot'] != origins['A0'] or any(s['method'] != ('CY-D' if k == 'A0' else 'CY-ED')
            or s.get('adapter_case') != k or s.get('diagnostic_instrumentation')
            for k, s in origins.items()):
        raise ValueError('Selected source does not match the previous native candidates')
    for snapshot in origins.values():
        if snapshot['workers'] != origins['A0']['workers'] or snapshot['development'] != history['development']:
            raise ValueError('Historical candidate resource configuration differs')
        verify_snapshot(snapshot)
        verify_build(snapshot['directory'])
    identity = assert_identical_native(origins.values())
    pipeline.check_row(confirmation)
    if (confirmation.get('patching') is not False or confirmation.get('source_changed') is not False
            or confirmation.get('method') != 'CY-D' or confirmation.get('adapter_case') != 'A0' or confirmation.get('token_count') != history['expected_tokens']
            or confirmation.get('cpu_affinity') != history['cpu_affinity']
            or Path(confirmation['bpe_source']).resolve() != Path(origins['A0']['directory']) / 'bpe.py'
            or confirmation.get('native_decode', {}).get('backend') != 'native'
            or (not args.development and not confirmation.get('active_wandb'))):
        raise ValueError('Plain confirmation does not match the accepted native baseline')
    expected_binary = Path(origins['A0']['directory']) / origins['A0']['native_build']['binary']
    if Path(confirmation['native_decode']['extension_path']).resolve() != expected_binary.resolve():
        raise ValueError('Plain confirmation used another native binary')
    workers = origins['A0']['workers']
    if not 1 <= workers <= 16 or (not args.development and
            (workers != 16 or history['expected_tokens'] != native.previous.EXPECTED_TOKENS)):
        raise ValueError('Formal experiment requires 16 workers and the complete validation token count')
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
            raise ValueError('Historical CPU allocation unavailable; no silent downgrade')
        native.previous.affinity(cpus, include_descendants=True)
    elif not args.development:
        raise ValueError('Formal experiment requires Linux affinity')
    return history, origins, hashes, identity, {
        'seconds': confirmation['total_seconds'],
        'context_ratio_to_plain_median': confirmation.get('context_ratio_to_plain_median'),
        'context_regression_flag': confirmation.get('context_regression_flag', False),
        'note': 'A context regression is retained as a warning, not treated as an ID failure'}


def system_load():
    return {'utc': datetime.now(UTC).isoformat(), 'average_1_5_15m': list(os.getloadavg()) if hasattr(os, 'getloadavg') else None}


def run_stages(args, trial):
    rng = random.Random(args.seed)
    if args.diagnostics:
        for case in ('M0', 'M1', 'M2'):
            pipeline.check_row(trial('diagnostic-assembly', case, diagnostic=True))
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
    medians = screen(('M0', 'M1', 'M2'), 'assembly-screening')
    candidate = min(medians, key=lambda case: (medians[case], case != 'M0', case))
    decision = {'promote_candidate': False, 'reason': 'baseline_screened_fastest'}
    recommended = 'M0'
    if candidate != 'M0':
        decision = paired('M0', candidate, 'assembly-paired')
        if decision['promote_candidate']:
            recommended = candidate
    mechanism = {'skipped': True, 'reason': 'M2_not_promoted'}
    if recommended == 'M2':
        mechanism = paired('M1', 'M2', 'integer-pool-mechanism')
        mechanism['ranking_effect'] = 'none; explains pool contribution only'
    combination = {'skipped': True, 'reason': 'no_promoted_assembly'}
    if recommended != 'M0':
        combined = recommended + '-A1'
        combination_times = screen((recommended, combined), 'combination-screening')
        combination = paired(recommended, combined, 'combination-paired')
        combination['screening_medians_seconds'] = combination_times
        if args.diagnostics:
            pipeline.check_row(trial('diagnostic-combination', combined, diagnostic=True))
        if combination['promote_candidate']:
            recommended = combined
    confirmation = trial('confirmation', recommended)
    pipeline.check_row(confirmation)
    return {'completed': True, 'recommended': recommended, 'assembly_candidate': candidate,
            'screening_medians_seconds': medians, 'assembly_decision': decision,
            'pool_mechanism': mechanism, 'combination_decision': combination,
            'confirmation_seconds': confirmation['total_seconds'], 'confirmation_trial': confirmation['trial_id'],
            'submission_provenance_verified': False}


def record_outputs(out, rows, selection):
    common.save_json(out / 'results.json', rows)
    columns = ['trial_id', 'stage', 'method', 'native_method', 'repeat', 'diagnostic', 'status', 'patching',
               'timing_scope', 'encode_workers', 'parts', 'cache_capacity', 'total_seconds', 'encode_seconds',
               'decode_seconds', 'compare_seconds', 'token_count', 'ids_match_reference', 'roundtrip_passed',
               'parent_peak_rss_bytes', 'encode_worker_peak_rss_max', 'encode_worker_peak_rss_sum',
               'cpu_affinity', 'assembly', 'system_load_before', 'system_load_after', 'native_decode', 'error']
    with (out / 'results.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# 任务一：原生列表组装与整数复用探索', '', 'M0=当前array扩展；M1=原生整数转换；M2=原生整数池。-A1表示固定组装后组合修复的原生编码。',
             '正常解码保留原二进制；诊断分阶段解码是独立实现，不进入排名。', '',
             '| 阶段 | 方案 | 次数 | 完整中位数(s) | 最小(s) | 最大(s) |',
             '|---|---|---:|---:|---:|---:|']
    groups = {}
    for row in rows:
        if row['status'] == 'ok' and not row['diagnostic']:
            groups.setdefault((row['stage'], row['method']), []).append(row['total_seconds'])
    for (stage, method), values in groups.items():
        lines.append(f'| {stage} | {method} | {len(values)} | {statistics.median(values):.3f} | '
                     f'{min(values):.3f} | {max(values):.3f} |')
    lines += ['', '## 选优与接入', '', '```json', json.dumps(selection, ensure_ascii=False, indent=2), '```', '',
              '诊断不参与排名；复制包含首次触页，分配时间不等于全部内存写入成本。',
              'worker峰值RSS之和不是同时发生的实际峰值。系统负载为整机load average，不是本组CPU利用率。',
              '活跃W&B超过接入中位数5%时保留回退标记，不直接将差异归因于W&B。',
              '不训练、不编译、不自动更新dev；公开接口探索不替代课程入口成绩。']
    for row in rows:
        if row['diagnostic']:
            detail = {k: row.get(k) for k in ('transport', 'assembly', 'native_decode', 'worker_metadata',
                                            'system_load_before', 'system_load_after')}
            lines += ['', '## 独立诊断 ' + row['trial_id'], '', '```json',
                      json.dumps(detail, ensure_ascii=False, indent=2), '```']
    (out / 'report.md').write_text('\n'.join(lines) + '\n')


def publish(run, out, rows, selection):
    if run is None:
        return
    import wandb
    columns = ['trial_id', 'stage', 'method', 'diagnostic', 'status', 'total_seconds', 'encode_seconds',
               'decode_seconds', 'compare_seconds', 'token_count', 'ids_match_reference', 'roundtrip_passed',
               'cpu_affinity', 'parent_peak_rss_bytes', 'system_load_before', 'system_load_after']
    run.log({'assembly/trials': wandb.Table(columns=columns, data=[[r.get(k) for k in columns] for r in rows])})
    run.summary.update({'completed': True, 'training': False, 'recommended': selection['recommended'],
                        'integration_passed': selection['integration']['passed'], 'submission_provenance_verified': False})
    manifest = json.loads((out / 'manifest.json').read_text())
    snapshots = {**manifest['origin_snapshots'], **selection['candidate_snapshots']}
    snapshots.update({'diagnostic-' + k: v for k, v in selection.get('diagnostic_snapshots', {}).items()})
    for label, snapshot in snapshots.items():
        artifact = wandb.Artifact(f'assembly-{label.lower()}-{run.id}', type='source-code')
        for name in [*snapshot['source_sha256'], 'snapshot.json']:
            artifact.add_file(str(Path(snapshot['directory']) / name), name=name)
        run.log_artifact(artifact)
    for path in out.iterdir():
        if path.is_file() and path.suffix in ('.json', '.csv', '.md'):
            run.save(str(path), base_path=str(out), policy='now')


def audit(out, manifest, hashes, snapshots):
    for snapshot in snapshots:
        verify_snapshot(snapshot)
        verify_build(snapshot['directory'])
        if snapshot.get('assembly_build'):
            verify_assembly_build(snapshot['directory'])
    changed = [p for p, digest in hashes.items() if common.sha256(p) != digest]
    source_changed = common.source_fingerprints() != manifest['source_sha256']
    common.save_json(out / 'integrity.json', {'changed_inputs': changed, 'source_changed': source_changed})
    if changed or source_changed:
        raise ValueError('Source/input changed; no valid recommendation')


def exploration(args):
    if args.repeats < (1 if args.development else 5) or (args.no_wandb and not args.development):
        raise ValueError('Formal exploration requires five repeats and W&B')
    history, origins, hashes, identity, prior_context = load_history(args)
    build_dir = Path(args.build_dir).resolve()
    build = verify_assembly_build(build_dir)
    if build['pyx_sha256'] != common.sha256(ROOT / 'task1_assembly.pyx'):
        raise ValueError('Assembly source changed since compilation; rebuild explicitly')
    for name, digest in build['files'].items():
        hashes[str(build_dir / name)] = digest
    hashes[str(build_dir / 'build.json')] = common.sha256(build_dir / 'build.json')
    out = Path(args.output_dir).resolve()
    if not args.development and not out.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Output must be in personal outputs')
    out.mkdir(parents=True, exist_ok=False)
    snapshots = {case: make_snapshot(out / ('candidate-' + case),
                                     origins['A1' if case.endswith('-A1') else 'A0'], case, args.build_dir)
                 for case in CASES}
    diagnostics = {case: make_snapshot(out / ('diagnostic-' + case),
                                      origins['A1' if case.endswith('-A1') else 'A0'], case, args.build_dir,
                                      diagnostic=True) for case in CASES} if args.diagnostics else {}
    all_snapshots = [*origins.values(), *snapshots.values(), *diagnostics.values()]
    assert_identical_native(all_snapshots)
    manifest = {'arguments': vars(args), 'python': sys.version, 'executable': sys.executable,
                'started_utc': datetime.now(UTC).isoformat(), 'source_sha256': common.source_fingerprints(),
                'input_sha256': hashes, 'origin_snapshots': origins, 'native_identity': identity, 'assembly_build': build,
                'cpu_affinity': history['cpu_affinity'], 'workers': origins['A0']['workers'],
                'parts': origins['A0']['workers'] * 16, 'cache_capacity': 1048576,
                'history_context': prior_context, 'training': False, 'compilation': False,
                'timing': 'cold public encode/list/decode/compare including first import and cleanup'}
    common.save_json(out / 'manifest.json', manifest)
    if args.no_wandb:
        context = nullcontext(None)
    else:
        from wandb_record import experiment
        context = experiment('bpe-task1-assembly-exploration', manifest, root=ROOT)
    rows, order, selection = [], [], {}
    try:
        with native.recording_context(context, out, selection) as run:
            def trial(stage, case, repeat=0, diagnostic=False, plain=False):
                name = f'trial-{len(order) + 1:03d}'
                record = {'trial_id': name, 'stage': stage, 'method': case, 'repeat': repeat, 'diagnostic': diagnostic}
                order.append(record)
                common.save_json(out / 'schedule.json', order)
                snapshot = (diagnostics if diagnostic else snapshots)[case]
                job = {**record, **{k: history[k] for k in ('tokenizer_prefix', 'valid_file', 'reference', 'expected_tokens')},
                       'method': snapshot['method'], 'snapshot': snapshot, 'workers': snapshot['workers'],
                       'cpus': history['cpu_affinity'], 'development': args.development, 'input_sha256': hashes,
                       'plain': plain, 'active_wandb': False}
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
                             diagnostic_snapshots=diagnostics, native_identity=identity, assembly_build=build, history_context=prior_context,
                             **{k: history[k] for k in ('tokenizer_prefix', 'valid_file', 'reference', 'expected_tokens',
                                                       'cpu_affinity', 'development')}, input_sha256=hashes)
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
        # Include changes during source upload/run finalization in the validity audit.
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
    if report['snapshot'].get('assembly_diagnostic'):
        raise ValueError('Cannot confirm a diagnostic decoder as a normal result')
    verify_build(report['snapshot']['directory'])
    verify_assembly_build(report['snapshot']['directory'])
    native.previous.plain_confirmation(args)
    path = Path(args.output_dir).resolve() / 'confirmation.json'
    result = json.loads(path.read_text())
    result['assembly_case'] = report['recommended']
    common.save_json(path, result)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('exploration', 'plain-confirmation'), default='exploration')
    parser.add_argument('--previous-selection', default='outputs/task1-native-adapter/selection.json')
    parser.add_argument('--plain-confirmation', default='outputs/task1-native-adapter-plain-confirmation/confirmation.json')
    parser.add_argument('--selection-report', default='outputs/task1-assembly/selection.json')
    parser.add_argument('--build-dir', default='outputs/task1-assembly-build')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261019)
    parser.add_argument('--diagnostics', action='store_true')
    parser.add_argument('--cpu-ids', type=int, nargs='+')
    parser.add_argument('--output-dir')
    parser.add_argument('--development', action='store_true')
    parser.add_argument('--no-wandb', action='store_true')
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = ('outputs/task1-assembly' if args.phase == 'exploration'
                           else 'outputs/task1-assembly-plain-confirmation')
    return args


def main():
    args = parse_args()
    native.previous.restrict_environment()
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Assembly exploration interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    if args.phase == 'plain-confirmation':
        plain_confirmation(args)
    else:
        exploration(args)


if __name__ == '__main__':
    main()
