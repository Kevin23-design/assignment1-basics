"""Recover saved results after the specific W&B SummaryDict.update keyword bug.

Never encodes, alters timings, overwrites original results or bypasses gates.
Only the two exact logging-only source corrections may differ from the manifest.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import benchmark_task1_encode_transport as benchmark

# Reverse only the known Summary.update fix to verify the original source digest.
FIXES = {
    'benchmark_task1_encode_transport.py': [
        ("    run.summary.update(completed=True, training=False, recommended=selection['recommended'],\n"
         "                       integration_passed=selection['integration']['passed'], submission_provenance_verified=False)",
         "    run.summary.update({'completed': True, 'training': False, 'recommended': selection['recommended'],\n"
         "                        'integration_passed': selection['integration']['passed'], 'submission_provenance_verified': False})"),
        ("                run.summary.update(completed=False, error=repr(error))",
         "                run.summary.update({'completed': False, 'error': repr(error)})"),
    ],
    'benchmark_task1_decode.py': [
        ("                run.summary.update(completed=False, error=repr(error))",
         "                run.summary.update({'completed': False, 'error': repr(error)})"),
    ],
}


def check_sources(recorded):
    current = benchmark.common.source_fingerprints()
    missing = set(recorded) - set(current)
    added = set(current) - set(recorded)
    allowed_added = {'recover_task1_encode_transport.py', 'tests/test_task1_summary_recovery.py'}
    if missing or added - allowed_added:
        raise ValueError(f'Unexpected source files changed: missing={sorted(missing)}, added={sorted(added)}')
    corrected = []
    for name, digest in recorded.items():
        if current[name] == digest:
            continue
        if name not in FIXES:
            raise ValueError(f'Non-logging source changed: {name}')
        source = (benchmark.ROOT / name).read_text()
        for old, new in FIXES[name]:
            if source.count(new) != 1:
                raise ValueError(f'Cannot prove exact Summary.update-only correction: {name}')
            source = source.replace(new, old)
        if hashlib.sha256(source.encode()).hexdigest() != digest:
            raise ValueError(f'Changes beyond Summary.update correction: {name}')
        corrected.append(name)
    return corrected


def known_failure(error):
    return (isinstance(error, str) and error.startswith('TypeError(')
            and "SummaryDict.update() got an unexpected keyword argument 'completed'" in error)


def validate_results(out, manifest, selection, rows, schedule):
    args = SimpleNamespace(**manifest['arguments'])
    development = bool(args.development)
    if bool(selection['development']) != development or args.repeats < (1 if development else 5):
        raise ValueError('Invalid repeat count or development provenance')
    workers, expected = manifest['workers'], selection['expected_tokens']
    if not development and (workers != 16 or expected != benchmark.previous.EXPECTED_TOKENS):
        raise ValueError('Formal resources/token count do not match full validation')
    if manifest['cpu_affinity'] != selection['cpu_affinity'] or selection['snapshot']['workers'] != workers:
        raise ValueError('Inconsistent saved resources')
    if selection['input_sha256'] != manifest['input_sha256']:
        raise ValueError('Saved input fingerprints disagree')
    if len(rows) != len(schedule) or not rows:
        raise ValueError('Missing trial results or schedule')
    seen = set()
    record_keys = ('trial_id', 'stage', 'method', 'repeat', 'diagnostic')
    for row, record in zip(rows, schedule, strict=True):
        if {k: row[k] for k in record_keys} != record:
            raise ValueError('Trial schedule/order differs from saved results')
        name = row['trial_id']
        if name in seen or name != f'trial-{len(seen) + 1:03d}':
            raise ValueError('Duplicate or invalid trial ID')
        seen.add(name)
        disk = json.loads((out / (name + '.json')).read_text())
        job = json.loads((out / (name + '.job.json')).read_text())
        if disk != row or any(job[k] != row[k] for k in record_keys):
            raise ValueError('Per-trial file/job disagrees with aggregate results')
        if (job['input_sha256'] != manifest['input_sha256']
                or job['baseline_snapshot'] != manifest['baseline_snapshot']
                or job['cpus'] != selection['cpu_affinity'] or job['workers'] != workers
                or bool(job['development']) != development
                or any(job[k] != selection[k] for k in ('tokenizer_prefix', 'valid_file', 'reference', 'expected_tokens'))):
            raise ValueError('Trial inputs/resources differ from the manifest')
        if job['plain'] != (row['stage'] == 'integration-plain'):
            raise ValueError('Unexpected trial interface')
        if job['plain'] and job['snapshot'] != selection['snapshot']:
            raise ValueError('Plain trial used a different candidate snapshot')
        if row['status'] == 'unavailable':
            if row['method'] not in ('T2', 'T3'):
                raise ValueError('Invalid unavailable baseline/pipe candidate')
            continue
        benchmark.pipeline.check_row(row)
        phases = [row[k] for k in ('encode_seconds', 'decode_seconds', 'compare_seconds')]
        if (row['token_count'] != expected or row['returned_type'] != 'list'
                or row['cpu_affinity'] != selection['cpu_affinity']
                or row['patching'] != (not job['plain'])
                or any(not math.isfinite(v) or v < 0 for v in phases)
                or not math.isfinite(row['total_seconds']) or row['total_seconds'] <= 0
                or not math.isclose(sum(phases), row['total_seconds'], rel_tol=1e-8, abs_tol=1e-6)):
            raise ValueError('Invalid saved correctness, timing or resource fields')
    # Replay selection decisions against saved rows only. No benchmark is launched.
    iterator = iter(rows)
    def replay(stage, method, repeat=0, diagnostic=False):
        row = next(iterator)
        if (row['stage'], row['method'], row['repeat'], row['diagnostic']) != (stage, method, repeat, diagnostic):
            raise ValueError('Saved trials do not reproduce the configured experiment')
        return row
    reproduced = benchmark.run_stages(args, replay)
    for key, value in reproduced.items():
        if key != 'completed' and selection.get(key) != value:
            raise ValueError(f'Saved selection cannot be reproduced: {key}')
    integration_rows = list(iterator)
    if (len(integration_rows) != args.repeats * 2
            or any(r['method'] != reproduced['recommended'] or r['stage'] not in
                   ('integration-experimental', 'integration-plain') for r in integration_rows)):
        raise ValueError('Incomplete or mismatched integration trials')
    integration = benchmark.integration_decision(
        [r for r in integration_rows if r['stage'] == 'integration-experimental'],
        [r for r in integration_rows if r['stage'] == 'integration-plain'], args.repeats)
    if integration != selection['integration'] or not integration['passed']:
        raise ValueError('Actual integration gate did not pass; cannot enable confirmation')
    if selection['snapshot']['method'] != reproduced['recommended']:
        raise ValueError('Snapshot method does not match recommendation')
    return integration


def recover(results_dir):
    out = Path(results_dir).resolve()
    def read(name):
        return json.loads((out / name).read_text())
    failure, manifest, selection = read('failure.json'), read('manifest.json'), read('selection.json')
    if not known_failure(failure.get('error')) or selection.get('error') != failure['error']:
        raise ValueError('Recovery is limited to the specific SummaryDict.update keyword failure')
    if read('integrity.json') != {'changed_inputs': [], 'source_changed': False}:
        raise ValueError('Original run did not pass source/input integrity checks')
    corrected = check_sources(manifest['source_sha256'])
    for snapshot in (manifest['baseline_snapshot'], selection['snapshot']):
        benchmark.verify_snapshot(snapshot)
    hashes = selection['input_sha256']
    required = [selection['valid_file'], selection['reference']] + [selection['tokenizer_prefix'] + suffix
                for suffix in ('.vocab.json', '.merges.json', '.config.json')]
    if any(path not in hashes for path in required):
        raise ValueError('Missing required input/reference fingerprints')
    if any(benchmark.common.sha256(path) != digest for path, digest in hashes.items()):
        raise ValueError('Input/reference fingerprints changed')
    if Path(selection['reference']).stat().st_size != selection['expected_tokens'] * 4:
        raise ValueError('Reference token size changed')
    rows, schedule = read('results.json'), read('schedule.json')
    integration = validate_results(out, manifest, selection, rows, schedule)
    recovered = {**selection, 'completed': True, 'ready_for_plain_confirmation': True, 'integration': integration}
    recovered.pop('error', None)
    # Preserve all failed-run artifacts; the new report explicitly records its provenance.
    files = ('manifest.json', 'selection.json', 'failure.json', 'integrity.json', 'results.json', 'schedule.json')
    recovered['input_sha256'] = {**hashes, **{str(out / name): benchmark.common.sha256(out / name) for name in files}}
    recovered['recovery'] = {'timestamp_utc': datetime.now(UTC).isoformat(),
                             'reason': failure['error'], 'logging_only_source_corrections': corrected,
                             'timings_recomputed': False, 'wandb_publication_repaired': False,
                             'original_results_directory': str(out)}
    target = out / 'recovered-selection.json'
    if target.exists():
        raise FileExistsError(f'{target} already exists; original artifacts were not overwritten')
    benchmark.common.save_json(target, recovered)
    print(f'Recovered validated results without re-encoding: {target}', flush=True)
    print('Original W&B publication remains failed; use this report for a new independent plain confirmation.', flush=True)
    return recovered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', default='outputs/task1-encode-transport')
    args = parser.parse_args()
    recover(args.results_dir)


if __name__ == '__main__':
    main()
