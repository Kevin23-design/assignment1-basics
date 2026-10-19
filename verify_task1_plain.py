"""Standalone ordinary-BPE verifier. No experiment backend patching and no training."""
from __future__ import annotations

import argparse
from array import array
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def check_sources(snapshot):
    root = Path(snapshot['directory'])
    if json.loads((root / 'snapshot.json').read_text()) != snapshot:
        raise ValueError('Plain candidate snapshot metadata changed')
    actual = {p.name: sha(p) for p in root.iterdir()
              if p.is_file() and p.suffix in ('.py', '.json', '.pyx', '.pxd', '.c', '.h', '.so', '.pyd') and p.name != 'snapshot.json'}
    if actual != snapshot['source_sha256']:
        raise ValueError('Plain candidate source changed')


def check_inputs(job):
    changed = [path for path, value in job['input_sha256'].items() if sha(path) != value]
    if changed:
        raise ValueError(f'Plain confirmation inputs changed: {changed}')


def compare_reference(ids, path):
    if Path(path).stat().st_size != len(ids) * 4:
        raise ValueError('Reference token count differs')
    offset = 0
    with open(path, 'rb') as handle:
        while raw := handle.read(262144 * 4):
            if len(raw) % 4:
                raise ValueError('Malformed reference')
            block = array('I')
            block.frombytes(raw)
            if sys.byteorder != 'little':
                block.byteswap()
            if block.itemsize != 4 or any(value != ids[offset + i] for i, value in enumerate(block)):
                raise ValueError('Full reference ID comparison failed')
            offset += len(block)


def execute(job, result_path):
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if job['cpus']:
        if not set(job['cpus']) <= os.sched_getaffinity(0):
            raise ValueError('Requested CPUs outside inherited allocation')
        os.sched_setaffinity(0, set(job['cpus']))
    elif not job['development']:
        raise RuntimeError('Formal confirmation needs Linux affinity')
    snapshot = job['snapshot']
    check_sources(snapshot)
    check_inputs(job)
    result = {'status': 'failed', 'patching': False, 'method': snapshot['method'],
              'active_wandb': bool(job.get('active_wandb')), 'cpu_affinity': job['cpus'],
              'timing_scope': 'ordinary-public-interface', 'diagnostic': False}
    run = None
    try:
        import wandb
        root = Path(snapshot['directory']).resolve()
        sys.path.insert(0, str(root))
        if 'bpe' in sys.modules:
            raise RuntimeError('Plain interpreter already loaded a BPE implementation')
        if job.get('active_wandb'):
            from wandb_record import experiment
            context = experiment('bpe-task1-plain-confirmation', job, root=root)
        else:
            context = nullcontext(None)
        with context as run:
            if job.get('active_wandb') and (run is None or wandb.run is None):
                raise RuntimeError('Confirmation requires a live W&B run')
            from bpe import BPE
            import bpe
            if Path(bpe.__file__).resolve() != root / 'bpe.py':
                raise RuntimeError('Ordinary BPE imported from the wrong source directory')
            tokenizer = BPE()
            tokenizer.load(job['tokenizer_prefix'])
            with open(job['valid_file'], encoding='utf-8', newline='') as handle:
                text = handle.read()
            if tokenizer._cache:
                raise RuntimeError('Plain confirmation must start with an empty cache')
            before = set(sys.modules)
            started = time.perf_counter()
            ids = tokenizer.encode(text)
            encoded = time.perf_counter()
            decoded = tokenizer.decode(ids)
            decoded_at = time.perf_counter()
            passed = decoded == text
            ended = time.perf_counter()
            if not isinstance(ids, list) or len(ids) != job['expected_tokens'] or not passed:
                raise ValueError('Plain BPE failed token count or exact roundtrip')
            if snapshot.get('native_build'):
                metrics = getattr(tokenizer, '_native_decode_metrics', None)
                result['native_decode'] = metrics
                if snapshot['method'] != 'B0' and (not metrics or metrics.get('backend') != 'native'):
                    raise ValueError('Formal candidate silently bypassed native decoding')
                result['native_extension_loaded_during'] = 'task1_native_runtime' not in before
                worker_metrics = getattr(tokenizer, '_task1_worker_metrics', [])
                result['worker_metadata'] = worker_metrics
                if not worker_metrics:
                    raise ValueError('Native-suite candidate bypassed direct workers')
                if snapshot['method'] == 'CY-ED' and any(not w.get('native_encoding') or
                        w['native_encoding']['fallback_calls'] or not w['native_encoding']['native_calls']
                        for w in worker_metrics):
                    raise ValueError('Candidate silently bypassed native encoding')
            if snapshot.get('assembly_build'):
                assembly = getattr(tokenizer, '_task1_assembly_metrics', None)
                if (snapshot.get('assembly_diagnostic') or not assembly
                        or assembly.get('method') != snapshot['assembly_method']
                        or assembly.get('converted_ids') != len(ids)):
                    raise ValueError('Ordinary candidate bypassed its normal assembly backend')
                result['assembly'] = assembly
                if assembly['method'] != 'M0':
                    expected = Path(snapshot['directory']) / snapshot['assembly_build']['binary']
                    if Path(assembly['extension_path']).resolve() != expected.resolve():
                        raise ValueError('Assembly extension loaded from another snapshot')
                expected = Path(snapshot['directory']) / snapshot['native_build']['binary']
                if Path(result['native_decode']['extension_path']).resolve() != expected.resolve():
                    raise ValueError('Ordinary candidate replaced the native decoder')
            compare_reference(ids, job['reference'])
            check_sources(snapshot)
            check_inputs(job)
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result.update(status='ok', total_seconds=ended - started, encode_seconds=encoded - started,
                          decode_seconds=decoded_at - encoded, compare_seconds=ended - decoded_at,
                          token_count=len(ids), returned_type=type(ids).__name__, ids_match_reference=True,
                          roundtrip_passed=True, parent_peak_rss_bytes=int(peak if sys.platform == 'darwin' else peak * 1024),
                          modules_before=sorted(before), modules_loaded_during=sorted(set(sys.modules) - before),
                          bpe_source=str(root / 'bpe.py'), source_changed=False,
                          wandb_mode=os.environ.get('WANDB_MODE', 'online'))
            if run is not None:
                columns = ['position', 'token_id', 'token_hex', 'token_bytes', 'decoded_token',
                           'standalone_utf8_valid', 'byte_start', 'byte_end', 'matches_input_bytes']
                rows, offset, raw = [], 0, text.encode('utf-8')
                for i, token_id in enumerate(ids[:200]):
                    piece = tokenizer.vocab[token_id]
                    end = offset + len(piece)
                    try:
                        piece.decode('utf-8', errors='strict')
                        valid = True
                    except UnicodeDecodeError:
                        valid = False
                    rows.append([i, token_id, piece.hex(), repr(piece), repr(tokenizer.decode([token_id])),
                                 valid, offset, end, piece == raw[offset:end]])
                    offset = end
                print('First 200 IDs:', ids[:200], flush=True)
                print('First 200 decoded:', repr(tokenizer.decode(ids[:200])), flush=True)
                run.log({'bpe/first_200_tokens': wandb.Table(columns=columns, data=rows)})
                run.summary.update({'bpe/valid_seconds': result['total_seconds'], 'bpe/valid_token_count': len(ids),
                                    'bpe/roundtrip_passed': True, 'patching': False, 'training': False,
                                    'submission_provenance_verified': False,
                                    'time/encode_seconds': result['encode_seconds'],
                                    'time/decode_seconds': result['decode_seconds'],
                                    'time/compare_seconds': result['compare_seconds']})
                result.update(wandb_url=run.url, run_id=run.id)
                Path(result_path).write_text(json.dumps(result, ensure_ascii=False, indent=2))
                run.save(str(result_path), base_path=str(Path(result_path).parent), policy='now')
        check_sources(snapshot)
        check_inputs(job)
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        Path(result_path).write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Plain BPE confirmation interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    parser.add_argument('--result', required=True)
    args = parser.parse_args()
    execute(json.loads(Path(args.job).read_text()), args.result)
