"""Independent full-validation cache experiment; never trains or changes BPE."""
from __future__ import annotations

import argparse
from array import array
from collections import OrderedDict
from contextlib import nullcontext
from functools import partial
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import resource
import statistics
import subprocess
import sys
import tempfile
import time

from bpe import BPE, CACHE_CAPACITY

ROOT = Path(__file__).resolve().parent
BLOCK = 262144


class LRUCache(OrderedDict):
    def get(self, key, default=None):
        value = super().get(key, default)
        if key in self:
            self.move_to_end(key)
        return value

    def clear(self):
        # BPE calls clear immediately before admission at capacity.
        self.popitem(last=False)


class FreezeCache(dict):
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity

    def clear(self):
        pass

    def __setitem__(self, key, value):
        if len(self) < self.capacity or key in self:
            super().__setitem__(key, value)


class Counters:
    """Only diagnostic runs pay for counters."""
    def get(self, key, default=None):
        self.stats['queries'] += 1
        value = super().get(key, default)
        self.stats['hits' if value is not None else 'misses'] += 1
        return value

    def clear(self):
        if self.policy == 'clear':
            self.stats['clears'] += 1
        elif self.policy == 'lru':
            self.stats['evictions'] += 1
        super().clear()

    def __setitem__(self, key, value):
        if self.policy == 'freeze' and len(self) >= self.capacity and key not in self:
            self.stats['rejected_insertions'] += 1
        else:
            self.stats['insertions'] += 1
        super().__setitem__(key, value)


def make_cache(policy, capacity, diagnostic=False):
    cls = {'clear': dict, 'lru': LRUCache, 'freeze': FreezeCache}[policy]
    if diagnostic:
        cls = type('DiagnosticCache', (Counters, cls), {})
    cache = cls(capacity) if policy == 'freeze' else cls()
    if diagnostic:
        cache.policy = policy
        cache.capacity = capacity
        cache.stats = dict.fromkeys(
            ['queries', 'hits', 'misses', 'insertions', 'clears', 'evictions', 'rejected_insertions'], 0
        )
    return cache


class ExperimentBPE(BPE):
    def configure(self, policy, capacity, diagnostic=False):
        self._cache = make_cache(policy, capacity, diagnostic)
        # Reuse the unchanged production merge/admission implementation.
        self._encode_uncached = partial(BPE._encode_uncached, self, cache_capacity=capacity)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def token_block(ids):
    values = array('I', ids)
    if values.itemsize != 4:
        raise RuntimeError('Requires 32-bit unsigned array elements')
    if sys.byteorder != 'little':
        values.byteswap()
    return values.tobytes()


def check_ids(ids, reference, write=False):
    with open(reference, 'wb' if write else 'rb') as f:
        for start in range(0, len(ids), BLOCK):
            block = token_block(ids[start:start + BLOCK])
            if write:
                f.write(block)
            elif f.read(len(block)) != block:
                raise ValueError(f'Token IDs differ in block starting at {start}')
        if not write and f.read(1):
            raise ValueError('Reference contains additional token IDs')


def pin_cpu(cpu):
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, {cpu})
        if os.sched_getaffinity(0) != {cpu}:
            raise RuntimeError('Could not enforce single-CPU affinity')
    elif cpu is not None:
        raise RuntimeError('CPU affinity requires Linux; omit --cpu for local smoke tests')


def worker(args):
    pin_cpu(args.cpu)
    config = {
        'purpose': 'cache exploration, not final submission', 'policy': args.policy,
        'capacity': args.capacity, 'diagnostic': args.diagnostic, 'cpu': args.cpu,
        'cpu_core_limit': 1, 'max_piece_characters': 128,
        'tokenizer_prefix': str(Path(args.tokenizer_prefix).resolve()),
        'valid_file': str(Path(args.valid_file).resolve()),
        'reference': args.reference_run, 'repeat': args.repeat,
    }
    tokenizer = BPE() if args.reference_run else ExperimentBPE()
    tokenizer.load(args.tokenizer_prefix)
    if not args.reference_run:
        tokenizer.configure(args.policy, args.capacity, args.diagnostic)
    if len(tokenizer.vocab) > 32000:
        raise ValueError('Vocabulary exceeds course limit of 32,000')
    config['special_tokens'] = tokenizer.special_tokens
    config['vocab_size'] = len(tokenizer.vocab)
    with open(args.valid_file, encoding='utf-8', newline='') as f:
        text = f.read()
    if not args.no_wandb:
        from wandb_record import experiment
        context = experiment('bpe-cache-exploration', config, root=ROOT)
    else:
        context = nullcontext(None)
    with context as run:
        if tokenizer._cache:
            raise RuntimeError('Timing must start with an empty cache')
        t0 = time.perf_counter()
        ids = tokenizer._encode_serial(text)
        t1 = time.perf_counter()
        decoded = b"".join(map(tokenizer.vocab.__getitem__, ids)).decode("utf-8", errors="replace")
        t2 = time.perf_counter()
        passed = decoded == text
        t3 = time.perf_counter()
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result = {
            **config, 'encode_seconds': t1 - t0, 'decode_seconds': t2 - t1,
            'compare_seconds': t3 - t2, 'total_seconds': t3 - t0,
            'token_count': len(ids), 'roundtrip_passed': passed,
            'cache_entries': len(tokenizer._cache),
            'peak_rss_bytes': peak if sys.platform == 'darwin' else peak * 1024,
            'peak_rss_scope': 'whole worker, including imports and W&B',
            'python': sys.version, 'cpu_affinity': sorted(os.sched_getaffinity(0))
            if hasattr(os, 'sched_getaffinity') else None,
        }
        if args.diagnostic:
            stats = dict(tokenizer._cache.stats)
            stats['hit_rate'] = stats['hits'] / stats['queries'] if stats['queries'] else 0
            result['diagnostics'] = stats
        try:
            if not passed:
                raise ValueError('Full text roundtrip failed')
            check_ids(ids, args.reference_file, write=args.reference_run)
            result['ids_match_reference'] = True
        except Exception as error:
            result['ids_match_reference'] = False
            result['error'] = str(error)
            write_json(args.result_file, result)
            if run:
                run.summary.update(result)
            raise
        if run:
            import wandb
            raw = text.encode('utf-8')
            rows, offset = [], 0
            for i, token_id in enumerate(ids[:200]):
                piece = tokenizer.vocab[token_id]
                end = offset + len(piece)
                rows.append([i, token_id, piece.hex(), repr(piece),
                             repr(tokenizer.decode([token_id])), offset, end, piece == raw[offset:end]])
                offset = end
            run.log({'bpe/first_200_tokens': wandb.Table(
                columns=['position', 'token_id', 'token_hex', 'token_bytes', 'decoded_token',
                         'byte_start', 'byte_end', 'matches_input_bytes'], data=rows)})
            run.summary.update(result)
            run.summary.update({'bpe/valid_seconds': result['total_seconds'],
                                'bpe/roundtrip_passed': passed, 'bpe/valid_token_count': len(ids)})
            result['wandb_url'] = run.url
        print('First 200 IDs:', ids[:200], flush=True)
        print('First 200 decoded:', repr(tokenizer.decode(ids[:200])), flush=True)
        write_json(args.result_file, result)
        print(json.dumps(result, ensure_ascii=False), flush=True)


def summarize(out, results, reference):
    rows = []
    for policy, capacity in sorted({(r['policy'], r['capacity']) for r in results}):
        group = [r for r in results if (r['policy'], r['capacity']) == (policy, capacity)]
        times = [r['total_seconds'] for r in group]
        median = statistics.median(times)
        rows.append({'policy': policy, 'capacity': capacity, 'runs': len(times),
                     'median_seconds': median, 'min_seconds': min(times), 'max_seconds': max(times),
                     'speedup_vs_original': reference['total_seconds'] / median})
    with open(out / 'summary.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    def table(items):
        lines = ['| 策略 | 容量 | 中位数(s) | 最小(s) | 最大(s) | 相对原版速度比 |',
                 '|---|---:|---:|---:|---:|---:|']
        lines += [f"| {r['policy']} | {r['capacity']} | {r['median_seconds']:.3f} | "
                  f"{r['min_seconds']:.3f} | {r['max_seconds']:.3f} | {r['speedup_vs_original']:.3f} |"
                  for r in items]
        return '\n'.join(lines)
    report = ['# BPE 缓存探索结果',
              f"原版 {CACHE_CAPACITY:,} 清空策略单次参考耗时：{reference['total_seconds']:.3f}s。",
              '速度比使用单次原版参考；候选使用多轮中位数。诊断不参与排名。',
              '所有列出的候选均通过完整 ID 精确比较与全文还原。',
              '固定容量比较策略：']
    for capacity in sorted({r['capacity'] for r in rows}):
        report += [f'## 容量 {capacity:,}', table([r for r in rows if r['capacity'] == capacity])]
    report += ['固定策略比较容量：']
    for policy in sorted({r['policy'] for r in rows}):
        report += [f'## {policy}', table([r for r in rows if r['policy'] == policy])]
    if (out / 'diagnostics.json').exists():
        report += ['## 独立诊断', '| 策略 | 容量 | 命中率 | 插入 | 清空 | 淘汰 | 拒绝新增 | 最终项数 | 峰值RSS(MiB) |',
                   '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in json.loads((out / 'diagnostics.json').read_text()):
            s = r['diagnostics']
            report.append(f"| {r['policy']} | {r['capacity']} | {s['hit_rate']:.2%} | {s['insertions']} | "
                          f"{s['clears']} | {s['evictions']} | {s['rejected_insertions']} | "
                          f"{r['cache_entries']} | {r['peak_rss_bytes'] / 2**20:.1f} |")
    report += ['策略差异包含维护缓存所需的开销；结合命中率解释，不将诊断时间当作正常测速。',
               '本报告不自动更改正式 BPE 配置，也不证明既有词表的训练来源或核心代码来源合规。']
    (out / 'report.md').write_text('\n\n'.join(report) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer-prefix', required=True)
    parser.add_argument('--valid-file', default='/data/assignment1-data/owt_valid.txt')
    parser.add_argument('--capacities', type=int, nargs='+', default=[65536, 262144, 1048576])
    parser.add_argument('--policies', choices=['clear', 'lru', 'freeze'], nargs='+', default=['clear', 'lru', 'freeze'])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20261012)
    parser.add_argument('--cpu', type=int)
    parser.add_argument('--diagnostics', action='store_true')
    parser.add_argument('--output-dir', default='outputs/bpe-cache-exploration')
    parser.add_argument('--no-wandb', action='store_true', help='Local development only; not course experiment evidence')
    # Internal child-process arguments.
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--policy', default='clear', help=argparse.SUPPRESS)
    parser.add_argument('--capacity', type=int, default=CACHE_CAPACITY, help=argparse.SUPPRESS)
    parser.add_argument('--diagnostic', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--reference-run', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--repeat', type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument('--reference-file', help=argparse.SUPPRESS)
    parser.add_argument('--result-file', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.capacities) <= 0 or args.repeats <= 0 or args.capacity <= 0:
        parser.error('Capacities and repeats must be positive')
    if args.worker:
        worker(args)
        return
    if hasattr(os, 'sched_getaffinity'):
        available = os.sched_getaffinity(0)
        if args.cpu is None:
            args.cpu = min(available)
        if args.cpu not in available:
            parser.error(f'CPU must be one of {sorted(available)}')
    elif not args.no_wandb:
        parser.error('Cloud measurements require Linux CPU affinity')
    pin_cpu(args.cpu)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = [Path(args.valid_file).resolve()] + [
        Path(args.tokenizer_prefix + suffix).resolve() for suffix in ['.vocab.json', '.merges.json', '.config.json']]
    sources = [ROOT / 'bpe.py', Path(__file__).resolve(), ROOT / 'wandb_record.py']
    manifest = {'python': sys.version, 'platform': platform.platform(), 'cpu': args.cpu,
                'cpu_core_limit': 1, 'seed': args.seed, 'arguments': vars(args),
                'sha256': {str(p): digest(p) for p in inputs + sources}}
    combinations = [(p, c) for p in dict.fromkeys(args.policies) for c in dict.fromkeys(args.capacities)]
    rng = random.Random(args.seed)
    schedule = []
    for repeat in range(1, args.repeats + 1):
        order = combinations.copy()
        rng.shuffle(order)
        schedule.extend({'policy': p, 'capacity': c, 'repeat': repeat} for p, c in order)
    manifest['schedule'] = schedule
    write_json(out / 'manifest.json', manifest)
    env = os.environ.copy()
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS']:
        env[name] = '1'
    def launch(spec, name, reference_file, reference=False, diagnostic=False):
        result_file = out / f'{name}.json'
        cmd = [sys.executable, str(Path(__file__).resolve()), '--worker',
               '--tokenizer-prefix', str(Path(args.tokenizer_prefix).resolve()), '--valid-file', str(inputs[0]),
               '--policy', spec['policy'], '--capacity', str(spec['capacity']), '--repeat', str(spec['repeat']),
               '--reference-file', str(reference_file), '--result-file', str(result_file)]
        if args.cpu is not None:
            cmd += ['--cpu', str(args.cpu)]
        if args.no_wandb:
            cmd += ['--no-wandb']
        if reference:
            cmd += ['--reference-run']
        if diagnostic:
            cmd += ['--diagnostic']
        print(f'Running {name}; log: {out / (name + ".log")}', flush=True)
        with open(out / f'{name}.log', 'w', encoding='utf-8') as log:
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=ROOT)
        return json.loads(result_file.read_text(encoding='utf-8'))
    with tempfile.TemporaryDirectory(prefix='reference-', dir=out) as temp:
        reference_file = Path(temp) / 'ids.u32le'
        reference = launch({'policy': 'clear', 'capacity': CACHE_CAPACITY, 'repeat': 0},
                           'reference', reference_file, reference=True)
        results = []
        for i, spec in enumerate(schedule, 1):
            results.append(launch(spec, f'run-{i:02d}', reference_file))
            write_json(out / 'results.json', results)
            summarize(out, results, reference)
        if args.diagnostics:
            diagnostics = []
            for i, (policy, capacity) in enumerate(combinations, 1):
                diagnostics.append(launch({'policy': policy, 'capacity': capacity, 'repeat': 0},
                                          f'diagnostic-{i:02d}', reference_file, diagnostic=True))
                write_json(out / 'diagnostics.json', diagnostics)
        summarize(out, results, reference)
    changed = [str(p) for p in inputs + sources if digest(p) != manifest['sha256'][str(p)]]
    write_json(out / 'integrity.json', {'changed_files': changed})
    if changed:
        raise RuntimeError(f'Inputs or source changed during experiment: {changed}')
    print(f'Completed. Report: {out / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
