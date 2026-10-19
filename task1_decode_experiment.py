"""Exploration adapters and immutable plain-source packaging; never edits production files."""
from __future__ import annotations

import ast
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import statistics

from bpe import BPE
from bpe_decode_runtime import METHODS, decode_ids
import benchmark_task1_efficiency as common
from task1_pipeline_runtime import selected_pipeline

ROOT = Path(__file__).resolve().parent


@contextmanager
def selected_decoder(method, diagnostic=False, development_workers=None):
    if method not in METHODS:
        raise ValueError('Unknown decoder')
    with selected_pipeline('D3', 'C2', diagnostic, development_workers) as (stats, calls):
        original = BPE.decode
        try:
            if method != 'R0':
                BPE.decode = lambda tokenizer, ids: decode_ids(
                    tokenizer.vocab, ids, method, stats.setdefault('decoder', {}) if diagnostic else None,
                    max_workers=development_workers or 16, development=development_workers is not None)
            yield stats, calls
        finally:
            BPE.decode = original


def replace_function(source, name, replacement, class_name=None):
    tree = ast.parse(source)
    scope = tree.body if class_name is None else next(n for n in tree.body if isinstance(n, ast.ClassDef)
                                                     and n.name == class_name).body
    node = next(n for n in scope if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = source.splitlines(keepends=True)
    return ''.join(lines[:node.lineno - 1]) + replacement + ''.join(lines[node.end_lineno:])


def source_hashes(directory):
    directory = Path(directory)
    return {path.relative_to(directory).as_posix(): common.sha256(path)
            for path in sorted(directory.iterdir()) if path.is_file() and path.suffix in ('.py', '.json')
            and path.name != 'snapshot.json'}


def make_snapshot(directory, method, workers=16, development=False):
    if method not in METHODS or not 1 <= workers <= 16 or (not development and workers != 16):
        raise ValueError('Invalid plain candidate configuration')
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    names = ('bpe.py', 'bpe_memory_runtime.py', 'parallel_bpe_encoding.py', 'task1_transport_runtime.py',
             'task1_pipeline_runtime.py', 'bpe_decode_runtime.py', 'wandb_record.py', 'train_bpe.py',
             'verify_task1_plain.py')
    for name in names:
        shutil.copyfile(ROOT / name, directory / name)
    source = (directory / 'bpe.py').read_text()
    updated = replace_function(source, 'decode', f'''    def decode(self, ids: list[int]) -> str:
        from bpe_decode_runtime import decode_ids
        return decode_ids(self.vocab, ids, {method!r}, max_workers={workers}, development={development!r})
''', 'BPE')
    if development:
        updated = updated.replace('_ENCODE_WORKERS = 16', f'_ENCODE_WORKERS = {workers}')
        updated = updated.replace('_PARALLEL_MIN_CHARACTERS = 8 << 20', '_PARALLEL_MIN_CHARACTERS = 1')
    # Ensure every core algorithm/interface except the decode adapter is byte-for-AST unchanged.
    before = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'BPE')
    after = next(n for n in ast.parse(updated).body if isinstance(n, ast.ClassDef) and n.name == 'BPE')
    assert [ast.dump(n) for n in before.body if getattr(n, 'name', None) != 'decode'] == [
        ast.dump(n) for n in after.body if getattr(n, 'name', None) != 'decode']
    (directory / 'bpe.py').write_text(updated)
    path = directory / 'bpe_memory_runtime.py'
    path.write_text(replace_function(path.read_text(), '_launch', '''def _launch(tokenizer, text, ranges, cpus, workers):
    from task1_pipeline_runtime import launch_all_first
    return launch_all_first(tokenizer, text, ranges, cpus, workers, False, {})
'''))
    if development:
        path.write_text(replace_function(path.read_text(), 'encode_if_parallel', '''def encode_if_parallel(tokenizer, text, workers, multiplier):
    ranges = _partition_ranges(tokenizer, text, workers * multiplier)
    if ranges is None:
        return None
    cpus = sorted(os.sched_getaffinity(0))[:workers] if hasattr(os, 'sched_getaffinity') else []
    return _launch(tokenizer, text, ranges, cpus, workers)
'''))
    snapshot = {'directory': str(directory), 'method': method, 'workers': workers, 'development': development,
                'source_sha256': source_hashes(directory), 'patching': False,
                'api': 'ordinary from bpe import BPE; no selected_backend/selected_pipeline context'}
    common.save_json(directory / 'snapshot.json', snapshot)
    return snapshot


def verify_snapshot(snapshot):
    if json.loads((Path(snapshot['directory']) / 'snapshot.json').read_text()) != snapshot:
        raise ValueError('Plain candidate snapshot metadata changed')
    if snapshot['source_sha256'] != source_hashes(snapshot['directory']):
        raise ValueError('Plain candidate source changed')


def integration_decision(experimental, plain, repeats):
    for rows in (experimental, plain):
        if len(rows) != repeats or {r['repeat'] for r in rows} != set(range(1, repeats + 1)):
            raise ValueError('Missing or duplicate integration pair')
        if any(r['status'] != 'ok' or not r.get('roundtrip_passed') or not r.get('ids_match_reference')
               or r.get('diagnostic') for r in rows):
            raise ValueError('Integration requires exact successful normal trials')
    reference = {r['repeat']: r for r in experimental}
    normal = {r['repeat']: r for r in plain}
    before = statistics.median(r['total_seconds'] for r in experimental)
    after = statistics.median(r['total_seconds'] for r in plain)
    acceptable = sum(normal[i]['total_seconds'] <= reference[i]['total_seconds'] * 1.05 for i in reference)
    required = (4 * repeats + 4) // 5
    return {'passed': after <= before * 1.03 and acceptable >= required,
            'experimental_median_seconds': before, 'plain_median_seconds': after,
            'median_ratio': after / before, 'pairs_within_105_percent': acceptable,
            'required_pairs': required, 'median_limit': 1.03, 'pair_limit': 1.05}


def previous_inputs(directory, hashes):
    directory = Path(directory).resolve()
    prior = json.loads((directory / 'manifest.json').read_text())
    selected = json.loads((directory / 'selection.json').read_text())
    if (not selected.get('completed') or selected.get('recommended') != 'D3'
            or selected.get('recommended_decoder') != 'C2'):
        raise ValueError('Previous pipeline must recommend completed D3/C2')
    for key, value in {'workers': 16, 'parts': 256, 'cache_capacity': 1048576,
                       'decode_block_size': 131072}.items():
        if prior.get(key) != value:
            raise ValueError('Previous resource configuration differs')
    names = [Path(prior['arguments']['valid_file']).name] + [
        Path(prior['arguments']['tokenizer_prefix']).name + s
        for s in ('.vocab.json', '.merges.json', '.config.json')]
    for name, digest in zip(names, hashes.values(), strict=True):
        old = [v for p, v in prior['input_sha256'].items() if Path(p).name == name]
        if len(old) != 1 or old[0] != digest:
            raise ValueError(f'Input/tokenizer differs from previous pipeline: {name}')
    return {'confirmation_seconds': selected['confirmation_seconds'], 'directory': str(directory),
            'note': 'Historical reference only; all rankings use fresh R0 trials'}
