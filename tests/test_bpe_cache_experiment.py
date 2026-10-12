"""Small cache and orchestration checks, never full OWT benchmarks."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from bpe import BPE
from benchmark_bpe_cache import ExperimentBPE, check_ids, make_cache


@pytest.mark.parametrize('diagnostic', [False, True])
@pytest.mark.parametrize('policy', ['clear', 'lru', 'freeze'])
def test_capacity_and_policy(policy, diagnostic):
    cache = make_cache(policy, 2, diagnostic)
    cache['a'] = (1,)
    cache['b'] = (2,)
    assert cache.get('a') == (1,)
    # Follow the exact admission contract used by BPE._encode_uncached.
    if len(cache) >= 2:
        cache.clear()
    cache['c'] = (3,)
    assert len(cache) <= 2
    assert cache.get('missing') is None
    if policy == 'clear':
        assert set(cache) == {'c'}
    elif policy == 'lru':
        assert set(cache) == {'a', 'c'}
    else:
        assert set(cache) == {'a', 'b'}
        assert cache.get('a') == (1,)
    if diagnostic:
        assert cache.stats['hits'] == (2 if policy == 'freeze' else 1)
        assert cache.stats['misses'] == 1
        assert cache.stats['insertions'] == (2 if policy == 'freeze' else 3)
        assert cache.stats['clears'] == int(policy == 'clear')
        assert cache.stats['evictions'] == int(policy == 'lru')
        assert cache.stats['rejected_insertions'] == int(policy == 'freeze')


def tokenizer(cls=BPE):
    vocab = dict(enumerate(bytes([i]) for i in range(256)))
    vocab[256] = b'ab'
    return cls(vocab, [(b'a', b'b')], ['<x>', '<x>long'])


@pytest.mark.parametrize('policy', ['clear', 'lru', 'freeze'])
def test_encoding_save_load_and_length_limit(tmp_path, policy):
    original = tokenizer()
    prefix = str(tmp_path / 'tokenizer')
    original.save(prefix)
    candidate = ExperimentBPE()
    candidate.load(prefix)
    candidate.configure(policy, 2, diagnostic=True)
    text = "ab cd ab ef ab\n\n中文🙂 <x>long <x> " + 'z' * 129
    ids = candidate.encode(text)
    assert ids == original.encode(text)
    assert candidate.decode(ids) == text
    assert len(candidate._cache) <= 2
    assert all(len(piece) <= 128 for piece in candidate._cache)
    assert candidate._cache.stats['queries'] > 0


def test_reference_detects_value_and_length_mismatch(tmp_path):
    path = tmp_path / 'ids'
    check_ids([1, 2, 3], path, write=True)
    check_ids([1, 2, 3], path)
    for ids in ([1, 4, 3], [1, 2], [1, 2, 3, 4]):
        with pytest.raises(ValueError):
            check_ids(ids, path)


def test_small_end_to_end(tmp_path):
    prefix = str(tmp_path / 'tokenizer')
    tokenizer().save(prefix)
    valid = tmp_path / 'valid.txt'
    valid.write_text(('ab cd ab ef\n中文🙂 <x>long <x>\n' * 10), encoding='utf-8')
    output = tmp_path / 'experiment'
    script = Path(__file__).resolve().parents[1] / 'benchmark_bpe_cache.py'
    subprocess.run([
        sys.executable, str(script), '--tokenizer-prefix', prefix, '--valid-file', str(valid),
        '--capacities', '2', '4', '--repeats', '2', '--diagnostics', '--no-wandb',
        '--output-dir', str(output),
    ], check=True, capture_output=True, text=True)
    results = json.loads((output / 'results.json').read_text())
    diagnostics = json.loads((output / 'diagnostics.json').read_text())
    assert len(results) == 12
    assert len(diagnostics) == 6
    assert all(r['roundtrip_passed'] and r['ids_match_reference'] for r in results + diagnostics)
    assert len(json.loads((output / 'manifest.json').read_text())['schedule']) == 12
    assert json.loads((output / 'integrity.json').read_text())['changed_files'] == []
    assert '独立诊断' in (output / 'report.md').read_text()
    assert not list(output.glob('reference-*'))
