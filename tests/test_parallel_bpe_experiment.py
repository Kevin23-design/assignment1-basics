"""Tiny fixtures only: no OWT data, GPU or W&B operations."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from bpe import BPE
from benchmark_parallel_bpe import choose_smallest, choose_workers
from parallel_bpe_encoding import (
    StreamingExperimentBPE, compare_roundtrip, compare_tokens, encode_parallel,
    special_matches, split_ranges, text_chunks, write_uint32,
)


def tokenizer(specials=None, cls=BPE):
    vocab = dict(enumerate(bytes([i]) for i in range(256)))
    vocab[256] = b'ab'
    return cls(vocab, [(b'a', b'b')], specials or ['<|endoftext|>'])


@pytest.mark.parametrize('capacity', [2, 4, 8])
def test_stream_cache_and_default(capacity):
    original = tokenizer()
    candidate = tokenizer(cls=StreamingExperimentBPE)
    candidate.configure(capacity, diagnostic=True)
    text = 'ab cd ef ab\r\n中文🙂  <|endoftext|> ab ' + 'z' * 129
    chunks = [text[i:i + 3] for i in range(0, len(text), 3)]
    ids = list(candidate.encode_iterable(chunks))
    assert ids == original.encode(text)
    assert candidate.decode(ids) == text
    assert len(candidate._cache) <= capacity
    assert all(len(piece) <= 128 for piece in candidate._cache)
    assert candidate._cache.stats['queries'] > 0
    assert len(original._cache) <= 2048


def test_scanner_matches_full_regex_with_overlap(tmp_path):
    bpe = tokenizer(['<x>', '<x>long', 'a<x>', '<y>', '🙂'])
    text = ('a<x> <x>long <x> <y>🙂\r\n' * 7)
    source = tmp_path / 'input'
    source.write_bytes(text.encode())
    expected = [(len(text[:m.start()].encode()), len(text[:m.end()].encode()), m.group().encode())
                for m in bpe._special_re.finditer(text)]
    for size in [1, 2, 3, 8, 64]:
        assert list(special_matches(source, bpe.special_tokens, size)) == expected
    ranges = split_ranges(source, bpe, 3, '<x>', block_size=2)
    assert ranges[0][0] == 0 and ranges[-1][1] == source.stat().st_size
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
    assert ''.join(''.join(text_chunks(source, a, b, 2)) for a, b in ranges) == text


def test_insufficient_and_invalid_split(tmp_path):
    source = tmp_path / 'input'
    source.write_text('中文 ab' * 30, encoding='utf-8')
    assert len(split_ranges(source, tokenizer(), 4, '<|endoftext|>')) == 1
    with pytest.raises(ValueError):
        split_ranges(source, tokenizer(), 2, '<missing>')
    source.write_text('a' * 100 + '<|endoftext|>' + 'b', encoding='utf-8')
    ranges = split_ranges(source, tokenizer(), 4, '<|endoftext|>', block_size=3)
    assert len(ranges) == 2  # Several targets map to one cut; no empty ranges.


def test_selection_thresholds():
    assert choose_smallest({2048: 103, 65536: 100, 1048576: 101}, .03) == 2048
    assert choose_smallest({2048: 103.01, 65536: 100}, .03) == 65536
    assert choose_workers({1: 300, 8: 104, 16: 100}) == 8
    assert choose_workers({8: 110, 16: 100}) == 16
    # Explicit <5% 16-vs-8 improvement rule, even just outside fastest * 1.05.
    assert choose_workers({8: 105.1, 16: 100}) == 8
    with pytest.raises(ValueError):
        choose_smallest({}, .03)


def test_worker_failure_cleanup(tmp_path):
    prefix = str(tmp_path / 'tokenizer')
    tokenizer().save(prefix)
    source = tmp_path / 'bad'
    source.write_bytes(b'bad\xff')
    destination = tmp_path / 'failed.tokens'
    with pytest.raises(RuntimeError):
        encode_parallel(source, prefix, destination, 1, 2, '<|endoftext|>', [])
    assert not destination.exists()
    assert not Path(str(destination) + '.partial').exists()
    assert not Path(str(destination) + '.parts').exists()


def test_uint32_and_roundtrip_mismatch(tmp_path):
    bpe = tokenizer()
    source = tmp_path / 'input'
    source.write_bytes(b'ab')
    encoded = tmp_path / 'encoded'
    with encoded.open('wb') as handle:
        write_uint32(handle, [256])
    compare_tokens(encoded, encoded, 1)
    compare_roundtrip(encoded, source, bpe)
    source.write_bytes(b'ac')
    with pytest.raises(ValueError):
        compare_roundtrip(encoded, source, bpe)
    with pytest.raises(ValueError):
        compare_tokens(encoded, encoded, 2)


def test_tiny_two_phase_workflow(tmp_path):
    bpe = tokenizer()
    prefix = str(tmp_path / 'tokenizer')
    bpe.save(prefix)
    inputs = {}
    for kind, repeats in [('train', 20), ('valid', 10)]:
        text = ('ab\r\n中文🙂 ab cd <|endoftext|> ef\n' * repeats)
        source = tmp_path / (kind + '.txt')
        source.write_bytes(text.encode())
        reference = tmp_path / (kind + '.tokens')
        with reference.open('wb') as handle:
            write_uint32(handle, bpe.encode(text))
        inputs[kind] = source, reference
    script = Path(__file__).resolve().parents[1] / 'benchmark_parallel_bpe.py'
    out = tmp_path / 'validation'
    cmd = [sys.executable, str(script), '--development', '--no-wandb',
           '--tokenizer-prefix', prefix, '--train-file', str(inputs['train'][0]),
           '--valid-file', str(inputs['valid'][0]), '--reference-train-tokens', str(inputs['train'][1]),
           '--reference-valid-tokens', str(inputs['valid'][1]), '--cache-capacities', '2', '4',
           '--workers', '1', '2', '--repeats', '1', '--diagnostics', '--output-dir', str(out)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    selection = json.loads((out / 'selection.json').read_text())
    assert selection['completed'] and len(selection['top_two_workers']) == 2
    rows = json.loads((out / 'results.json').read_text())
    assert len(rows) == 6
    assert all(r['id_comparison_passed'] and r['roundtrip_bytes_passed'] for r in rows)
    assert not list(out.glob('*.tokens'))
    confirm = tmp_path / 'confirmation'
    subprocess.run([sys.executable, str(script), '--phase', 'train-confirmation',
                    '--validation-report', str(out / 'selection.json'), '--output-dir', str(confirm),
                    '--development', '--no-wandb'], check=True, capture_output=True, text=True)
    assert (confirm / 'train.tokens').read_bytes() == inputs['train'][1].read_bytes()
    assert len(list(confirm.glob('*.tokens'))) == 1
    assert json.loads((confirm / 'integrity.json').read_text())['unchanged']
