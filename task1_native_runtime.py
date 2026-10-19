"""Lazy experiment-only native adapters. Never compiles or caches source text."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import sysconfig
import time

_DIRECTORY = Path(__file__).resolve().parent
_loaded = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_build(directory, full=True):
    directory = Path(directory).resolve()
    path = directory / ('native_build.json' if (directory / 'native_build.json').exists() else 'build.json')
    if not path.is_file():
        raise RuntimeError('Native build metadata missing; explicitly run build_task1_native.py first')
    metadata = json.loads(path.read_text())
    for key, expected in {'implementation': sys.implementation.name, 'cache_tag': sys.implementation.cache_tag,
                          'soabi': sysconfig.get_config_var('SOABI'), 'machine': platform.machine()}.items():
        if metadata.get(key) != expected:
            raise RuntimeError(f'Native build ABI mismatch: {key}; rebuild with this experiment Python')
    files = metadata['files'] if full else {metadata['binary']: metadata['files'][metadata['binary']]}
    for name, digest in files.items():
        target = (directory / name).resolve()
        if not target.is_relative_to(directory) or not target.is_file() or sha(target) != digest:
            raise RuntimeError(f'Native build file missing or changed: {name}')
    return metadata


def load_native():
    global _loaded
    metadata = verify_build(_DIRECTORY, full=False) if _loaded is None else None
    if _loaded is None:
        path = _DIRECTORY / metadata['binary']
        spec = importlib.util.spec_from_file_location('_task1_native', path)
        if spec is None or spec.loader is None:
            raise ImportError(f'Cannot load native extension: {path}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _loaded = module
    metadata_path = Path(_loaded.__file__).resolve()
    if metadata_path.parent != _DIRECTORY.resolve():
        raise RuntimeError('Native extension loaded from the wrong snapshot')
    return _loaded


def decode(tokenizer, ids):
    metrics = {}
    native = load_native()
    result = native.decode(tokenizer.vocab, ids, metrics)
    metrics['extension_path'] = str(Path(native.__file__).resolve())
    tokenizer._native_decode_metrics = metrics
    return result


def encode_uncached(tokenizer, piece, cache_capacity):
    native = load_native()
    metrics = getattr(tokenizer, '_native_encode_metrics', None)
    if metrics is None:
        metrics = {'native_calls': 0, 'fallback_calls': 0, 'table_build_seconds': 0.}
        tokenizer._native_encode_metrics = metrics
    state = getattr(tokenizer, '_native_encoder_state', None)
    if state is None:
        began = time.perf_counter()
        try:
            state = native.Encoder(tokenizer._ranks, tokenizer._ids, len(tokenizer.merges))
        except native.UnsupportedEncoding:
            state = False
        metrics['table_build_seconds'] += time.perf_counter() - began
        tokenizer._native_encoder_state = state
    if state is False:
        metrics['fallback_calls'] += 1
        return tokenizer._python_encode_uncached(piece, cache_capacity)
    ids = state.encode(piece.encode('utf-8'))
    metrics['native_calls'] += 1
    if len(piece) <= 128:
        if len(tokenizer._cache) >= cache_capacity:
            tokenizer._cache.clear()
        tokenizer._cache[piece] = ids
    return ids
