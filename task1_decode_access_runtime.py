"""Lazy, explicitly built decode extension. No input text or cross-call ID buffers."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import platform
import sys
import sysconfig

from task1_native_runtime import sha

_DIRECTORY = Path(__file__).resolve().parent / 'bpe_native' / 'decode'
_loaded = None


def verify_build(directory, full=True):
    if sys.implementation.name != 'cpython' or sys.version_info[:2] != (3, 13) or sysconfig.get_config_var('Py_GIL_DISABLED'):
        raise RuntimeError('Decode access requires GIL-enabled CPython 3.13')
    directory = Path(directory).resolve()
    path = directory / ('decode_access_build.json' if (directory / 'decode_access_build.json').exists() else 'build.json')
    if not path.is_file():
        raise RuntimeError('Decode access extension missing; explicitly run build_task1_decode_access.py')
    metadata = json.loads(path.read_text())
    for key, value in {'implementation': sys.implementation.name, 'cache_tag': sys.implementation.cache_tag,
                       'soabi': sysconfig.get_config_var('SOABI'), 'machine': platform.machine(), 'gil_disabled': False}.items():
        if metadata.get(key) != value:
            raise RuntimeError(f'Decode access ABI mismatch: {key}')
    if not metadata['binary'].startswith('_task1_decode_access.'):
        raise RuntimeError('Not a decode access build')
    files = metadata['files'] if full else {metadata['binary']: metadata['files'][metadata['binary']]}
    for name, digest in files.items():
        target = (directory / name).resolve()
        if not target.is_relative_to(directory) or not target.is_file() or sha(target) != digest:
            raise RuntimeError(f'Decode access artifact missing or changed: {name}')
    return metadata


def load_access():
    global _loaded
    if _loaded is not None:
        return _loaded
    build = verify_build(_DIRECTORY, full=False)
    expected = (_DIRECTORY / build['binary']).resolve()
    spec = importlib.util.spec_from_file_location('_task1_decode_access', expected)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load decode access extension: {expected}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != expected:
        raise RuntimeError('Decode access extension loaded from another path')
    _loaded = module
    return module


def decode(tokenizer, ids, case, diagnostic=False):
    if case not in ('N1', 'N2'):
        raise ValueError('Unknown decode access case')
    module = load_access()
    metrics = {}
    function = module.profile_decode if diagnostic else module.decode
    result = function(tokenizer.vocab, ids, case == 'N2', metrics)
    metrics['extension_path'] = str(module.__file__)
    tokenizer._native_decode_metrics = metrics
    return result


def check_result(snapshot, result):
    """Called only AFTER timing, by both trial and ordinary-interface verifier."""
    verify_build(snapshot['directory'])
    metrics = result.get('native_decode') or {}
    if metrics.get('backend') != 'native' or metrics.get('decoder_case') != snapshot['decode_access_case']:
        raise ValueError('Candidate bypassed its decode access backend')
    expected = Path(snapshot['directory']) / snapshot['decode_access_build']['binary']
    if Path(metrics['extension_path']).resolve() != expected.resolve():
        raise ValueError('Decode access binary loaded from another snapshot')
    expected_bytes = result['token_count'] * 4 if snapshot['decode_access_case'] == 'N2' else 0
    if metrics.get('index_buffer_bytes') != expected_bytes:
        raise ValueError('Decode index buffer does not match the actual input list')
