"""Native deployment metadata only; extension imports remain inside encode/decode."""
from functools import cache
import hashlib
import json
import os
from pathlib import Path
import sys
import sysconfig

ROOT = Path(__file__).resolve().parent
ROLES = {'native': 'task1_native.pyx', 'decode': 'task1_decode_access.pyx', 'parent': 'task1_parent_assembly.pyx'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(directory=None, full=True):
    root = ROOT / 'bpe_native' if directory is None else Path(directory)
    if root.is_symlink() or not root.resolve().is_relative_to(ROOT.resolve()):
        raise RuntimeError('Native BPE artifacts must stay inside the project')
    manifest = json.loads((root / 'deployment.json').read_text())
    if manifest.get('schema') != 1 or manifest.get('backend') != 'Q3':
        raise RuntimeError('Unknown BPE native deployment; rebuild with build_bpe_native.py')
    if (sys.implementation.name != 'cpython' or sys.version_info[:2] != (3, 13)
            or sysconfig.get_config_var('Py_GIL_DISABLED')):
        raise RuntimeError('Native BPE requires GIL-enabled CPython 3.13')
    for role, name in ROLES.items():
        if (root / role).is_symlink():
            raise RuntimeError('Native BPE build roles cannot be symlinked')
        meta = manifest['builds'][role]
        if (meta['soabi'] != sysconfig.get_config_var('SOABI') or meta['cache_tag'] != sys.implementation.cache_tag
                or meta['pyx_sha256'] != sha(ROOT / name)):
            raise RuntimeError(f'Native BPE source/ABI changed: {role}; run build_bpe_native.py --rebuild')
        if json.loads((root / role / 'build.json').read_text()) != meta:
            raise RuntimeError(f'Native BPE build manifest changed: {role}')
        # --check audits everything. Each loader hashes its actual binary on
        # first use; fresh workers need not hash unrelated generated C files.
        for relative, digest in (meta['files'].items() if full else ()):
            path = (root / role / relative).resolve()
            if not path.is_relative_to((root / role).resolve()) or not path.is_file() or sha(path) != digest:
                raise RuntimeError(f'Native BPE artifact missing/changed: {role}/{relative}')
    return manifest


@cache
def enabled():
    if os.environ.get('BPE_PYTHON_ONLY') == '1':
        if os.environ.get('BPE_REQUIRE_NATIVE') == '1':
            raise RuntimeError('BPE_PYTHON_ONLY conflicts with BPE_REQUIRE_NATIVE')
        return False
    if not (ROOT / 'bpe_native' / 'deployment.json').is_file():
        if os.environ.get('BPE_REQUIRE_NATIVE') == '1':
            raise RuntimeError('Compile native BPE first: python build_bpe_native.py')
        return False
    check(full=False)
    return True
