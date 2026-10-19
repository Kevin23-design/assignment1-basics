"""Explicit, out-of-tree build of the decode-access auxiliary extension."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(output_dir):
    if sys.implementation.name != 'cpython' or sys.version_info[:2] != (3, 13) or sysconfig.get_config_var('Py_GIL_DISABLED'):
        raise RuntimeError('Build requires the GIL-enabled experiment CPython 3.13 interpreter')
    try:
        import Cython
        from Cython.Build import cythonize
        from setuptools import Distribution, Extension
        import setuptools
    except ImportError as error:
        raise RuntimeError('Install Cython, setuptools, wheel and a C compiler in conda base first') from error
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = out / 'inputs'
    inputs.mkdir()
    source = inputs / 'task1_decode_access.pyx'
    shutil.copyfile(ROOT / source.name, source)
    flags = ['-O3']
    extension = Extension('_task1_decode_access', [str(source)], extra_compile_args=flags)
    extensions = cythonize([extension], build_dir=str(out / 'generated'), compiler_directives={'language_level': 3})
    distribution = Distribution({'name': 'task1-decode-access-exploration', 'ext_modules': extensions})
    command = distribution.get_command_obj('build_ext')
    command.build_lib, command.build_temp, command.parallel = str(out), str(out / 'temp'), 1
    command.ensure_finalized()
    command.run()
    binary = Path(command.get_ext_fullpath('_task1_decode_access')).resolve()
    compiler = command.compiler
    compiler_command = getattr(compiler, 'compiler_so', None) or shlex.split(os.environ.get('CC', sysconfig.get_config_var('CC')))
    version = subprocess.run([*compiler_command[:1], '--version'], text=True, capture_output=True, check=True).stdout
    files = [source, binary, *sorted((out / 'generated').rglob('*.c'))]
    metadata = {'directory': str(out), 'python': sys.version, 'executable': sys.executable,
                'implementation': sys.implementation.name, 'cache_tag': sys.implementation.cache_tag,
                'soabi': sysconfig.get_config_var('SOABI'), 'platform': platform.platform(),
                'machine': platform.machine(), 'cython': Cython.__version__, 'setuptools': setuptools.__version__,
                'compiler_command': compiler_command, 'compiler_version': version,
                'compile_flags': flags, 'environment_flags': {k: os.environ.get(k) for k in ('CC', 'CFLAGS', 'CPPFLAGS', 'LDFLAGS')},
                'binary': binary.name, 'files': {str(p.relative_to(out)): sha(p) for p in files},
                'build_script_sha256': sha(__file__), 'pyx_sha256': sha(source), 'runtime_compilation': False, 'gil_disabled': bool(sysconfig.get_config_var('Py_GIL_DISABLED'))}
    (out / 'build.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    print(f'Built for {sys.executable}: {out / "build.json"}', flush=True)
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='outputs/task1-decode-access-build')
    build(parser.parse_args().output_dir)
