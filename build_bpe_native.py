"""Explicit build/check of Q3. All canonical and generated sources stay in the project."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import bpe_native_deployment as deployment

ROOT = Path(__file__).resolve().parent


def build(rebuild=False):
    directory = ROOT / 'bpe_native'
    if directory.is_symlink():
        raise ValueError('Refusing symlinked BPE deployment directory')
    if (directory / 'deployment.json').is_file() and not rebuild:
        manifest = deployment.check(directory)
        print(f'Reusing verified Q3 deployment: {directory}', flush=True)
        return manifest
    if directory.exists() and not rebuild:
        raise FileExistsError(f'{directory} exists without a valid deployment; use --rebuild')
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir()
    from build_task1_native import build as native
    from build_task1_decode_access import build as decode
    from build_task1_parent_assembly import build as parent
    builds = {role: builder(directory / role) for role, builder in [('native', native), ('decode', decode), ('parent', parent)]}
    manifest = {'schema': 1, 'backend': 'Q3', 'workers': 16, 'parts': 256, 'cache_capacity': 1048576,
                'decoder': 'N2', 'source_location': 'project sources; never an outputs snapshot',
                'executable': sys.executable, 'builds': builds,
                'promotion': {'paired_improvement_fraction': .07363916582137142, 'paired_wins': 5,
                              'integration_passed': False, 'integration_pairs_within_105_percent': 3,
                              'note': 'Fastest measured candidate; cloud integration stability recheck required'}}
    (directory / 'deployment.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    deployment.check(directory)
    print(f'Built Q3 for {sys.executable}: {directory}', flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rebuild', action='store_true', help='Replace only generated project bpe_native artifacts')
    parser.add_argument('--check', action='store_true', help='Validate existing deployment without importing any extension')
    args = parser.parse_args()
    if args.check and args.rebuild:
        parser.error('--check and --rebuild cannot be combined')
    if args.check:
        deployment.check()
        print('Q3 sources, ABI, generated C and binaries verified; no extensions warmed', flush=True)
    else:
        build(args.rebuild)


if __name__ == '__main__':
    main()
