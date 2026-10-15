#!/bin/bash
# Runs inside the existing CUDA Docker image; the checkout is read-only.
set -euo pipefail

: "${RELEASE_COMMIT:?Pass the original checkout SHA through build_pycunls_in_docker.sh}"
ACTUAL_CUDA_VERSION=$(python3 - <<'PY'
import json
import os
import re
import subprocess
from pathlib import Path

cuda = Path('/usr/local/cuda')
match = re.search(r'release (\d+\.\d+)', subprocess.check_output([str(cuda / 'bin/nvcc'), '--version'], text=True))
if not match:
    raise SystemExit('Cannot identify the installed CUDA toolkit; use a CUDA development image')
path = cuda / 'version.json'
actual = json.loads(path.read_text())['cuda']['version'] if path.exists() else match[1]
if actual.split('.')[:2] != match[1].split('.'):
    raise SystemExit(f'CUDA version.json {actual} disagrees with nvcc {match[1]}')
requested = os.environ.get('CUDA_VERSION')
if requested and not (actual == requested or actual.startswith(requested + '.')):
    raise SystemExit(f'Requested CUDA {requested}, but the build image contains {actual}; correct CUDA_VERSION or BASE_IMAGE')
print(actual)
PY
)
export CUDA_VERSION=$ACTUAL_CUDA_VERSION
export RELEASE_COMMIT
export RELEASE_TAG=${RELEASE_TAG:-}

WORK_DIR=$(mktemp -d)
trap 'rm -rf "$WORK_DIR"' EXIT
mkdir -p "$WORK_DIR/raw" "$WORK_DIR/repaired"
cp -r /cunls "$WORK_DIR/source"
rm -rf "$WORK_DIR/source/python/build" "$WORK_DIR/source/build_python"
cd "$WORK_DIR/source/python"

python3 - <<'PY'
import json
import os
import re
from pathlib import Path
from packaging.version import Version

cuda = os.environ['CUDA_VERSION']
major = cuda.split('.')[0]
if major not in ('12', '13'):
    raise SystemExit(f'Unsupported CUDA version: {cuda}; expected CUDA 12 or 13')
sha = os.environ['RELEASE_COMMIT']
if not re.fullmatch(r'[0-9a-f]{40}', sha):
    raise SystemExit('RELEASE_COMMIT must be the full original checkout SHA')
path = Path('pyproject.toml')
text = path.read_text()
text, count = re.subn(r'^name = "pycunls"$', f'name = "pycunls-cu{major}"', text, flags=re.M)
if count != 1:
    raise SystemExit('Expected one pycunls project name in pyproject.toml')
text = text.replace('cupy-cuda12x', f'cupy-cuda{major}x')
tag = os.environ['RELEASE_TAG']
if tag:
    if not tag.startswith('v'):
        raise SystemExit('Release tag must start with v')
    version = Version(tag[1:])
    if version.local or str(version) != tag[1:]:
        raise SystemExit('Release tag must contain a normalized public PEP 440 version')
    current = re.findall(r'^version = "([^"]+)"$', text, flags=re.M)
    if current != [str(version)]:
        raise SystemExit(f'Package version {current} differs from {tag}; update version sources before tagging')
    init_version = re.findall(r'^__version__ = "([^"]+)"$', Path('pycunls/__init__.py').read_text(), flags=re.M)
    if init_version != [str(version)]:
        raise SystemExit(f'Import version {init_version} differs from {tag}; update version sources before tagging')
path.write_text(text)
Path('pycunls/_build_provenance.json').write_text(json.dumps({
    'git_sha': sha, 'cuda_version': cuda,
}, sort_keys=True) + '\n')
PY

export CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}
pip wheel . --no-build-isolation --no-deps --wheel-dir "$WORK_DIR/raw"

# The system CUDA toolkit supplies these linked runtime libraries. Match SONAME
# families rather than assuming CUDA 12 and 13 use the same SONAME versions.
PLATFORM=$(python3 -c 'import os, platform; print("manylinux_" + os.confstr("CS_GNU_LIBC_VERSION").split()[1].replace(".", "_") + "_" + platform.machine())')
for wheel in "$WORK_DIR/raw"/*.whl; do
  auditwheel repair "$wheel" --plat "$PLATFORM" --wheel-dir "$WORK_DIR/repaired" \
    --exclude 'libcudart.so.*' --exclude 'libcusparse.so.*' \
    --exclude 'libcublas.so.*' --exclude 'libcublasLt.so.*' \
    --exclude 'libcusolver.so.*' --exclude 'libcuda.so.*'
done

python3 - "$WORK_DIR/repaired" <<'PY'
import json
import os
import re
import sys
import zipfile
from pathlib import Path

wheels = list(Path(sys.argv[1]).glob('*.whl'))
if len(wheels) != 1:
    raise SystemExit(f'Expected one repaired wheel, found {len(wheels)}')
with zipfile.ZipFile(wheels[0]) as archive:
    provenance = json.loads(archive.read('pycunls/_build_provenance.json'))
    if provenance != {'git_sha': os.environ['RELEASE_COMMIT'], 'cuda_version': os.environ['CUDA_VERSION']}:
        raise SystemExit('Repaired wheel lost the original build provenance')
    for member in archive.namelist():
        if re.search(r'(^|/)lib(cudart|cusparse|cublas|cusolver|cudss|cuda)[^/]*\.so', member):
            raise SystemExit(f'CUDA runtime library must remain external: {member}')
PY

OUTPUT=${OUTPUT_DIR:-/output}
mkdir -p "$OUTPUT"
cp "$WORK_DIR/repaired"/*.whl "$OUTPUT/"
ls -lh "$OUTPUT"/*.whl
