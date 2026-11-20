#!/bin/bash
# Installs the exact wheel and its extras inside the CUDA image.
set -euo pipefail
shopt -s nullglob
wheels=(/cunls_install/*.whl)
if [ "${#wheels[@]}" -ne 1 ]; then
  echo "Expected one wheel in /cunls_install; build into a separate directory for each matrix cell." >&2
  exit 1
fi
# The wheels are CUDA 13 builds and use the image's CUDA libraries.
cuda=$(sed -n 's/.*release \([0-9]*\)\..*/\1/p' <(/usr/local/cuda/bin/nvcc --version))
if [ "$cuda" != "13" ]; then
  echo "test_pycunls.sh: the image has CUDA $cuda, but pycunls wheels need CUDA 13 (got $cuda," \
       "expected 13). Fix: ./scripts/build_pycunls_in_docker.sh <wheel_dir> (builds the CUDA 13" \
       "image cunls:local), then run this test again." >&2
  exit 1
fi
# Test with the Python the wheel was built for (its tag, e.g. cp313), in a fresh environment.
tag=$(basename "${wheels[0]}" | cut -d- -f3)
uv venv --seed --python "${tag:2:1}.${tag:3}" /tmp/test-venv
# WARP_REQUIREMENT (e.g. "warp-lang==1.18.*") constrains Warp; empty: the newest the wheel allows.
/tmp/test-venv/bin/python -m pip install "${wheels[0]}[test]" ${WARP_REQUIREMENT:+"$WARP_REQUIREMENT"}
cd /cunls
/tmp/test-venv/bin/python -m pytest python/tests/ --junitxml=/cunls_install/python-test-results.xml -v
