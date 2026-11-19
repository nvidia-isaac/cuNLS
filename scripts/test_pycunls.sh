#!/bin/bash
# Installs the exact wheel and its extras inside the CUDA image.
set -euo pipefail
shopt -s nullglob
wheels=(/cunls_install/*.whl)
if [ "${#wheels[@]}" -ne 1 ]; then
  echo "Expected one wheel in /cunls_install; build into a separate directory for each matrix cell." >&2
  exit 1
fi
# Test with the Python the wheel was built for (its tag, e.g. cp313), in a fresh environment.
tag=$(basename "${wheels[0]}" | cut -d- -f3)
uv venv --seed --python "${tag:2:1}.${tag:3}" /tmp/test-venv
/tmp/test-venv/bin/python -m pip install "${wheels[0]}[test]"
cd /cunls
/tmp/test-venv/bin/python -m pytest python/tests/ --junitxml=/cunls_install/python-test-results.xml -v
