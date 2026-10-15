#!/bin/bash
# Installs the exact wheel and its extras inside the CUDA image.
set -euo pipefail
shopt -s nullglob
wheels=(/cunls_install/*.whl)
if [ "${#wheels[@]}" -ne 1 ]; then
  echo "Expected one wheel in /cunls_install; build into a separate directory for each matrix cell." >&2
  exit 1
fi
pip install "${wheels[0]}[test]"
cd /cunls
python -m pytest python/tests/ --junitxml=/cunls_install/python-test-results.xml -v
