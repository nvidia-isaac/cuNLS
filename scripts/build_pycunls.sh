#!/bin/bash
# Runs inside the Docker container to build the pycunls wheel.
# Expected mounts:
#   /cunls                    (read-only)  — source tree
#   OUTPUT_DIR env or /output (read-write) — wheel output directory
set -euo pipefail

WHEEL_DIR=/tmp/pycunls_wheel
mkdir -p "$WHEEL_DIR"

cp -r /cunls /tmp/cunls_src
rm -rf /tmp/cunls_src/python/build /tmp/cunls_src/build_python

cd /tmp/cunls_src/python
pip wheel . --no-build-isolation --no-deps --wheel-dir "$WHEEL_DIR"

echo ""
echo "========================================="
echo "  Wheel built successfully"
echo "========================================="
ls -lh "$WHEEL_DIR"/*.whl

OUTPUT=${OUTPUT_DIR:-/output}
cp "$WHEEL_DIR"/*.whl "$OUTPUT"/
echo "Wheel copied to $OUTPUT/"
