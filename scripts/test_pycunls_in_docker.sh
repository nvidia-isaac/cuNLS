#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: ./scripts/test_pycunls_in_docker.sh <wheel_dir>"
  exit 1
fi

WHEEL_DIR=$(realpath "$1")

if ! ls "$WHEEL_DIR"/*.whl 1>/dev/null 2>&1; then
  echo "Error: no .whl files found in $WHEEL_DIR"
  echo "Run ./scripts/build_pycunls_in_docker.sh $WHEEL_DIR first."
  exit 1
fi

TTY_FLAG=""
[ -t 0 ] && TTY_FLAG="-it"

# Runs in the CUDA 13 image cunls:local that build_pycunls_in_docker.sh built (test_pycunls.sh
# checks it). WARP_REQUIREMENT (e.g. "warp-lang==1.18.*") pins Warp; empty: the newest allowed.
# Container mounts:
#   /cunls          (ro) — source tree
#   /cunls_install  (rw) — wheel dir + test output (host: $WHEEL_DIR)
docker run --runtime=nvidia --gpus all --rm $TTY_FLAG \
  -v "$(pwd):/cunls:ro" \
  -v "$WHEEL_DIR:/cunls_install" \
  -e WARP_REQUIREMENT="${WARP_REQUIREMENT:-}" \
  cunls:local bash /cunls/scripts/test_pycunls.sh
