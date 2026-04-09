#!/bin/bash
set -euo pipefail

if [ "$#" -gt 1 ]; then
  echo "Usage: ./scripts/build_pycunls_in_docker.sh [local_output_dir]"
  exit 1
fi

LOCAL_OUTPUT_DIR=${1:-$(pwd)/dist}
LOCAL_OUTPUT_DIR=$(realpath -m "$LOCAL_OUTPUT_DIR")

mkdir -p "$LOCAL_OUTPUT_DIR"

DOCKERFILE=$(dirname "$(realpath "$0")")/Dockerfile
DOCKER_BUILD_ARGS=""
[ -n "${CUDA_VERSION:-}" ] && DOCKER_BUILD_ARGS="$DOCKER_BUILD_ARGS --build-arg CUDA_VERSION=$CUDA_VERSION"
[ -n "${UBUNTU_VERSION:-}" ] && DOCKER_BUILD_ARGS="$DOCKER_BUILD_ARGS --build-arg UBUNTU_VERSION=$UBUNTU_VERSION"
docker build -f "$DOCKERFILE" . --network host $DOCKER_BUILD_ARGS --tag cunls:local

TTY_FLAG=""
[ -t 0 ] && TTY_FLAG="-it"

# Source is read-only; builds happen inside the container's local filesystem.
# Only the final wheel output directory is mounted to the host.
docker run --gpus all --rm $TTY_FLAG \
  -v "$(pwd):/cunls:ro" \
  -v "$LOCAL_OUTPUT_DIR:/output" \
  cunls:local bash /cunls/scripts/build_pycunls.sh

echo ""
echo "Wheel(s) written to: $LOCAL_OUTPUT_DIR"
ls -lh "$LOCAL_OUTPUT_DIR"/*.whl 2>/dev/null || echo "(no .whl files found — check Docker output above)"
