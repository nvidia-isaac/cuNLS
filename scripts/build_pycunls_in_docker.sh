#!/bin/bash
set -euo pipefail

if [ "$#" -gt 1 ]; then
  echo "Usage: ./scripts/build_pycunls_in_docker.sh [local_output_dir]"
  exit 1
fi

LOCAL_OUTPUT_DIR=${1:-$(pwd)/dist}
LOCAL_OUTPUT_DIR=$(realpath -m "$LOCAL_OUTPUT_DIR")

mkdir -p "$LOCAL_OUTPUT_DIR"

CHECKOUT_COMMIT=$(git rev-parse HEAD)
RELEASE_COMMIT=${RELEASE_COMMIT:-$CHECKOUT_COMMIT}
if [ "$RELEASE_COMMIT" != "$CHECKOUT_COMMIT" ]; then
  echo "RELEASE_COMMIT differs from the checked-out source; build the release checkout." >&2
  exit 1
fi
if [ -n "${RELEASE_TAG:-}" ] && ! git diff --quiet HEAD --; then
  echo "Release wheels require committed source; commit the changes before building a release tag." >&2
  exit 1
fi

DOCKERFILE=$(dirname "$(realpath "$0")")/Dockerfile
DOCKER_BUILD_ARGS=""
[ -n "${CUDA_VERSION:-}" ] && DOCKER_BUILD_ARGS="$DOCKER_BUILD_ARGS --build-arg CUDA_VERSION=$CUDA_VERSION"
[ -n "${UBUNTU_VERSION:-}" ] && DOCKER_BUILD_ARGS="$DOCKER_BUILD_ARGS --build-arg UBUNTU_VERSION=$UBUNTU_VERSION"
[ -n "${BASE_IMAGE:-}" ] && DOCKER_BUILD_ARGS="$DOCKER_BUILD_ARGS --build-arg BASE_IMAGE=$BASE_IMAGE"
[ -n "${UBUNTU_PORTS_MIRROR:-}" ] \
  && DOCKER_BUILD_ARGS="$DOCKER_BUILD_ARGS --build-arg UBUNTU_PORTS_MIRROR=$UBUNTU_PORTS_MIRROR"
docker build -f "$DOCKERFILE" . --network host $DOCKER_BUILD_ARGS --tag cunls:local

TTY_FLAG=""
[ -t 0 ] && TTY_FLAG="-it"

# Container mounts:
#   /cunls          (ro) — source tree
#   /cunls_install  (rw) — wheel output (host: $LOCAL_OUTPUT_DIR)
docker run --runtime=nvidia --gpus all --rm $TTY_FLAG \
  -v "$(pwd):/cunls:ro" \
  -v "$LOCAL_OUTPUT_DIR:/cunls_install" \
  -e OUTPUT_DIR=/cunls_install \
  -e RELEASE_COMMIT="$RELEASE_COMMIT" \
  -e RELEASE_TAG="${RELEASE_TAG:-}" \
  -e CUDA_VERSION="${CUDA_VERSION:-}" \
  -e CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-}" \
  cunls:local bash /cunls/scripts/build_pycunls.sh

echo ""
echo "Wheel(s) written to: $LOCAL_OUTPUT_DIR"
ls -lh "$LOCAL_OUTPUT_DIR"/*.whl 2>/dev/null || echo "(no .whl files found — check Docker output above)"
