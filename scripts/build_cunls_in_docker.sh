#!/bin/bash

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage ./scripts/build_cunls_in_docker.sh <CMAKE_BUILD_TYPE = Release | Coverage> [local_install_dir]"
  exit 1
fi

CMAKE_BUILD_TYPE=$1
LOCAL_INSTALL_DIR=${2:-$(pwd)/build_docker}
LOCAL_INSTALL_DIR=$(realpath -m "$LOCAL_INSTALL_DIR")
mkdir -p "$LOCAL_INSTALL_DIR"

DOCKERFILE=$(dirname "$(realpath $0)")/Dockerfile
DOCKER_BUILD_ARGS=""
[ -n "${CUDA_VERSION:-}" ] && DOCKER_BUILD_ARGS="$DOCKER_BUILD_ARGS --build-arg CUDA_VERSION=$CUDA_VERSION"
[ -n "${UBUNTU_VERSION:-}" ] && DOCKER_BUILD_ARGS="$DOCKER_BUILD_ARGS --build-arg UBUNTU_VERSION=$UBUNTU_VERSION"
docker build -f $DOCKERFILE . --network host $DOCKER_BUILD_ARGS --tag cunls:local

# Source is read-only; builds happen in the install directory so artifacts
# persist on the host (needed by test_cunls_in_docker.sh).
INSTALL_DIR="/cunls_install"
DOCKER_VOLUMES="-v $(pwd):/cunls:ro -v $LOCAL_INSTALL_DIR:$INSTALL_DIR"

BUILD_CMD="cd $INSTALL_DIR"
BUILD_CMD="$BUILD_CMD && CUNLS_SOURCE_DIR=/cunls EXTRA_CMAKE_ARGS='${EXTRA_CMAKE_ARGS:-} -DBUILD_SHARED_LIBS=ON' /cunls/scripts/build_cunls.sh build_shared $CMAKE_BUILD_TYPE $INSTALL_DIR/shared"
BUILD_CMD="$BUILD_CMD && CUNLS_SOURCE_DIR=/cunls EXTRA_CMAKE_ARGS='${EXTRA_CMAKE_ARGS:-} -DBUILD_SHARED_LIBS=OFF' /cunls/scripts/build_cunls.sh build_static $CMAKE_BUILD_TYPE $INSTALL_DIR/static"

TTY_FLAG=""
[ -t 0 ] && TTY_FLAG="-it"
docker run --gpus all --rm $TTY_FLAG $DOCKER_VOLUMES cunls:local /bin/bash -c "$BUILD_CMD"
