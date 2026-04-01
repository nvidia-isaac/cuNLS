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
docker build -f $DOCKERFILE . --network host --tag cunls:local

# Source is read-only; builds happen inside the container's local filesystem.
# Only the final install directory is mounted to the host.
INSTALL_DIR="/cunls_install"
DOCKER_VOLUMES="-v $(pwd):/cunls:ro -v $LOCAL_INSTALL_DIR:$INSTALL_DIR"

BUILD_CMD="cd /tmp"
BUILD_CMD="$BUILD_CMD && CUNLS_SOURCE_DIR=/cunls EXTRA_CMAKE_ARGS='-DBUILD_SHARED_LIBS=ON' /cunls/scripts/build_cunls.sh build_shared $CMAKE_BUILD_TYPE $INSTALL_DIR"
BUILD_CMD="$BUILD_CMD && CUNLS_SOURCE_DIR=/cunls EXTRA_CMAKE_ARGS='-DBUILD_SHARED_LIBS=OFF' /cunls/scripts/build_cunls.sh build_static $CMAKE_BUILD_TYPE $INSTALL_DIR"

docker run --gpus all --rm -it $DOCKER_VOLUMES cunls:local /bin/bash -c "$BUILD_CMD"
