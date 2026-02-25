#!/bin/bash

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage ./scripts/build_cunls_in_docker.sh <CMAKE_BUILD_TYPE = Release | Coverage> [local_install_dir]"
  exit 1
fi

CMAKE_BUILD_TYPE=$1
LOCAL_INSTALL_DIR=$2

DOCKERFILE=$(dirname "$(realpath $0)")/Dockerfile
docker build -f $DOCKERFILE . --network host --tag cunls:local

# Use a separate build dir to avoid CMake cache path mismatches
# (host build/ may contain paths like /home/user/... that don't exist in container)
BUILD_DIR="build_docker"
DOCKER_VOLUMES="-v $(pwd):/cunls"
# Clean build dir so we don't reuse stale CMake cache from host or previous runs
BUILD_CMD="rm -rf $BUILD_DIR && ./scripts/build_cunls.sh $BUILD_DIR $CMAKE_BUILD_TYPE"

if [ -n "$LOCAL_INSTALL_DIR" ]; then
  LOCAL_INSTALL_DIR=$(realpath "$LOCAL_INSTALL_DIR")
  mkdir -p "$LOCAL_INSTALL_DIR"
  CONTAINER_INSTALL_DIR="/cunls_install"
  DOCKER_VOLUMES="$DOCKER_VOLUMES -v $LOCAL_INSTALL_DIR:$CONTAINER_INSTALL_DIR"
  BUILD_CMD="rm -rf $BUILD_DIR && ./scripts/build_cunls.sh $BUILD_DIR $CMAKE_BUILD_TYPE $CONTAINER_INSTALL_DIR"
fi

docker run --gpus all --rm -it $DOCKER_VOLUMES cunls:local /bin/bash -c "$BUILD_CMD"
