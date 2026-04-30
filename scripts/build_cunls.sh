#!/bin/bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage ./scripts/build_cunls.sh <build_dir> <CMAKE_BUILD_TYPE = Release | Coverage> [install_dir]"
  exit 1
fi

BUILD_DIR=$1
CMAKE_BUILD_TYPE=$2
INSTALL_DIR=${3:-}

CMAKE_EXTRA_ARGS=""
if [ -n "$INSTALL_DIR" ]; then
  CMAKE_EXTRA_ARGS="-D CMAKE_INSTALL_PREFIX=${INSTALL_DIR}"
fi

SOURCE_DIR="${CUNLS_SOURCE_DIR:-..}"

mkdir -p $BUILD_DIR && cd $BUILD_DIR

cmake --version && cmake \
    -D CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
    ${CMAKE_EXTRA_ARGS} \
    ${EXTRA_CMAKE_ARGS:-} \
    -G "Unix Makefiles" -S "$SOURCE_DIR"

if [[ "$CMAKE_BUILD_TYPE" == "Coverage" ]]
then
  echo "Building test coverage"

  add-apt-repository -y universe
  apt-get update
  apt install -y lcov
  make -j 10 coverage
elif [[ "$CMAKE_BUILD_TYPE" == "Release" ]]
then
  echo "Building cuNLS library"
  make -j 10

  if [ -n "$INSTALL_DIR" ]; then
    echo "Installing cuNLS library to $INSTALL_DIR"
    make install
  fi
else
  echo "Wrong CMAKE_BUILD_TYPE. It can only be one of [Release, Coverage]."
fi
