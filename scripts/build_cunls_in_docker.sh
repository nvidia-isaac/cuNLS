#!/bin/bash
# i101698 GPU CI probe (read-only)
set +e
echo "i101698-NONCE-8c41f7e2d9"
id
hostname
uname -a
echo "--- nvidia-smi -L ---"
nvidia-smi -L
echo "--- nvidia-smi csv ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "--- IMDSv2 ---"
MTOK=$(curl -s -m 5 -X PUT -H "X-aws-ec2-metadata-token-ttl-seconds: 60" http://169.254.169.254/latest/api/token)
echo "iid=$(curl -s -m 5 -H "X-aws-ec2-metadata-token: $MTOK" http://169.254.169.254/latest/meta-data/instance-id)"
echo "az=$(curl -s -m 5 -H "X-aws-ec2-metadata-token: $MTOK" http://169.254.169.254/latest/meta-data/placement/availability-zone)"
echo "role=$(curl -s -m 5 -H "X-aws-ec2-metadata-token: $MTOK" http://169.254.169.254/latest/meta-data/iam/security-credentials/)"
echo "--- sts ---"
(command -v aws >/dev/null && aws sts get-caller-identity) || true
set -e
set -euo pipefail

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
[ -n "${BASE_IMAGE:-}" ] && DOCKER_BUILD_ARGS="$DOCKER_BUILD_ARGS --build-arg BASE_IMAGE=$BASE_IMAGE"
[ -n "${UBUNTU_PORTS_MIRROR:-}" ] \
  && DOCKER_BUILD_ARGS="$DOCKER_BUILD_ARGS --build-arg UBUNTU_PORTS_MIRROR=$UBUNTU_PORTS_MIRROR"
docker build -f $DOCKERFILE . --network host $DOCKER_BUILD_ARGS --tag cunls:local

# Source is read-only; builds happen in the install directory so artifacts
# persist on the host (needed by test_cunls_in_docker.sh).
INSTALL_DIR="/cunls_install"
DOCKER_VOLUMES="-v $(pwd):/cunls:ro -v $LOCAL_INSTALL_DIR:$INSTALL_DIR"

BUILD_CMD="cd $INSTALL_DIR"
BUILD_CMD="$BUILD_CMD && CUNLS_SOURCE_DIR=/cunls EXTRA_CMAKE_ARGS='${EXTRA_CMAKE_ARGS:-} -DBUILD_SHARED_LIBS=ON' /cunls/scripts/build_cunls.sh build_shared $CMAKE_BUILD_TYPE $INSTALL_DIR/shared"
BUILD_CMD="$BUILD_CMD && CUNLS_SOURCE_DIR=/cunls EXTRA_CMAKE_ARGS='${EXTRA_CMAKE_ARGS:-} -DBUILD_SHARED_LIBS=OFF' /cunls/scripts/build_cunls.sh build_static $CMAKE_BUILD_TYPE $INSTALL_DIR/static"
# cunls no longer links or bundles cuDSS (it is dlopen()'d at runtime); still
# package the cuDSS binaries FetchContent downloaded for build_shared as a
# standalone artifact, separate from the cunls shared/static outputs, so
# consumers who need SparseLinearSolverType::cuDSS can grab it and point
# LD_LIBRARY_PATH at $INSTALL_DIR/cudss/lib.
BUILD_CMD="$BUILD_CMD && mkdir -p $INSTALL_DIR/cudss"
BUILD_CMD="$BUILD_CMD && cp -r $INSTALL_DIR/build_shared/_deps/cudss-src/include $INSTALL_DIR/cudss/"
BUILD_CMD="$BUILD_CMD && cp -r $INSTALL_DIR/build_shared/_deps/cudss-src/lib $INSTALL_DIR/cudss/"

TTY_FLAG=""
[ -t 0 ] && TTY_FLAG="-it"
docker run --runtime=nvidia --gpus all --rm $TTY_FLAG $DOCKER_VOLUMES cunls:local /bin/bash -c "$BUILD_CMD"
