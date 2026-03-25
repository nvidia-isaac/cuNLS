#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
REPO_ROOT="$(realpath "$SCRIPT_DIR/..")"
DOCKERFILE="$REPO_ROOT/scripts/Dockerfile"
HOST_OUTPUT_DIR="${1:-$REPO_ROOT/install_output}"
HOST_OUTPUT_DIR="$(realpath -m "$HOST_OUTPUT_DIR")"

mkdir -p "$HOST_OUTPUT_DIR"

echo "=== Building Docker image ==="
docker build -f "$DOCKERFILE" "$REPO_ROOT" --network host --tag cunls:local

docker run --gpus all --rm \
  -v "$REPO_ROOT:/cunls:ro" \
  -v "$HOST_OUTPUT_DIR:/output" \
  cunls:local /bin/bash -c '
set -euo pipefail

PASS=0
FAIL=0

check_deps() {
  local binary="$1"
  local variant="$2"
  echo ""
  echo "--- ldd $variant: $binary ---"
  ldd "$binary"
  echo ""
  if ldd "$binary" | grep -qiE "spdlog|cudss"; then
    echo "FAIL: $variant binary has unexpected spdlog/cuDSS runtime dependency"
    FAIL=$((FAIL + 1))
  else
    echo "PASS: $variant binary depends only on CUDA + system libs"
    PASS=$((PASS + 1))
  fi
}

# ---- Shared variant ----
echo ""
echo "========================================="
echo "  Building cuNLS (shared library)"
echo "========================================="
rm -rf /tmp/cunls_build_shared
cmake -S /cunls -B /tmp/cunls_build_shared \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_INSTALL_PREFIX=/output/shared \
  -G "Unix Makefiles"
cmake --build /tmp/cunls_build_shared -j$(nproc)
cmake --install /tmp/cunls_build_shared

echo ""
echo "--- Shared library deps ---"
ldd /output/shared/lib/libcunls.so

echo ""
echo "Building install test (shared)..."
rm -rf /tmp/test_build_shared
cmake -S /cunls/tests/install_test -B /tmp/test_build_shared \
  -DCMAKE_BUILD_TYPE=Release \
  -Dcunls_DIR=/output/shared/lib/cmake/cunls \
  -G "Unix Makefiles"
cmake --build /tmp/test_build_shared -j$(nproc)
check_deps /tmp/test_build_shared/install_test "shared"

# ---- Static variant ----
echo ""
echo "========================================="
echo "  Building cuNLS (static library)"
echo "========================================="
rm -rf /tmp/cunls_build_static
cmake -S /cunls -B /tmp/cunls_build_static \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_INSTALL_PREFIX=/output/static \
  -G "Unix Makefiles"
cmake --build /tmp/cunls_build_static -j$(nproc)
cmake --install /tmp/cunls_build_static

echo ""
echo "--- Static archive size ---"
ls -lh /output/static/lib/libcunls.a

echo ""
echo "Building install test (static)..."
rm -rf /tmp/test_build_static
cmake -S /cunls/tests/install_test -B /tmp/test_build_static \
  -DCMAKE_BUILD_TYPE=Release \
  -Dcunls_DIR=/output/static/lib/cmake/cunls \
  -G "Unix Makefiles"
cmake --build /tmp/test_build_static -j$(nproc)
check_deps /tmp/test_build_static/install_test "static"

# ---- Summary ----
echo ""
echo "========================================="
echo "  Summary"
echo "========================================="
echo "Passed: $PASS  Failed: $FAIL"
if [ "$FAIL" -ne 0 ]; then
  exit 1
fi
echo "All checks passed."
'

echo ""
echo "Done. Installed libraries in: $HOST_OUTPUT_DIR"
echo "  shared -> $HOST_OUTPUT_DIR/shared/"
echo "  static -> $HOST_OUTPUT_DIR/static/"
