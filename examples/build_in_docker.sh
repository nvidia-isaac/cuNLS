#!/bin/bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <CMAKE_BUILD_TYPE = Release | Coverage> <host_output_dir>"
  exit 1
fi

CMAKE_BUILD_TYPE="$1"
HOST_OUTPUT_DIR="$2"
HOST_OUTPUT_DIR="$(realpath "$HOST_OUTPUT_DIR")"

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
REPO_ROOT="$(realpath "$SCRIPT_DIR/..")"
DOCKERFILE="$REPO_ROOT/scripts/Dockerfile"

mkdir -p "$HOST_OUTPUT_DIR"

docker build -f "$DOCKERFILE" "$REPO_ROOT" --network host --tag cunls:local

docker run --gpus all --rm -it \
  -v "$REPO_ROOT:/cunls" \
  -v "$HOST_OUTPUT_DIR:/output" \
  cunls:local /bin/bash -c "
    set -euo pipefail
    rm -rf /tmp/cunls_build /tmp/examples_build

    cmake -S /cunls -B /tmp/cunls_build \
      -D CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
      -D CMAKE_INSTALL_PREFIX=/output \
      -G 'Unix Makefiles'
    cmake --build /tmp/cunls_build -j 10
    cmake --install /tmp/cunls_build

    cmake -S /cunls/examples -B /tmp/examples_build \
      -D CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
      -D CUNLS_INSTALL_DIR=/output \
      -G 'Unix Makefiles'
    cmake --build /tmp/examples_build -j 10

    cp /tmp/examples_build/sparse_bundle_adjustment_example /output/
    cp /tmp/examples_build/pose_graph_optimization_example /output/
    cp /tmp/examples_build/custom_factor_example /output/

    cat > /output/run_all_examples.sh <<'EOF'
#!/bin/bash
set -euo pipefail

SCRIPT_DIR=\"\$(dirname \"\$(realpath \"\$0\")\")\"
export LD_LIBRARY_PATH=\"\$SCRIPT_DIR/lib:\${LD_LIBRARY_PATH:-}\"

\"\$SCRIPT_DIR/sparse_bundle_adjustment_example\"
\"\$SCRIPT_DIR/pose_graph_optimization_example\"
\"\$SCRIPT_DIR/custom_factor_example\"
EOF
    chmod +x /output/run_all_examples.sh
  "

echo "Done. Installed cuNLS and examples into: ${HOST_OUTPUT_DIR}"
echo "Run all examples with: ${HOST_OUTPUT_DIR}/run_all_examples.sh"
