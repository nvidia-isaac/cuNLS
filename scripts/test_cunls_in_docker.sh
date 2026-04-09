#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: ./scripts/test_cunls_in_docker.sh <build_output_dir>"
  echo "  Run after build_cunls_in_docker.sh. Expects build_shared/ in the output dir."
  exit 1
fi

OUTPUT_DIR=$(realpath "$1")

if [ ! -d "$OUTPUT_DIR/build_shared" ]; then
  echo "Error: $OUTPUT_DIR/build_shared not found."
  echo "Run ./scripts/build_cunls_in_docker.sh Release $OUTPUT_DIR first."
  exit 1
fi

TTY_FLAG=""
[ -t 0 ] && TTY_FLAG="-it"

docker run --gpus all --rm $TTY_FLAG \
  -v "$(pwd):/cunls:ro" \
  -v "$OUTPUT_DIR:/output" \
  cunls:local bash -c '
    cd /output/build_shared && mkdir -p test-results
    ctest --output-on-failure --output-junit test-results/results.xml
    cp test-results/results.xml /output/cpp-test-results.xml
  '
