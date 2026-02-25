#!/bin/bash
set -euo pipefail

if [ "$#" -gt 1 ]; then
  echo "Usage: $0 [host_output_dir]"
  exit 1
fi

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
REPO_ROOT="$(realpath "$SCRIPT_DIR/..")"
DOCKERFILE="$REPO_ROOT/scripts/Dockerfile"

HOST_OUTPUT_DIR="${1:-$REPO_ROOT/docs/build}"
HOST_OUTPUT_DIR="$(realpath "$HOST_OUTPUT_DIR")"
mkdir -p "$HOST_OUTPUT_DIR"

docker build -f "$DOCKERFILE" "$REPO_ROOT" --network host --tag cunls:local

docker run --rm -it \
  -v "$REPO_ROOT:/cunls" \
  -v "$HOST_OUTPUT_DIR:/output" \
  cunls:local /bin/bash -c "
    set -euo pipefail
    rm -rf /tmp/sphinx_html
    python3 -m sphinx -b html /cunls/docs/sphinx /tmp/sphinx_html
    rm -rf /output/*
    cp -a /tmp/sphinx_html/. /output/
  "

echo "Done. Documentation written to: $HOST_OUTPUT_DIR"
