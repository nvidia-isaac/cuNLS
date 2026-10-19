#!/bin/bash
# Publishes the built wheel filenames unchanged as public GitHub release assets.
set -euo pipefail

: "${RELEASE_TAG:?Expected v<version>}"
: "${RELEASE_COMMIT:?Expected the built checkout SHA}"
: "${GH_REPO:?Expected nvidia-isaac/cuNLS}"
[ "$GH_REPO" = nvidia-isaac/cuNLS ] || { echo "Only the canonical repository releases wheels." >&2; exit 1; }
[[ "$RELEASE_TAG" = v* ]] || { echo "Expected a v<version> tag." >&2; exit 1; }

sh "$(dirname "$0")/publish_pypi.sh" prepare
validated_paths=$(jq -er '.wheels[].path' pypi-manifest.json)
mapfile -t wheels <<< "$validated_paths"

# Existing assets are immutable inputs to approval. Retry only pypi-publish
# after a dry run or a partial failure, rather than replacing release bytes.
gh release create "$RELEASE_TAG" "${wheels[@]}" --verify-tag \
  --target "$RELEASE_COMMIT" --title "cuNLS $RELEASE_TAG" \
  --notes "Built from commit $RELEASE_COMMIT. Wheels require the matching CUDA toolkit; the Python import remains pycunls."
