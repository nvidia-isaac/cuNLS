#!/bin/bash
# Publishes the built wheel filenames unchanged as public GitHub release assets.
set -euo pipefail

: "${RELEASE_TAG:?Expected v<version>}"
: "${RELEASE_COMMIT:?Expected the built checkout SHA}"
: "${GH_REPO:?Expected nvidia-isaac/cuNLS}"
[ "$GH_REPO" = nvidia-isaac/cuNLS ] || { echo "Only the canonical repository releases wheels." >&2; exit 1; }
[[ "$RELEASE_TAG" = v* ]] || { echo "Expected a v<version> tag." >&2; exit 1; }

sha256sum --check pypi-manifest.sha256
jq -r '.wheels[] | "\(.sha256)  \(.path)"' pypi-manifest.json | sha256sum --check
validated_paths=$(jq -er '.wheels[].path' pypi-manifest.json)
mapfile -t wheels <<< "$validated_paths"

# gh attaches every asset to a draft before publishing the immutable release.
gh release create "$RELEASE_TAG" "${wheels[@]}" pypi-manifest.json pypi-manifest.sha256 --verify-tag \
  --target "$RELEASE_COMMIT" --title "cuNLS $RELEASE_TAG" \
  --notes "Built from commit $RELEASE_COMMIT. Wheels require the matching CUDA toolkit; the Python import remains pycunls."
