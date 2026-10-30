#!/bin/bash
# Publishes the built wheel filenames unchanged as public GitHub release assets.
set -euo pipefail

: "${RELEASE_TAG:?Expected v<version>}"
: "${RELEASE_COMMIT:?Expected the built checkout SHA}"
: "${GH_REPO:?Expected nvidia-isaac/cuNLS}"
[ "$GH_REPO" = nvidia-isaac/cuNLS ] || { echo "Only the canonical repository releases wheels." >&2; exit 1; }
[[ "$RELEASE_TAG" = v* ]] || { echo "Expected a v<version> tag." >&2; exit 1; }

jq -e --arg tag "$RELEASE_TAG" --arg commit "$RELEASE_COMMIT" --arg repository "$GH_REPO" '
  .template_version == "2.0.0" and .tag == $tag and .commit == $commit and
  .repository == $repository and .asset_source == "github-release" and
  (.wheels | length > 0)' pypi-manifest.json >/dev/null
sha256sum --check pypi-manifest.sha256
validated_paths=$(jq -er '.wheels[].path' pypi-manifest.json)
mapfile -t wheels <<< "$validated_paths"
jq -r '.wheels[] | [.path,.sha256,.bytes] | @tsv' pypi-manifest.json |
  while IFS=$'\t' read -r wheel checksum bytes; do
    actual=$(sha256sum "$wheel"); actual=${actual%% *}
    [ "$actual" = "$checksum" ] || { echo "Wheel hash differs: $wheel" >&2; exit 1; }
    actual_bytes=$(wc -c < "$wheel" | tr -d ' ')
    [ "$actual_bytes" = "$bytes" ] || { echo "Wheel size differs: $wheel" >&2; exit 1; }
  done

# gh attaches every asset to a draft before publishing the immutable release.
gh release create "$RELEASE_TAG" "${wheels[@]}" pypi-manifest.json pypi-manifest.sha256 --verify-tag \
  --target "$RELEASE_COMMIT" --title "cuNLS $RELEASE_TAG" \
  --notes "Built from commit $RELEASE_COMMIT. Wheels require the matching CUDA toolkit; the Python import remains pycunls."
