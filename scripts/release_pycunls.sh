#!/bin/bash
# Publishes the built wheel filenames unchanged as public GitHub release assets.
set -euo pipefail

: "${RELEASE_TAG:?Expected v<version>}"
: "${RELEASE_COMMIT:?Expected the built checkout SHA}"
: "${GH_REPO:?Expected nvidia-isaac/cuNLS}"
: "${PYPI_PUBLISH_CELLS:?Expected the release matrix}"
[ "$GH_REPO" = nvidia-isaac/cuNLS ] || { echo "Only the canonical repository releases wheels." >&2; exit 1; }
[[ "$RELEASE_TAG" = v* ]] || { echo "Expected a v<version> tag." >&2; exit 1; }

shopt -s nullglob
wheels=(pypi-wheels/*/*.whl)
expected=$(printf '%s' "$PYPI_PUBLISH_CELLS" | jq -er 'length | select(. > 0)')
[ "${#wheels[@]}" -eq "$expected" ] || { echo "Expected $expected release wheels; check matrix artifacts." >&2; exit 1; }
declare -A seen
for wheel in "${wheels[@]}"; do
  name=${wheel##*/}
  [ -z "${seen[$name]:-}" ] || { echo "Duplicate release filename: $name" >&2; exit 1; }
  seen[$name]=1
done

# Existing assets are immutable inputs to approval. Retry only pypi-publish
# after a dry run or a partial failure, rather than replacing release bytes.
gh release create "$RELEASE_TAG" "${wheels[@]}" --verify-tag \
  --target "$RELEASE_COMMIT" --title "cuNLS $RELEASE_TAG" \
  --notes "Built from commit $RELEASE_COMMIT. Wheels require the matching CUDA toolkit; the Python import remains pycunls."
