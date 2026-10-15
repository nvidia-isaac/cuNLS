#!/bin/sh
# TEMPLATE_VERSION=1.9.0
set -eu
set -f

fail() {
    printf '%s: actual=%s; expected=%s; fix=%s' "$1" "$2" "$3" "$4" | tr '\n' ' ' >&2
    printf '\n' >&2
    exit 1
}
for dependency in curl jq unzip sha256sum; do
    command -v "$dependency" >/dev/null || fail "$0" "$dependency missing" 'curl jq unzip sha256sum' "install $dependency on the runner host"
done
mode=${1:-}
case "$mode" in prepare|publish) ;; *) fail "$0" "$mode" 'prepare | publish' 'select one command' ;; esac
PYPI_ASSET_SOURCE=${PYPI_ASSET_SOURCE:-github-release}
KITMAKER_PIC=${KITMAKER_PIC:-}
RELEASE_TAG=${RELEASE_TAG:-${GITHUB_REF_NAME:-}}
RELEASE_COMMIT=${RELEASE_COMMIT:-${GITHUB_SHA:-}}
PYPI_PLATFORM=${PYPI_PLATFORM:-manylinux_*_x86_64}
PYPI_DRY_RUN_ONLY=${PYPI_DRY_RUN_ONLY:-1}
PYPI_PER_RELEASE_PLC=${PYPI_PER_RELEASE_PLC:-no}
PYPI_POLL_SECONDS=${PYPI_POLL_SECONDS:-10}
PYPI_TIMEOUT_SECONDS=${PYPI_TIMEOUT_SECONDS:-1800}
PYPI_INDEX_TIMEOUT_SECONDS=${PYPI_INDEX_TIMEOUT_SECONDS:-600}
KITMAKER_API=https://kitmaker-portal.nvidia.com/api/v0
manifest=pypi-manifest.json
umask 077
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT HUP INT TERM
normalize() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[-_.][-_\.]*/-/g'; }
required() {
    missing=
    for input_name in "$@"; do
        eval 'input_value=${'"$input_name"':-}'
        [ -n "$input_value" ] || missing="$missing $input_name"
    done
    [ -z "$missing" ] || fail "$0" "missing:$missing" 'all required inputs set' 'export the listed names from the confirmed context'
}
repository_guard() {
    [ "$PYPI_ASSET_SOURCE" = github-release ] || fail "$0" "$PYPI_ASSET_SOURCE" 'github-release' 'only public GitHub release assets are supported'
    actual_repository=${GITHUB_REPOSITORY:-}
    [ "$actual_repository" = "$PYPI_PUBLISH_REPOSITORY" ] || fail "$0" "$actual_repository" "$PYPI_PUBLISH_REPOSITORY" "private copies don't publish; run in the canonical repository"
}
public_version() {
    printf '%s' "$1" | jq -eRs 'test("^([0-9]+!)?[0-9]+(\\.[0-9]+)*((a|b|rc)[0-9]+)?(\\.post[0-9]+)?(\\.dev[0-9]+)?$")' >/dev/null
}
project_map() {
    printf '%s\n' "${KITMAKER_PROJECTS:-}" | jq -Rsc '
      split("\n") | map(select(length > 0) | split("=")) |
      if length == 0 or any(.[]; length != 2 or (.[0] | test("^[a-z0-9]+(-[a-z0-9]+)*$") | not) or (.[1] | test("^[1-9][0-9]*$") | not))
      then error("expected canonical-name=numeric-id, one per line")
      elif (map(.[0]) | unique | length) != length or (map(.[1]) | unique | length) != length
      then error("duplicate name or project ID")
      else map({name:.[0], id:.[1]}) end' > "$scratch/map.json"
}
valid_cell_directories() {
    printf '%s' "$1" | jq -e 'type == "array" and length > 0 and all(.[];
      (.cuda | type == "string") and (.artifact_dir | type == "string" and (. == "." or (test("^[A-Za-z0-9_][A-Za-z0-9_./-]*$") and (split("/") | all(.[]; . != ".." and . != "." and . != "")))))) and
      (length == 1 or all(.[]; .artifact_dir != ".")) and
      (map(.artifact_dir) | unique | length) == length' >/dev/null
}
# Preserve error bodies and never retry a POST automatically.
http() {
    http_status=000
    http_rc=0
    : > "$scratch/response"
    http_status=$(curl --fail-with-body --show-error --location --proto '=https' --proto-redir '=https' --connect-timeout 15 --max-time 120 --output "$scratch/response" --write-out '%{http_code}' "$@") || http_rc=$?
    printf 'HTTP %s (curl=%s)\n' "$http_status" "$http_rc"
}
kitmaker() {
    http --header "Authorization: Bearer $KITMAKER_TOKEN" --header 'Content-Type: application/json' "$@"
    cat "$scratch/response"
    printf '\n'
    [ "$http_rc" -eq 0 ] || fail "$KITMAKER_API" "HTTP $http_status, curl $http_rc" 'successful response' 'inspect the response; check token, network and NVIDIA CA; never blindly resubmit'
}

prepare() {
    required PYPI_PUBLISH_REPOSITORY PYPI_ASSET_SOURCE RELEASE_TAG RELEASE_COMMIT WHEEL_DIR KITMAKER_PROJECTS KITMAKER_PIC PYPI_PROVENANCE_PATH PYPI_PUBLISH_CELLS
    repository_guard
    case "$PYPI_PROVENANCE_PATH" in /*|../*|*/../*|*..|*\**|*\?*|*\[*) fail "$0" "$PYPI_PROVENANCE_PATH" 'literal wheel-relative provenance path' 'use import_package/_build_provenance.json' ;; esac
    case "$RELEASE_COMMIT" in *[!a-fA-F0-9]*|'') fail "$0" "$RELEASE_COMMIT" 'full git commit SHA' 'use the commit built by the release matrix' ;; esac
    [ "${#RELEASE_COMMIT}" -eq 40 ] || fail "$0" "$RELEASE_COMMIT" '40-character git SHA' 'resolve the full release commit'
    project_map || fail "$0" "$KITMAKER_PROJECTS" 'unique canonical-name=numeric-id pairs' 'correct KITMAKER_PROJECTS'
    valid_cell_directories "$PYPI_PUBLISH_CELLS" || fail "$0" "$PYPI_PUBLISH_CELLS" 'selected cells with unique safe artifact_dir paths' 'record each selected cell artifact directory from the producer'
    printf '%s\n' "$PYPI_PUBLISH_CELLS" > "$scratch/cells"
    version=${RELEASE_TAG#v}
    public_version "$version" || fail "$0" "$version" 'normalized public PEP 440 version, no +local' 'set v<version> and rebuild all wheels'
    [ -d "$WHEEL_DIR" ] || fail "$WHEEL_DIR" 'missing directory' 'downloaded build artifacts' 'download the selected wheel artifacts'
    find "$WHEEL_DIR" -type f -name '*.whl' | LC_ALL=C sort > "$scratch/files"
    : > "$scratch/entries"
    while IFS= read -r wheel; do
        filename=${wheel##*/}
        stem=${filename%.whl}
        platform=${stem##*-}
        matched=0
        old_ifs=$IFS; IFS=.
        for wheel_platform in $platform; do
            # PYPI_PLATFORM is intentionally a caller-selected shell pattern.
            # shellcheck disable=SC2254
            case "$wheel_platform" in $PYPI_PLATFORM) matched=1 ;; esac
        done
        IFS=$old_ifs
        if [ "$matched" -eq 0 ]; then printf 'SKIP %s: platform=%s; selected=%s\n' "$wheel" "$platform" "$PYPI_PLATFORM"; continue; fi
        fields=$(printf '%s' "$stem" | awk -F- '{print NF}')
        [ "$fields" -eq 5 ] || fail "$wheel" "$filename ($fields fields)" 'five wheel filename fields; no build tag' 'remove the build tag in the producer and rebuild'
        distribution=$(normalize "${stem%%-*}")
        remainder=${stem#*-}; wheel_version=${remainder%%-*}
        project_id=$(jq -r --arg name "$distribution" '.[] | select(.name == $name) | .id' "$scratch/map.json")
        [ -n "$project_id" ] || fail "$wheel" "$distribution" "$KITMAKER_PROJECTS" 'register the distribution and add its Kitmaker project mapping'
        [ "$wheel_version" = "$version" ] || fail "$wheel" "$wheel_version" "$version (no +local)" 'set the build version from the release tag and rebuild'
        unzip -Z1 "$wheel" > "$scratch/members" || fail "$wheel" 'invalid ZIP' 'readable wheel archive' 'rebuild the wheel'
        metadata_count=$(awk '/^[^\/]+\.dist-info\/METADATA$/ {n++} END {print n+0}' "$scratch/members")
        [ "$metadata_count" -eq 1 ] || fail "$wheel" "$metadata_count METADATA entries" 'exactly one .dist-info/METADATA' 'rebuild the wheel with its packaging backend'
        metadata_path=$(awk '/^[^\/]+\.dist-info\/METADATA$/ {print}' "$scratch/members")
        unzip -p "$wheel" "$metadata_path" > "$scratch/metadata" || fail "$wheel" 'unreadable METADATA' 'readable core metadata' 'rebuild the wheel'
        metadata_name=$(sed -n 's/^Name: //p' "$scratch/metadata" | tr -d '\r')
        metadata_version=$(sed -n 's/^Version: //p' "$scratch/metadata" | tr -d '\r')
        [ "$(normalize "$metadata_name")" = "$distribution" ] || fail "$wheel" "METADATA Name=$metadata_name" "$distribution" 'change package metadata before building; do not rename the wheel'
        [ "$metadata_version" = "$wheel_version" ] || fail "$wheel" "METADATA Version=$metadata_version" "$wheel_version" 'align the version sources and rebuild'
        case ".$platform." in *.linux_*.*) fail "$wheel" "$platform" 'repaired manylinux platform; no bare linux_*' 'run auditwheel repair excluding CUDA libraries' ;; esac
        duplicate=$(jq -rs --arg file "$filename" '[.[] | select(.filename == $file)][0].path // empty' "$scratch/entries")
        [ -z "$duplicate" ] || fail "$wheel" "duplicate $filename at $duplicate and $wheel" 'unique wheel filenames across selected cells' 'select one cell for each CUDA-major, Python/Ubuntu and architecture combination'
        provenance_count=$(awk -v path="$PYPI_PROVENANCE_PATH" '$0 == path {n++} END {print n+0}' "$scratch/members")
        [ "$provenance_count" -eq 1 ] || fail "$wheel" "$provenance_count $PYPI_PROVENANCE_PATH entries" 'one provenance file' 'write and package provenance before pip wheel'
        unzip -p "$wheel" "$PYPI_PROVENANCE_PATH" > "$scratch/provenance" || fail "$wheel" 'unreadable provenance' 'JSON provenance' 'rebuild the wheel'
        git_sha=$(jq -er '.git_sha | select(type == "string")' "$scratch/provenance") || fail "$wheel" 'missing or invalid git_sha' "$RELEASE_COMMIT" 'write JSON provenance before pip wheel'
        [ "$git_sha" = "$RELEASE_COMMIT" ] || fail "$wheel" "git_sha=$git_sha" "$RELEASE_COMMIT" 'rebuild from the release commit'
        cuda_major=$(printf '%s' "$distribution" | sed -n 's/.*-cu\([0-9][0-9]*\)$/\1/p')
        if [ -n "$cuda_major" ]; then
            cuda_version=$(jq -r '.cuda_version // empty' "$scratch/provenance")
            [ "${cuda_version%%.*}" = "$cuda_major" ] || fail "$wheel" "cuda_version=$cuda_version" "CUDA major $cuda_major" 'build the named CUDA variant and regenerate provenance'
        fi
        bytes=$(wc -c < "$wheel" | tr -d ' ')
        [ "$bytes" -le 1048576000 ] || fail "$wheel" "$bytes bytes" 'at most 1048576000 bytes (1000 MiB)' 'reduce the wheel size or arrange a different release strategy'
        checksum=$(sha256sum "$wheel"); checksum=${checksum%% *}
        jq -n --arg filename "$filename" --arg path "$wheel" --arg distribution "$distribution" --arg project_id "$project_id" --arg version "$wheel_version" --arg sha256 "$checksum" --argjson bytes "$bytes" '{filename:$filename,path:$path,distribution:$distribution,project_id:$project_id,version:$version,sha256:$sha256,bytes:$bytes}' >> "$scratch/entries"
    done < "$scratch/files"
    [ -s "$scratch/entries" ] || fail "$WHEEL_DIR" 'zero matching wheels' "$PYPI_PLATFORM" 'download the selected wheel artifacts and check the platform filter'
    jq -r '.[] | [.artifact_dir,(.cuda | split(".")[0])] | @tsv' "$scratch/cells" > "$scratch/expected-cells"
    while IFS="$(printf '\t')" read -r cell_dir cell_cuda; do
        if [ "$cell_dir" = . ]; then cell_prefix="${WHEEL_DIR%/}/"; else cell_prefix="${WHEEL_DIR%/}/$cell_dir/"; fi
        cell_count=$(jq -rs --arg prefix "$cell_prefix" --arg suffix "-cu$cell_cuda" '[.[] | select((.path | startswith($prefix)) and (.distribution | endswith($suffix)))] | length' "$scratch/entries")
        [ "$cell_count" -eq 1 ] || fail "$cell_prefix" "$cell_count selected CUDA $cell_cuda wheels" 'exactly one finished wheel per selected matrix cell' 'restore the missing cell artifact or separate extra build outputs'
    done < "$scratch/expected-cells"
    jq -es --slurpfile cells "$scratch/cells" --arg root "${WHEEL_DIR%/}/" 'length == ($cells[0] | length) and all(.[]; . as $wheel | any($cells[0][]; (if .artifact_dir == "." then $root else $root + .artifact_dir + "/" end) as $prefix | $wheel.path | startswith($prefix)))' "$scratch/entries" >/dev/null || fail "$WHEEL_DIR" 'extra wheel or wheel outside selected cell directories' 'only the confirmed publish cells' 'narrow the artifact download selection'
    jq -s --arg tag "$RELEASE_TAG" --arg commit "$RELEASE_COMMIT" --arg repository "$PYPI_PUBLISH_REPOSITORY" --arg source "$PYPI_ASSET_SOURCE" --arg plc "$PYPI_PER_RELEASE_PLC" '{template_version:"1.9.0",tag:$tag,commit:$commit,repository:$repository,asset_source:$source,per_release_plc:$plc,wheels:.}' "$scratch/entries" > "$manifest"
    {
        printf '| Wheel | Bytes | SHA256 |\n|---|---:|---|\n'
        jq -r '.wheels[] | "| \(.filename) | \(.bytes) | \(.sha256) |"' "$manifest"
        printf '\nApproval checklist (record in the approval comment):\n- [ ] IP self/peer review done\n- [ ] Artifacts and hashes above confirmed\n'
        [ "$PYPI_PER_RELEASE_PLC" != yes ] || printf '%s\n' '- [ ] L1 Feature PLC complete'
    } > "$scratch/summary"
    cat "$scratch/summary"
    if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then cat "$scratch/summary" >> "$GITHUB_STEP_SUMMARY"; fi
}

poll_release() {
    deadline=$(($(date +%s) + PYPI_TIMEOUT_SECONDS))
    while :; do
        kitmaker "$KITMAKER_API/status/$release_uuid"
        cp "$scratch/response" "pypi-status-$release_uuid.json"
        state=$(jq -er '.status' "$scratch/response") || fail "$release_uuid" 'missing status' 'status JSON' 'inspect the saved response'
        case "$state" in
            completed) return 0 ;;
            failed)
                printf 'Batch outcomes (unlisted wheels may be unstarted; inspect both indexes):\n'
                jq -r '.jobs[]? | "\(.status) \(.wheel_url)"' "$scratch/response"
                cat "$scratch/completed"
                fail "$release_uuid" 'failed or partially published batch' 'all wheels completed' 'inspect every wheel on both indexes; retain UUID; do not resubmit the batch' ;;
        esac
        [ "$(date +%s)" -lt "$deadline" ] || fail "$release_uuid" "$state at polling deadline" 'terminal completed state' 'inspect saved status and both indexes; do not resubmit or infer cancellation'
        printf 'POLL %s: %s; waiting %s seconds\n' "$release_uuid" "$state" "$PYPI_POLL_SECONDS"
        sleep "$PYPI_POLL_SECONDS"
    done
}
submit_projects() {
    jq -r '.wheels[].project_id' "$manifest" | LC_ALL=C sort -u > "$scratch/ids"
    : > "$scratch/completed"
    while IFS= read -r id; do
        jq --arg id "$id" --arg pic "$KITMAKER_PIC" --argjson upload "$upload" '.wheels | map(select(.project_id == $id)) | {project_name:.[0].distribution,payload:map({pic:$pic,job_type:"wheel-release-job",url:.url,upload:$upload,mirroring_strategy:"upload-both"})}' "$scratch/publish-manifest" > "$scratch/request"
        kitmaker --request POST --data-binary "@$scratch/request" "$KITMAKER_API/projects/$id/releases"
        release_uuid=$(jq -er '.release_uuid | select(type == "string" and test("^[A-Za-z0-9-]+$"))' "$scratch/response") || fail "$id" 'missing release_uuid' 'release UUID' 'inspect response; do not resubmit an uncertain request'
        printf '%s %s %s\n' "$id" "$upload" "$release_uuid" >> pypi-release-uuids.txt
        poll_release
        jq -r --arg id "$id" --arg upload "$upload" '.wheels[] | select(.project_id == $id) | "COMPLETED upload=\($upload) \(.filename)"' "$manifest" >> "$scratch/completed"
        cat "$scratch/completed"
    done < "$scratch/ids"
}
verify_indexes() {
    deadline=$(($(date +%s) + PYPI_INDEX_TIMEOUT_SECONDS))
    while :; do
        missing=0
        while IFS="$(printf '\t')" read -r name version filename checksum; do
            http "https://pypi.org/pypi/$name/$version/json"
            if [ "$http_status" = 200 ] && [ "$http_rc" -eq 0 ] && jq -e --arg file "$filename" --arg sum "$checksum" 'any(.urls[]; .filename == $file and .digests.sha256 == $sum)' "$scratch/response" >/dev/null; then printf 'VERIFIED pypi.org %s %s\n' "$filename" "$checksum"; else missing=$((missing + 1)); printf 'WAIT pypi.org %s %s\n' "$filename" "$checksum"; cat "$scratch/response"; fi
            http "https://pypi.nvidia.com/$name/"
            if [ "$http_status" = 200 ] && [ "$http_rc" -eq 0 ] && grep -F "$filename#sha256=$checksum" "$scratch/response" >/dev/null; then printf 'VERIFIED pypi.nvidia.com %s %s\n' "$filename" "$checksum"; else missing=$((missing + 1)); printf 'WAIT pypi.nvidia.com %s %s\n' "$filename" "$checksum"; cat "$scratch/response"; fi
        done < "$scratch/expected"
        [ "$missing" -ne 0 ] || return 0
        [ "$(date +%s)" -lt "$deadline" ] || fail "$manifest" "$missing index entries missing or hash mismatch" 'every filename and SHA256 on both indexes' 'inspect indexes and saved UUIDs; do not resubmit'
        sleep "$PYPI_POLL_SECONDS"
    done
}
publish() {
    required PYPI_PUBLISH_REPOSITORY PYPI_ASSET_SOURCE RELEASE_TAG RELEASE_COMMIT
    repository_guard
    required KITMAKER_TOKEN KITMAKER_PIC
    for number in "$PYPI_POLL_SECONDS" "$PYPI_TIMEOUT_SECONDS" "$PYPI_INDEX_TIMEOUT_SECONDS"; do
        case "$number" in ''|*[!0-9]*) fail "$0" "$number" 'positive integer deadline/interval' 'correct polling configuration' ;; esac
        [ "$number" -gt 0 ] || fail "$0" "$number" 'positive polling value' 'correct polling configuration'
    done
    case "$PYPI_DRY_RUN_ONLY" in 0|1) ;; *) fail "$0" "$PYPI_DRY_RUN_ONLY" '0 or 1' 'set the dry-run switch explicitly' ;; esac
    [ -f "$manifest" ] || fail "$manifest" 'missing' 'manifest from successful prepare' 'download the prepare artifact'
    jq -e --arg tag "$RELEASE_TAG" --arg commit "$RELEASE_COMMIT" --arg repository "$PYPI_PUBLISH_REPOSITORY" --arg source "$PYPI_ASSET_SOURCE" '
        .template_version == "1.9.0" and .tag == $tag and .commit == $commit and .repository == $repository and .asset_source == $source and
        (.wheels | type == "array" and length > 0 and all(.[]; (.filename | test("^[A-Za-z0-9_.!+-]+\\.whl$")) and (.sha256 | test("^[a-f0-9]{64}$")) and (.project_id | test("^[1-9][0-9]*$"))))' "$manifest" >/dev/null || fail "$manifest" 'invalid manifest or tag/commit/repository/source mismatch' "$RELEASE_TAG $RELEASE_COMMIT $PYPI_PUBLISH_REPOSITORY $PYPI_ASSET_SOURCE" 'use the manifest from this release and template version'
    jq -r '.wheels[] | [.distribution,.version,.filename,.sha256] | @tsv' "$manifest" > "$scratch/expected"
    jq -r '.wheels[] | [.path,.filename,.sha256] | @tsv' "$manifest" > "$scratch/to-publish"
    : > "$scratch/urls"
    while IFS="$(printf '\t')" read -r wheel filename checksum; do
        tag_uri=$(printf '%s' "$RELEASE_TAG" | jq -sRr @uri)
        file_uri=$(printf '%s' "$filename" | jq -sRr @uri)
        url="https://github.com/$PYPI_PUBLISH_REPOSITORY/releases/download/$tag_uri/$file_uri"
        http "$url"
        if [ "$http_status" = 404 ]; then fail "$filename" 'HTTP 404' 'public GitHub release asset' "a draft release isn't readable; publish the draft before approval"; fi
        [ "$http_rc" -eq 0 ] || fail "$filename" "HTTP $http_status curl=$http_rc" 'anonymous asset download' 'fix release asset visibility or URL'
        actual=$(sha256sum "$scratch/response"); actual=${actual%% *}
        [ "$actual" = "$checksum" ] || fail "$filename" "$actual" "$checksum" 'restore the exact immutable artifact; regenerate approval after any change'
        jq -n --arg filename "$filename" --arg url "$url" '{filename:$filename,url:$url}' >> "$scratch/urls"
    done < "$scratch/to-publish"
    jq --slurpfile urls "$scratch/urls" '.wheels |= map(. as $wheel | . + ($urls[] | select(.filename == $wheel.filename)))' "$manifest" > "$scratch/publish-manifest"
    upload=false
    submit_projects
    if [ "$PYPI_DRY_RUN_ONLY" = 1 ]; then printf 'DRY RUN complete; no Kitmaker upload submitted.\n'; return 0; fi
    upload=true
    submit_projects
    verify_indexes
}
case "$mode" in prepare) prepare ;; publish) publish ;; esac
