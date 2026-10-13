# cuNLS Python publishing

Author: Heron Ordonez Guillen

`release.yml` builds four x86_64 wheels for `pycunls-cu12` and `pycunls-cu13`,
attaches them to a public GitHub release, and prepares a manifest containing
their filenames, source commit and SHA256 hashes. The protected `pypi-publish`
job submits the release assets to Kitmaker projects 6865 and 6866 with owner
`hguillen@nvidia.com`. Kitmaker publishes to PyPI and NVIDIA's public index.

## GitHub configuration

Credentials come from the operator environment. Inspect the repository,
runner pool and environment protection with:

```bash
gh auth status
gh repo view nvidia-isaac/cuNLS --json nameWithOwner,visibility
gh api repos/nvidia-isaac/cuNLS/actions/runners --paginate \
  --jq '.runners[] | {name,status,labels:[.labels[].name]}'
gh api repos/nvidia-isaac/cuNLS/actions/permissions/selected-actions
gh api repos/nvidia-isaac/cuNLS/environments/pypi \
  --jq '{protection_rules,deployment_branch_policy}'
gh api repos/nvidia-isaac/cuNLS/environments/pypi/deployment-branch-policies \
  --jq '.branch_policies[] | {name,type}'
gh secret list --repo nvidia-isaac/cuNLS --env pypi
gh variable list --repo nvidia-isaac/cuNLS
```

In Settings → Environments → `pypi`, configure required reviewers, prevent
self-review, disable administrator bypass, and restrict deployment to tags
matching `v*`. Reviewer membership is managed in GitHub. A release requires
approval by an account other than its initiator.

Store `KITMAKER_TOKEN` in the `pypi` environment. This command sets or rotates
the credential without displaying its value:

```bash
: "${KITMAKER_TOKEN:?Load the Kitmaker service token from approved credential storage}"
printf '%s' "$KITMAKER_TOKEN" | gh secret set KITMAKER_TOKEN \
  --repo nvidia-isaac/cuNLS --env pypi
```

The publisher uses `[self-hosted, gpu]` runners and requires Kitmaker access,
NVIDIA CA trust, POSIX sh, curl supporting `--fail-with-body`, jq, unzip and sha256sum.
The repository variables control runner selection, publication and approval:

```bash
gh variable set PYPI_PUBLISH_RUNNER --repo nvidia-isaac/cuNLS \
  --body '["self-hosted","gpu"]'
gh variable set PYPI_PUBLISH_ENABLED --repo nvidia-isaac/cuNLS --body 'false'
gh variable set PYPI_DRY_RUN_ONLY --repo nvidia-isaac/cuNLS --body '1'
gh variable set PYPI_PER_RELEASE_PLC --repo nvidia-isaac/cuNLS --body 'yes'
```

`PYPI_PUBLISH_ENABLED=true` enables the protected publisher. With
`PYPI_DRY_RUN_ONLY=1`, Kitmaker validates the assets without uploading them.
`PYPI_PER_RELEASE_PLC=yes` includes L1 Feature PLC completion in the approval
checklist. Each approval records IP self/peer review completion and the selected
filenames and SHA256 hashes.

## Release dry run

Run the tag commands from a clean checkout of merged `main`.
`python/pyproject.toml` and `python/pycunls/__init__.py` must contain the same
public version. The wheel build enforces agreement between these versions and
the release tag. Tag creation starts the release workflow.

```bash
gh variable set PYPI_DRY_RUN_ONLY --repo nvidia-isaac/cuNLS --body '1'
gh variable set PYPI_PUBLISH_ENABLED --repo nvidia-isaac/cuNLS --body 'true'
RELEASE_VERSION=$(sed -n 's/^version = "\([^" ]*\)"$/\1/p' python/pyproject.toml)
: "${RELEASE_VERSION:?Package version was not found}"
RELEASE_TAG="v$RELEASE_VERSION"
RELEASE_COMMIT=$(git rev-parse HEAD)
gh api --method POST repos/nvidia-isaac/cuNLS/git/refs \
  -f ref="refs/tags/$RELEASE_TAG" -f sha="$RELEASE_COMMIT"
RELEASE_RUN_ID=$(gh run list --repo nvidia-isaac/cuNLS --workflow release.yml \
  --branch "$RELEASE_TAG" --event push --limit 1 --json databaseId \
  --jq '.[0].databaseId')
: "${RELEASE_RUN_ID:?No tag-triggered release run found}"
gh run watch "$RELEASE_RUN_ID" --repo nvidia-isaac/cuNLS --exit-status
PUBLICATION_ATTEMPT=$(gh api "repos/nvidia-isaac/cuNLS/actions/runs/$RELEASE_RUN_ID" --jq .run_attempt)
gh run download "$RELEASE_RUN_ID" --repo nvidia-isaac/cuNLS \
  --name "pypi-publication-record-$PUBLICATION_ATTEMPT" --dir "pypi-publication-record-$PUBLICATION_ATTEMPT"
```

Review the prepare summary and approve the waiting `pypi` deployment.
A successful dry run includes all build/test cells, the public GitHub release,
the offline manifest and a `completed` Kitmaker validation for each package.
Publication records contain the UUIDs, status JSON and manifest.

## Live publication and recovery

After approving the dry-run results, set live mode and rerun only the publisher.
The tag, source commit, release assets and manifest stay fixed.

```bash
gh variable set PYPI_DRY_RUN_ONLY --repo nvidia-isaac/cuNLS --body '0'
PUBLISH_JOB_ID=$(gh run view "$RELEASE_RUN_ID" --repo nvidia-isaac/cuNLS \
  --json jobs --jq '.jobs[] | select(.name == "pypi-publish") | .databaseId')
: "${PUBLISH_JOB_ID:?Publisher job was not found}"
gh run rerun "$RELEASE_RUN_ID" --repo nvidia-isaac/cuNLS --job "$PUBLISH_JOB_ID"
gh run watch "$RELEASE_RUN_ID" --repo nvidia-isaac/cuNLS --exit-status
```

Approve the `pypi` deployment and check that the rerun logs show
`PYPI_DRY_RUN_ONLY=0`. A rerun that retains `1` performs validation only.
Publication records are retained separately per run attempt.

Verify every wheel filename and SHA256 on both indexes, then test installation
for each supported CUDA variant. Publication failure does not roll back uploaded
files. Inspect the batch UUIDs, status JSON and both indexes before retrying.
Review missing artifacts separately and never resubmit the complete batch automatically.
