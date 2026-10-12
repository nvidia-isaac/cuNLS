# Configure cuNLS Python publishing

Author: Heron Ordonez Guillen

Run these commands after the workflow is merged into the canonical cuNLS repository.
Credentials come from the operator environment, never from this document. Public binary distribution was
confirmed legally approved by the owner on 2026-10-12.

```bash
gh auth status
gh repo view nvidia-isaac/cuNLS --json nameWithOwner,visibility
gh api user --jq .login
gh api users/alexkorovko --jq .login
gh api repos/nvidia-isaac/cuNLS/actions/runners --paginate \
  --jq '.runners[] | {name,status,labels:[.labels[].name]}'
gh api repos/nvidia-isaac/cuNLS/actions/permissions/selected-actions
```

MANUAL: In GitHub → nvidia-isaac/cuNLS → Settings → Environments, create or
edit the existing `pypi` environment. Set `hguillen` as a required reviewer,
enable prevent self-review and disable administrator bypass. Restrict deployment
to tags matching `v*`. The owner requested a single initial reviewer; add
`alexkorovko` when ready. With one reviewer and self-review prevented, another
trusted account must initiate a release that this reviewer approves.

Verify protection before storing the environment credential.

```bash
gh api repos/nvidia-isaac/cuNLS/environments/pypi \
  --jq '{protection_rules,deployment_branch_policy}'
gh api repos/nvidia-isaac/cuNLS/environments/pypi/deployment-branch-policies \
  --jq '.branch_policies[] | {name,type}'
gh secret list --repo nvidia-isaac/cuNLS --env pypi
```

`KITMAKER_TOKEN` is already present in the `pypi` environment (verified 2026-10-12).
Keep that scope. To rotate it from approved credential storage, use this
idempotent command; it does not display the value.

```bash
: "${KITMAKER_TOKEN:?Load the Kitmaker service token from approved credential storage}"
printf '%s' "$KITMAKER_TOKEN" | gh secret set KITMAKER_TOKEN \
  --repo nvidia-isaac/cuNLS --env pypi
```

MANUAL: Confirm the existing `[self-hosted, gpu]` runner pool has Kitmaker reachability, NVIDIA CA trust,
and POSIX sh, curl with `--fail-with-body`, jq, unzip and sha256sum. The first
protected dry run verifies reachability from the selected runner.
Portal IDs 6865/6866, names `pycunls-cu12`/`pycunls-cu13` and owner
`hguillen@nvidia.com` were verified through the read-only Portal API on 2026-10-12.

Configure nonsecret variables; keep publication disabled during setup.

```bash
gh variable set PYPI_PUBLISH_RUNNER --repo nvidia-isaac/cuNLS \
  --body '["self-hosted","gpu"]'
gh variable set PYPI_PUBLISH_ENABLED --repo nvidia-isaac/cuNLS --body 'false'
gh variable set PYPI_DRY_RUN_ONLY --repo nvidia-isaac/cuNLS --body '1'
gh variable set PYPI_PER_RELEASE_PLC --repo nvidia-isaac/cuNLS --body 'yes'
```

MANUAL: The per-release L1 Feature PLC policy is pending owner confirmation. The
workflow includes the checkbox by default. Set `PYPI_PER_RELEASE_PLC=no` only
when the owner confirms it is not required. Each approval records IP self/peer
review completion and the selected filenames and SHA256 hashes.

## Verify setup and the first dry run

```bash
gh secret list --repo nvidia-isaac/cuNLS --env pypi
gh variable list --repo nvidia-isaac/cuNLS
gh api repos/nvidia-isaac/cuNLS/environments/pypi
gh api repos/nvidia-isaac/cuNLS/environments/pypi/deployment-branch-policies
```

MANUAL: Confirm the secret value and scope in approved credential storage, the
reviewer protections in Settings → Environments → pypi, and runner registration
in Settings → Actions → Runners. Secret values are write-only.

After setup verification and owner approval, create the first tag from merged
`main` in dry-run mode. The input gate verified `v0.2.0` and both index versions
were absent on 2026-10-12; recheck before creating a tag. Tag creation triggers
the release workflow automatically. Run this block once for the first release;
use publisher-only retries after the producer has succeeded.

```bash
gh variable set PYPI_DRY_RUN_ONLY --repo nvidia-isaac/cuNLS --body '1'
gh variable set PYPI_PUBLISH_ENABLED --repo nvidia-isaac/cuNLS --body 'true'
RELEASE_COMMIT=$(gh api repos/nvidia-isaac/cuNLS/commits/main --jq .sha)
gh api --method POST repos/nvidia-isaac/cuNLS/git/refs \
  -f ref=refs/tags/v0.2.0 -f sha="$RELEASE_COMMIT"
RELEASE_RUN_ID=$(gh run list --repo nvidia-isaac/cuNLS --workflow release.yml \
  --branch v0.2.0 --event push --limit 1 --json databaseId \
  --jq '.[0].databaseId')
: "${RELEASE_RUN_ID:?No tag-triggered release run found}"
gh run watch "$RELEASE_RUN_ID" --repo nvidia-isaac/cuNLS --exit-status
PUBLICATION_ATTEMPT=$(gh api "repos/nvidia-isaac/cuNLS/actions/runs/$RELEASE_RUN_ID" --jq .run_attempt)
gh run download "$RELEASE_RUN_ID" --repo nvidia-isaac/cuNLS \
  --name "pypi-publication-record-$PUBLICATION_ATTEMPT" --dir "pypi-publication-record-$PUBLICATION_ATTEMPT"
```

MANUAL: Approve the waiting `pypi` deployment after reviewing the prepare summary.
Expect all build/test cells, the public GitHub release and offline manifest to
succeed. Expect each Kitmaker dry run to reach `completed`, with no upload
request; retain the UUIDs, status JSON and manifest. This does not verify index
writes or customer installation.

After the owner authorizes live publication, rerun only `pypi-publish`, preserving
the original tag, commit, release bytes and manifest. Job database IDs come from
`gh run view`; use the [GitHub CLI job rerun command](https://cli.github.com/manual/gh_run_rerun).

```bash
gh variable set PYPI_DRY_RUN_ONLY --repo nvidia-isaac/cuNLS --body '0'
PUBLISH_JOB_ID=$(gh run view "$RELEASE_RUN_ID" --repo nvidia-isaac/cuNLS \
  --json jobs --jq '.jobs[] | select(.name == "pypi-publish") | .databaseId')
: "${PUBLISH_JOB_ID:?Publisher job was not found}"
gh run rerun "$RELEASE_RUN_ID" --repo nvidia-isaac/cuNLS --job "$PUBLISH_JOB_ID"
gh run watch "$RELEASE_RUN_ID" --repo nvidia-isaac/cuNLS --exit-status
```

MANUAL: Verify another environment approval is required and the rerun logs show
`PYPI_DRY_RUN_ONLY=0`. Confirm this behavior on the deployed GitHub instance
before live publication; a variable captured by the original run can leave it
in dry-run mode. Publication records are retained separately per run attempt.

MANUAL: After live publication, confirm every wheel filename and SHA256 on both
PyPI and NVIDIA's public index, then test installation on each supported CUDA
variant. A partial failure requires separate review of the missing artifacts;
never resubmit the complete batch automatically.
