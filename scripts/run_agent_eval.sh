#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Runs an agent on the cuNLS agent tasks (python/agent_tasks) and scores its answers.
#
# For each task and seed: prepare a fresh directory with data.npz and PROMPT.md, run the agent
# command there with the prompt as its last argument, then check answer.json against the hidden
# truth. Results: one JSON line per run in OUT_DIR/results.jsonl, a pass table at the end, and the
# agent's output in OUT_DIR/<task>-<seed>/transcript.txt.
#
# Not run in CI: an agent costs tokens. Compare pass rates with and without a change.
#
# Usage:
#   AGENT_CMD="claude -p --permission-mode acceptEdits" scripts/run_agent_eval.sh [OUT_DIR]
#   AGENT_CMD=reference scripts/run_agent_eval.sh /tmp/eval    # plumbing check: reference answers
#
# Environment:
#   AGENT_CMD  command that receives the prompt as its last argument (required); "reference"
#              writes the reference solution instead of running an agent
#   TASKS      space-separated task names (default: all; see python -m agent_tasks list)
#   SEEDS      space-separated seeds (default: "0 1 2")
#   TIMEOUT    seconds per run (default: 900)
#   PYTHON     Python with pycunls installed (default: python3)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-${REPO_DIR}/agent_eval_$(date +%Y%m%d_%H%M%S)}"
PYTHON="${PYTHON:-python3}"
SEEDS="${SEEDS:-0 1 2}"
TIMEOUT="${TIMEOUT:-900}"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${REPO_DIR}/python"

if [ -z "${AGENT_CMD:-}" ]; then
  echo "Set AGENT_CMD to the agent command (it gets the prompt as its last argument), e.g." >&2
  echo "  AGENT_CMD=\"claude -p --permission-mode acceptEdits\" $0 ${OUT_DIR}" >&2
  echo "or AGENT_CMD=reference to check the plumbing with the reference solutions." >&2
  exit 2
fi
if ! "${PYTHON}" -c "import pycunls" 2>/dev/null; then
  echo "${PYTHON} cannot import pycunls. Build and install the wheel first:" >&2
  echo "  ./scripts/build_pycunls_in_docker.sh && pip install dist/pycunls-*.whl" >&2
  exit 2
fi
TASKS="${TASKS:-$("${PYTHON}" -c 'from agent_tasks import TASKS; print(" ".join(TASKS))')}"

mkdir -p "${OUT_DIR}"
RESULTS="${OUT_DIR}/results.jsonl"
: > "${RESULTS}"
for task in ${TASKS}; do
  for seed in ${SEEDS}; do
    run_dir="${OUT_DIR}/${task}-${seed}"
    "${PYTHON}" -m agent_tasks prepare "${task}" "${run_dir}" --seed "${seed}" > /dev/null
    start=$(date +%s)
    if [ "${AGENT_CMD}" = "reference" ]; then
      "${PYTHON}" -m agent_tasks reference "${task}" "${run_dir}" --seed "${seed}" \
        > "${run_dir}/transcript.txt" 2>&1 || true
    else
      # shellcheck disable=SC2086  # AGENT_CMD is a command line, split on purpose
      (cd "${run_dir}" && timeout "${TIMEOUT}" ${AGENT_CMD} "$(cat PROMPT.md)") \
        > "${run_dir}/transcript.txt" 2>&1 || true
    fi
    seconds=$(( $(date +%s) - start ))
    result=$("${PYTHON}" -m agent_tasks check "${task}" "${run_dir}/answer.json" --seed "${seed}" \
             || true)
    echo "${result%\}}, \"seconds\": ${seconds}}" | tee -a "${RESULTS}"
  done
done

"${PYTHON}" - "${RESULTS}" <<'PY'
import collections, json, sys
rows = [json.loads(line) for line in open(sys.argv[1])]
by_task = collections.defaultdict(list)
for row in rows:
    by_task[row["task"]].append(row["passed"])
print(f"\n{'task':<22} passed")
for task, passed in by_task.items():
    print(f"{task:<22} {sum(passed)}/{len(passed)}")
print(f"{'total':<22} {sum(r['passed'] for r in rows)}/{len(rows)}")
PY
