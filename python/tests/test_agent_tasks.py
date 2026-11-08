# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Every agent task is solvable with the documented API (the reference passes) and has a real
trap (the naive answer fails), for several seeds."""

import json
import os
import pathlib
import subprocess
import sys

import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # agent_tasks
from agent_tasks import TASKS  # noqa: E402

SEEDS = (0, 1, 2)


@pytest.mark.parametrize("name", sorted(TASKS))
@pytest.mark.parametrize("seed", SEEDS)
def test_reference_passes(name, seed):
    task = TASKS[name]
    data, truth = task.make(seed)
    passed, message = task.check(task.reference(data), truth)
    assert passed, message


@pytest.mark.parametrize("name", sorted(TASKS))
def test_naive_answer_fails(name):
    task = TASKS[name]
    failures = 0
    for seed in SEEDS:
        data, truth = task.make(seed)
        passed, _ = task.check(task.naive(data), truth)
        failures += not passed
    # inadequate_model's naive answer is right when the model happens to be adequate.
    assert failures >= (1 if name == "inadequate_model" else len(SEEDS))


def test_malformed_answers_get_a_message():
    task = TASKS["decay_curve"]
    _, truth = task.make(0)
    assert task.check({}, truth) == (False, "the answer needs 'params': numbers of shape (3,)")
    assert not task.check({"params": [1, 2]}, truth)[0]


def test_command_line_round_trip(tmp_path):
    root = pathlib.Path(__file__).resolve().parents[1]
    # The pycunls this test imports first, then the agent_tasks package.
    import pycunls
    package_parent = os.path.dirname(os.path.dirname(os.path.abspath(pycunls.__file__)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([package_parent, str(root)]))
    run = [sys.executable, "-m", "agent_tasks"]
    subprocess.run(run + ["prepare", "decay_curve", str(tmp_path), "--seed", "3"], env=env,
                   check=True, capture_output=True, cwd=tmp_path)
    assert (tmp_path / "data.npz").exists() and (tmp_path / "PROMPT.md").exists()
    subprocess.run(run + ["reference", "decay_curve", str(tmp_path), "--seed", "3"], env=env,
                   check=True, capture_output=True, cwd=tmp_path)
    out = subprocess.run(run + ["check", "decay_curve", str(tmp_path / "answer.json"), "--seed",
                                "3"], env=env, capture_output=True, text=True, cwd=tmp_path)
    assert out.returncode == 0 and json.loads(out.stdout)["passed"]


def _run_eval(tmp_path, agent_cmd, out_dir):
    root = pathlib.Path(__file__).resolve().parents[2]
    import pycunls
    package_parent = os.path.dirname(os.path.dirname(os.path.abspath(pycunls.__file__)))
    env = dict(os.environ, AGENT_CMD=agent_cmd, TASKS="decay_curve", SEEDS="0", TIMEOUT="300",
               PYTHON=sys.executable, PYTHONPATH=package_parent)
    proc = subprocess.run(["bash", str(root / "scripts" / "run_agent_eval.sh"), str(out_dir)],
                          env=env, capture_output=True, text=True, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    rows = [json.loads(line, parse_constant=lambda c: pytest.fail(f"non-strict JSON {c}"))
            for line in (out_dir / "results.jsonl").read_text().splitlines()]
    return rows


@pytest.mark.skipif(not (pathlib.Path(__file__).resolve().parents[2] / "scripts" /
                         "run_agent_eval.sh").exists(), reason="source tree not available")
def test_eval_runner_with_the_installed_package(tmp_path):
    # Plumbing: the reference answers pass with the installed pycunls.
    rows = _run_eval(tmp_path, "reference", tmp_path / "ref")
    assert [r["passed"] for r in rows] == [True]

    # An agent that answers nothing fails, even with a stale passing answer from an earlier run;
    # it imports the installed pycunls but not the hidden task package.
    stale = tmp_path / "ref" / "decay_curve-0" / "answer.json"
    assert stale.exists()
    stub = tmp_path / "stub_agent.py"
    stub.write_text("import importlib.util, pathlib, pycunls\n"
                    "pathlib.Path('seen.txt').write_text(str(pycunls._pycunls_core.__name__) + ' '"
                    " + str(importlib.util.find_spec('agent_tasks') is None))\n")
    rows = _run_eval(tmp_path, f"{sys.executable} {stub}", tmp_path / "ref")
    assert [r["passed"] for r in rows] == [False]
    assert (tmp_path / "ref" / "decay_curve-0" / "seen.txt").read_text() == \
        "pycunls._pycunls_core True"
