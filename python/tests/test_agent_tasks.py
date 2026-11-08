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
