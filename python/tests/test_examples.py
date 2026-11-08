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

"""The self-contained Python examples run (the ones that need datasets or a viewer are left out),
so the examples the docs point to cannot rot."""

import os
import pathlib
import subprocess
import sys

import pytest

import pycunls

EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"
if not EXAMPLES.is_dir():
    pytest.skip("source tree not available", allow_module_level=True)

CASES = {
    "model_fitting.py": ["--quick"],
    "pose_graph_optimization.py": [],
    "sparse_bundle_adjustment.py": [],
    "imu_bundle_adjustment.py": [],
    "ransac_pnp.py": [],
    "custom_warp_factor.py": [],
    "custom_warp_state.py": [],
}


@pytest.mark.parametrize("script", sorted(CASES))
def test_example_runs(script, tmp_path):
    # The pycunls this test imports, not the source tree's (which has no native module).
    package_parent = os.path.dirname(os.path.dirname(os.path.abspath(pycunls.__file__)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([package_parent, str(EXAMPLES)]))
    out = subprocess.run([sys.executable, str(EXAMPLES / script)] + CASES[script], env=env,
                         cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
