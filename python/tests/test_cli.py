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

"""python -m pycunls: guide, doctor, describe, catalog (the help an agent finds without the repo)."""

import json
import pathlib

import pycunls
from pycunls.__main__ import doctor, main


def test_guide_ships_with_the_package(capsys):
    assert main(["guide"]) == 0
    text = capsys.readouterr().out
    assert "pycunls.fit" in text and "ProblemBuilder" in text and "T_right^-1 T_left" in text


def test_guide_names_only_existing_api():
    text = (pathlib.Path(pycunls.__file__).with_name("AGENT_GUIDE.md")).read_text()
    import re
    for name in set(re.findall(r"pycunls\.(\w+)", text)):
        if name == "X":  # the placeholder in help(pycunls.X)
            continue
        assert hasattr(pycunls, name) or name in ("warp", "mpc"), name


def test_doctor_passes_on_a_working_install(capsys):
    ok, results = doctor()
    assert ok, results
    assert main(["doctor", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] and {r["check"] for r in report["checks"]} >= {"gpu", "solve", "fit"}


def test_describe_and_catalog(capsys):
    assert main(["describe", "PnPFactorBatch"]) == 0
    assert "points_world" in capsys.readouterr().out
    assert main(["describe", "PnP"]) == 1
    assert main(["catalog", "between"]) == 0
    assert "SE3BetweenFactorBatch" in capsys.readouterr().out
