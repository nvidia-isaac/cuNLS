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


# ---- a broken install: the CLI still reports what is broken ---------------------------------

import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402


def _run(args, pythonpath, cwd):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(p) for p in pythonpath))
    return subprocess.run([sys.executable] + args, env=env, cwd=cwd, capture_output=True,
                          text=True)


def _installed_parent():
    return pathlib.Path(pycunls.__file__).resolve().parents[1]


def _without_native_module(tmp_path):
    """A copy of the pycunls package without its compiled extension."""
    src = pathlib.Path(pycunls.__file__).resolve().parent
    dst = tmp_path / "broken" / "pycunls"
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("_pycunls_core*.so", "__pycache__"))
    return dst.parent


def _fake_module(tmp_path, name):
    root = tmp_path / f"no_{name}"
    (root / name).mkdir(parents=True)
    (root / name / "__init__.py").write_text(f"raise ImportError('simulated: no {name}')\n")
    return root


def _doctor_json(proc):
    report = json.loads(proc.stdout, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    return report, {c["check"]: c for c in report["checks"]}


def test_doctor_reports_a_missing_native_module(tmp_path):
    broken = _without_native_module(tmp_path)
    proc = _run(["-m", "pycunls", "doctor", "--json"], [broken], tmp_path)
    assert proc.returncode == 1, proc.stderr
    report, checks = _doctor_json(proc)
    assert not report["ok"]
    assert not checks["native module"]["ok"] and "_pycunls_core" in checks["native module"]["detail"]
    assert "reinstall" in checks["native module"]["fix"]
    assert checks["solve"]["detail"].startswith("skipped: needs")
    # The guide and the catalog need no native module; a plain import keeps the original error.
    assert _run(["-m", "pycunls", "guide"], [broken], tmp_path).returncode == 0
    described = _run(["-m", "pycunls", "describe", "SE3BetweenFactorBatch"], [broken], tmp_path)
    assert described.returncode == 0 and "T_right^-1" in described.stdout
    plain = _run(["-c", "import pycunls"], [broken], tmp_path)
    assert plain.returncode != 0 and "_pycunls_core" in plain.stderr


def test_doctor_reports_missing_cupy(tmp_path):
    proc = _run(["-m", "pycunls", "doctor", "--json"],
                [_fake_module(tmp_path, "cupy"), _installed_parent()], tmp_path)
    assert proc.returncode == 1, proc.stderr
    report, checks = _doctor_json(proc)
    assert not checks["gpu"]["ok"] and "simulated: no cupy" in checks["gpu"]["detail"]
    assert "cupy-cuda12x" in checks["gpu"]["fix"]
    assert checks["solve"]["detail"].startswith("skipped: needs gpu")


def test_missing_optional_warp_is_not_a_failure(tmp_path):
    proc = _run(["-m", "pycunls", "doctor", "--json"],
                [_fake_module(tmp_path, "warp"), _installed_parent()], tmp_path)
    report, checks = _doctor_json(proc)
    assert not checks["warp"]["ok"] and not checks["warp"]["required"]
    assert report["ok"] and proc.returncode == 0, proc.stdout
