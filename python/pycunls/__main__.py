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

"""Command line of pycunls.

    python -m pycunls guide              # the agent guide (how to use pycunls, conventions, fixes)
    python -m pycunls doctor [--json]    # check the GPU, CUDA, CuPy, Warp and a small solve
    python -m pycunls describe NAME      # one built-in state or factor batch
    python -m pycunls catalog [QUERY]    # list the built-in state and factor batches
"""

import argparse
import json
import pathlib
import platform
import sys
from typing import Callable, List, Tuple


def _guide() -> str:
    return (pathlib.Path(__file__).with_name("AGENT_GUIDE.md")).read_text()


def _check(name: str, fn: Callable[[], str], fix: str, required: bool = True):
    try:
        return {"check": name, "ok": True, "detail": fn(), "required": required}
    except Exception as exc:  # noqa: BLE001 -- report every failure
        return {"check": name, "ok": False, "detail": f"{type(exc).__name__}: {exc}",
                "fix": fix, "required": required}


def doctor() -> Tuple[bool, List[dict]]:
    """Runs the environment checks; returns (all required checks passed, the results)."""
    results = []

    def python_version():
        if sys.version_info < (3, 10):
            raise RuntimeError(f"Python {platform.python_version()} < 3.10")
        return platform.python_version()

    def cupy_gpu():
        import cupy as cp
        dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        return (f"CuPy {cp.__version__}, GPU {dev.id}: {name} (sm_{props['major']}{props['minor']}"
                f"), driver {cp.cuda.runtime.driverGetVersion()}, runtime "
                f"{cp.cuda.runtime.runtimeGetVersion()}")

    def native():
        import pycunls
        from pycunls import _pycunls_core
        return f"pycunls {pycunls.__version__} ({pathlib.Path(_pycunls_core.__file__).name})"

    def solve():
        import cupy as cp
        import pycunls
        x = cp.asarray([[5.0, -3.0, 2.0]], dtype=cp.float32)
        target = cp.asarray([[1.0, 2.0, 3.0]], dtype=cp.float32)
        states = pycunls.VectorStateBatch3(x, 1)
        states.set_num_active_states(1)
        prior = pycunls.PriorVectorFactorBatch3(target, 1)
        prior.set_num_active_factors(1)
        problem = pycunls.Problem()
        problem.add_state_batch(states)
        problem.add_factor_batch(prior, [states.state_device_ptr(0)])
        summary = pycunls.LevenbergMarquardtMinimizer().minimize(pycunls.CudaStream(), problem)
        if not summary.converged() or float(cp.abs(x - target).max()) > 1e-3:
            raise RuntimeError(summary.message)
        return f"{summary.num_iterations} iterations, {summary.solve_time_ms:.1f} ms"

    def fit():
        import numpy as np
        import pycunls
        xs = np.linspace(0, 2, 20)
        res = pycunls.fit(lambda x, p: p[:, 0] * x + p[:, 1], xs, 2 * xs - 1, p0=[0, 0])
        if not np.allclose(res.params, [2, -1], atol=1e-3):
            raise RuntimeError(f"fit returned {res.params}")
        return "line fit ok"

    def warp():
        import warp as wp
        return f"warp {wp.__version__} (custom Warp factors and Warp fit() models)"

    def cudss():
        import cupy as cp
        import pycunls
        x = cp.asarray([[5.0, -3.0, 2.0]], dtype=cp.float32)
        states = pycunls.VectorStateBatch3(x, 1)
        states.set_num_active_states(1)
        prior = pycunls.PriorVectorFactorBatch3(cp.zeros((1, 3), cp.float32), 1)
        prior.set_num_active_factors(1)
        problem = pycunls.Problem()
        problem.add_state_batch(states)
        problem.add_factor_batch(prior, [states.state_device_ptr(0)])
        options = pycunls.MinimizerOptions()
        options.sparse_linear_solver_type = pycunls.SparseLinearSolverType.cuDSS
        pycunls.GaussNewtonMinimizer(options).minimize(pycunls.CudaStream(), problem)
        return "cuDSS solver available"

    results.append(_check("python", python_version, "use Python 3.10 or newer"))
    results.append(_check("gpu", cupy_gpu, "install a CUDA driver and the CuPy build for your "
                          "CUDA version (pip install cupy-cuda12x / cupy-cuda13x); check "
                          "nvidia-smi and CUDA_VISIBLE_DEVICES"))
    results.append(_check("native module", native, "reinstall the pycunls wheel built for your "
                          "CUDA version"))
    results.append(_check("solve", solve, "run with CUNLS_LOG_LEVEL=debug and report the output"))
    results.append(_check("fit", fit, "check the solve result above"))
    results.append(_check("warp", warp, "pip install 'pycunls[warp]' (optional: custom Warp "
                          "factors)", required=False))
    results.append(_check("cudss", cudss, "optional: set the library path so libcudss.so is "
                          "found (SparseLinearSolverType.cuDSS only)", required=False))
    ok = all(r["ok"] for r in results if r["required"])
    return ok, results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pycunls", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("guide", help="print the agent guide")
    d = sub.add_parser("doctor", help="check the environment")
    d.add_argument("--json", action="store_true", help="one JSON object instead of text")
    s = sub.add_parser("describe", help="describe a built-in state or factor batch")
    s.add_argument("name")
    c = sub.add_parser("catalog", help="list the built-in state and factor batches")
    c.add_argument("query", nargs="?")
    args = parser.parse_args(argv)

    if args.command == "guide":
        print(_guide())
        return 0
    if args.command == "describe":
        import pycunls
        try:
            print(pycunls.describe(args.name))
        except KeyError as exc:
            print(exc.args[0], file=sys.stderr)
            return 1
        return 0
    if args.command == "catalog":
        import pycunls
        for entry in pycunls.catalog(query=args.query):
            print(f"{entry['name']:<46} {entry['kind']:<7} {entry.get('use', entry.get('layout'))}")
        return 0
    ok, results = doctor()
    if args.json:
        print(json.dumps({"ok": ok, "checks": results}))
    else:
        for r in results:
            mark = "ok  " if r["ok"] else ("FAIL" if r["required"] else "skip")
            print(f"[{mark}] {r['check']:<14} {r['detail']}")
            if not r["ok"]:
                print(f"       fix: {r['fix']}")
        print("pycunls is ready." if ok else "pycunls is NOT ready: see the failed checks.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
