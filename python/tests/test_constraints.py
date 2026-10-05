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

"""Constraint batches and AugmentedLagrangianMinimizer (augmented Lagrangian).

References are exact and dependency-free: random convex QPs are solved in
float64 by enumerating active sets (KKT system per set), and the
Hock-Schittkowski problems have known solutions.
"""

import itertools

import cupy as cp
import numpy as np
import pytest

import pycunls


class _StreamHandle:
    """Exposes a raw cudaStream_t handle through the CUDA stream protocol."""

    def __init__(self, handle):
        self.handle = handle

    def __cuda_stream__(self):
        return (0, self.handle)


def cupy_stream(handle):
    if hasattr(cp.cuda.Stream, "from_external"):  # CuPy >= 14
        return cp.cuda.Stream.from_external(_StreamHandle(handle))
    return cp.cuda.ExternalStream(handle)


def make_minimizer(kind):
    options = pycunls.MinimizerOptions()
    options.max_num_iterations = 50
    options.state_tolerance = 1e-12
    options.cost_tolerance = 1e-12
    options.sparse_linear_solver_type = pycunls.SparseLinearSolverType.DenseLDLT
    if kind == "gn":
        return pycunls.GaussNewtonMinimizer(options)
    lm = pycunls.LevenbergMarquardtMinimizerOptions()
    lm.base_options = options
    lm.relative_reduction_tolerance = 1e-12
    return pycunls.LevenbergMarquardtMinimizer(lm)


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["gn", "lm"])
def test_box_projection(stream, kind):
    target = np.array([2.0, -3.0, 0.5, 7.0, -4.0, 0.25], dtype=np.float32)
    lower = np.array([-1, -1, -1, -np.inf, -2, 0.5], dtype=np.float32)
    upper = np.array([1, 1, 1, 3, np.inf, np.inf], dtype=np.float32)
    x = cp.zeros(6, dtype=cp.float32)
    t, lo, hi = cp.asarray(target), cp.asarray(lower), cp.asarray(upper)
    states = pycunls.VectorStateBatch3(x, 2)
    states.set_num_active_states(2)
    prior = pycunls.PriorVectorFactorBatch3(t, 2)
    prior.set_num_active_factors(2)
    bounds = pycunls.BoundFactorBatch3(lo, hi, 2)
    bounds.set_num_active_factors(2)
    assert bounds.kind == pycunls.ConstraintKind.Inequality
    assert bounds.residuals_size == 6 and bounds.state_sizes() == [3]
    problem = pycunls.Problem()
    problem.add_state_batch(states)
    ptrs = [states.state_device_ptr(i) for i in range(2)]
    problem.add_factor_batch(prior, ptrs)
    problem.add_factor_batch(bounds, ptrs)

    solver = pycunls.AugmentedLagrangianMinimizer(make_minimizer(kind))
    summary = solver.minimize(stream, problem)
    assert summary.status == pycunls.AugmentedLagrangianMinimizerStatus.Converged
    assert summary.max_violation <= 1e-4
    np.testing.assert_allclose(cp.asnumpy(x), np.clip(target, lower, upper), atol=2e-4)


@pytest.mark.parametrize("kind", ["gn", "lm"])
def test_projected_state_bounds(stream, kind):
    """The same box through VectorStateBatch.set_bounds: a plain solve, exact,
    from an infeasible initial guess."""
    target = np.array([2.0, -3.0, 0.5, 7.0, -4.0, 0.25], dtype=np.float32)
    lower = np.array([-1, -1, -1, -np.inf, -2, 0.5], dtype=np.float32)
    upper = np.array([1, 1, 1, 3, np.inf, np.inf], dtype=np.float32)
    x = cp.asarray(np.array([5, 0, 0, 0, 0, 0], dtype=np.float32))
    t, lo, hi = cp.asarray(target), cp.asarray(lower), cp.asarray(upper)
    states = pycunls.VectorStateBatch3(x, 2)
    states.set_num_active_states(2)
    assert not states.has_bounds
    states.set_bounds(lo, hi)
    assert states.has_bounds
    prior = pycunls.PriorVectorFactorBatch3(t, 2)
    prior.set_num_active_factors(2)
    problem = pycunls.Problem()
    problem.add_state_batch(states)
    problem.add_factor_batch(prior, [states.state_device_ptr(i) for i in range(2)])
    make_minimizer(kind).minimize(stream, problem)
    # LM stops when its float32 cost decrease vanishes, a little before GN's exact step.
    atol = 1e-5 if kind == "gn" else 1e-3
    np.testing.assert_allclose(cp.asnumpy(x), np.clip(target, lower, upper), atol=atol)
    states.set_bounds(None, None)
    assert not states.has_bounds
    with pytest.raises(ValueError):
        states.set_bounds(lo, None)


# ---------------------------------------------------------------------------
# Random convex QPs, batched (one QP per subproblem)
# ---------------------------------------------------------------------------


def solve_qp_reference(P, q, G, h, A, b):
    """min ½xᵀPx + qᵀx s.t. Gx <= h, Ax = b, by active-set enumeration (float64)."""
    n = P.shape[0]
    best = None
    for k in range(G.shape[0] + 1):
        for active in itertools.combinations(range(G.shape[0]), k):
            C = np.vstack([A, G[list(active)]])
            d = np.concatenate([b, h[list(active)]])
            if np.linalg.matrix_rank(C) < C.shape[0]:
                continue
            kkt = np.block([[P, C.T], [C, np.zeros((C.shape[0], C.shape[0]))]])
            sol = np.linalg.solve(kkt, np.concatenate([-q, d]))
            x, mult = sol[:n], sol[n:]
            if np.all(G @ x <= h + 1e-9) and np.all(mult[A.shape[0]:] >= -1e-9):
                best = x
                break
        if best is not None:
            break
    assert best is not None, "reference QP solver found no KKT point"
    return best


@pytest.mark.parametrize("kind", ["gn", "lm"])
def test_random_convex_qps_match_reference(stream, kind):
    rng = np.random.default_rng(3)
    batch, n, m_in, m_eq = 32, 3, 4, 1
    qps = []
    for _ in range(batch):
        M = rng.normal(size=(n, n))
        P = M.T @ M + 0.5 * np.eye(n)
        q = rng.normal(size=n) * 3.0
        x0 = rng.normal(size=n)  # a feasible point
        G = rng.normal(size=(m_in, n))
        h = G @ x0 + rng.uniform(0.0, 1.0, size=m_in)
        A = rng.normal(size=(m_eq, n))
        b = A @ x0
        qps.append((P, q, G, h, A, b))
    expected = np.stack([solve_qp_reference(*qp) for qp in qps])

    # ½xᵀPx + qᵀx = ½‖S(x - c)‖² + const with SᵀS = P and c = -P⁻¹q.
    centers = np.stack([-np.linalg.solve(P, q) for P, q, *_ in qps]).astype(np.float32)
    sqrt_info = np.stack([np.linalg.cholesky(P).T for P, *_ in qps]).astype(np.float32)
    normals_in = np.concatenate([G for _, _, G, *_ in qps]).astype(np.float32)
    offsets_in = np.concatenate([h for *_, h, _, _ in qps]).astype(np.float32)
    normals_eq = np.concatenate([A for *_, A, _ in qps]).astype(np.float32)
    offsets_eq = np.concatenate([b for *_, b in qps]).astype(np.float32)

    x = cp.zeros(batch * n, dtype=cp.float32)
    buffers = [cp.asarray(a.reshape(-1)) for a in
               (centers, sqrt_info, normals_in, offsets_in, normals_eq, offsets_eq)]
    c_buf, s_buf, gn_buf, go_buf, an_buf, ao_buf = buffers
    states = pycunls.VectorStateBatch3(x, batch)
    states.set_num_active_states(batch)
    prior = pycunls.PriorVectorFactorBatch3(c_buf, batch)
    prior.set_num_active_factors(batch)
    objective = pycunls.InformationFactorBatch(prior, s_buf)
    halfspaces = pycunls.HalfspaceFactorBatch3(gn_buf, go_buf, batch * m_in)
    halfspaces.set_num_active_factors(batch * m_in)
    inequalities = pycunls.ConstraintFactorBatch(halfspaces, pycunls.ConstraintKind.Inequality)
    hyperplanes = pycunls.HalfspaceFactorBatch3(an_buf, ao_buf, batch * m_eq)
    hyperplanes.set_num_active_factors(batch * m_eq)
    equalities = pycunls.ConstraintFactorBatch(hyperplanes, pycunls.ConstraintKind.Equality)

    problem = pycunls.Problem()
    problem.add_state_batch(states)
    ptr = [states.state_device_ptr(s) for s in range(batch)]
    problem.add_factor_batch(objective, ptr)
    problem.add_factor_batch(inequalities, [p for p in ptr for _ in range(m_in)])
    problem.add_factor_batch(equalities, [p for p in ptr for _ in range(m_eq)])
    problem.set_problem_partition(batch, [cp.arange(batch, dtype=cp.int32)])

    solver = pycunls.AugmentedLagrangianMinimizer(make_minimizer(kind))
    summary = solver.minimize(stream, problem)
    assert summary.status == pycunls.AugmentedLagrangianMinimizerStatus.Converged, summary
    assert summary.num_converged == batch
    assert summary.max_violation <= 1e-4
    result = cp.asnumpy(x).reshape(batch, n)
    np.testing.assert_allclose(result, expected, atol=5e-4)


# ---------------------------------------------------------------------------
# Hock-Schittkowski problems (nonlinear objective / constraints)
# ---------------------------------------------------------------------------

# Rows of the HS1/HS2/HS6 functions of x = (x1, x2):
#   which 0: r = 10 (x2 - x1²)    which 1: r = 1 - x1    which 2: both rows
_hs_kernel = cp.RawKernel(r"""
extern "C" __global__
void hs_rows(int which, const unsigned long long* state_ptrs, float* res, float* jac, int n) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  const float* x = (const float*)state_ptrs[t];
  int rows = which == 2 ? 2 : 1;
  int r = t * rows;
  if (which != 1) {
    res[r] = 10.f * (x[1] - x[0] * x[0]);
    if (jac) { jac[2 * r] = -20.f * x[0]; jac[2 * r + 1] = 10.f; }
    ++r;
  }
  if (which != 0) {
    res[r] = 1.f - x[0];
    if (jac) { jac[2 * r] = -1.f; jac[2 * r + 1] = 0.f; }
  }
}
""", "hs_rows")


class HsRows(pycunls.CustomFactorBatch):
    def __init__(self, which):
        super().__init__(2 if which == 2 else 1, [2], 1)
        self.which = which

    def evaluate(self, res_ptr, jac_ptr, sp_ptr, stream_handle, factor_ids_ptr, num_factor_ids):
        n = num_factor_ids
        with cupy_stream(stream_handle):
            _hs_kernel((1,), (128,), (cp.int32(self.which), cp.uint64(sp_ptr),
                                      cp.uint64(res_ptr), cp.uint64(jac_ptr), cp.int32(n)))
        return True


def hs_problem(x0, objective_rows, constraint=None, bounds=None):
    x = cp.asarray(np.array(x0, dtype=np.float32))
    states = pycunls.VectorStateBatch2(x, 1)
    states.set_num_active_states(1)
    problem = pycunls.Problem()
    problem.add_state_batch(states)
    keep = [states]
    objective = HsRows(objective_rows)
    objective.set_num_active_factors(1)
    problem.add_factor_batch(objective, [states.state_device_ptr(0)])
    keep.append(objective)
    if constraint is not None:
        constraint.set_num_active_factors(1)
        problem.add_factor_batch(constraint, [states.state_device_ptr(0)])
        keep.append(constraint)
    if bounds is not None:
        lo, hi = (cp.asarray(np.array(v, dtype=np.float32)) for v in bounds)
        bound = pycunls.BoundFactorBatch2(lo, hi, 1)
        bound.set_num_active_factors(1)
        problem.add_factor_batch(bound, [states.state_device_ptr(0)])
        keep += [bound, lo, hi]
    return x, problem, keep


@pytest.mark.parametrize("kind", ["gn", "lm"])
def test_hs2_active_bound(stream, kind):
    # min 100 (x2 - x1²)² + (1 - x1)²  s.t.  x2 >= 1.5;  x* = (1.2243707..., 1.5).
    # From the standard start (-2, 1) a local method may stop at the other KKT
    # point on the bound, x1 = -1.2210 (f = 4.94); start in x*'s basin instead.
    x, problem, keep = hs_problem([2.0, 1.0], 2, bounds=([-np.inf, 1.5], [np.inf, np.inf]))
    summary = pycunls.AugmentedLagrangianMinimizer(make_minimizer(kind)).minimize(stream, problem)
    assert summary.status == pycunls.AugmentedLagrangianMinimizerStatus.Converged
    np.testing.assert_allclose(cp.asnumpy(x), [1.2243707487363527, 1.5], atol=2e-4)
    # HS2 objective value 0.0504261879 is twice the least-squares cost.
    assert summary.final_cost == pytest.approx(0.0504261879 / 2, rel=1e-3)


@pytest.mark.parametrize("kind", ["gn", "lm"])
def test_hs6_nonlinear_equality(stream, kind):
    # min (1 - x1)²  s.t.  10 (x2 - x1²) = 0;  x* = (1, 1).
    rows = HsRows(0)
    constraint = pycunls.ConstraintFactorBatch(rows, pycunls.ConstraintKind.Equality)
    x, problem, keep = hs_problem([-1.2, 1.0], 1, constraint=constraint)
    summary = pycunls.AugmentedLagrangianMinimizer(make_minimizer(kind)).minimize(stream, problem)
    assert summary.status == pycunls.AugmentedLagrangianMinimizerStatus.Converged
    assert summary.max_violation <= 1e-4
    np.testing.assert_allclose(cp.asnumpy(x), [1.0, 1.0], atol=1e-3)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def test_constraint_wrapper_api(stream):
    obs = cp.zeros(3 * 4, dtype=cp.float32)
    inner = pycunls.PriorVectorFactorBatch3(obs, 4)
    c = pycunls.ConstraintFactorBatch(inner, pycunls.ConstraintKind.Equality, scale=2.0)
    assert c.capacity == 4 and c.num_active_factors == 0
    c.set_num_active_factors(3)
    assert inner.num_active_factors == 3
    assert c.kind == pycunls.ConstraintKind.Equality and c.scale == 2.0
    assert c.residuals_size == 3 and c.state_sizes() == [3]
    c.set_penalty(5.0, stream)
    cp.cuda.runtime.streamSynchronize(stream.get_stream())
    penalties = cp.ndarray((4,), cp.float32,
                           cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(c.penalties_ptr, 16, c), 0))
    np.testing.assert_array_equal(cp.asnumpy(penalties), 5.0)
    assert c.multipliers_ptr != 0
    with pytest.raises(ValueError):
        pycunls.ConstraintFactorBatch(inner, pycunls.ConstraintKind.Equality, scale=0.0)
    # Wrapping a constraint would hide it from the augmented Lagrangian.
    with pytest.raises(ValueError, match="constraint"):
        pycunls.WeightedFactorBatch(c, 2.0)
    with pytest.raises(ValueError, match="constraint"):
        pycunls.InformationFactorBatch(c, cp.zeros(4 * 9, dtype=cp.float32))
    options = pycunls.AugmentedLagrangianMinimizerOptions()
    assert options.constraint_tolerance == pytest.approx(1e-4)
    assert options.inner_line_search_steps == 10
    assert pycunls.MinimizerOptions().max_line_search_steps == 0
