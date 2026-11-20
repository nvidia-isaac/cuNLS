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

"""Tests for pycunls.warp.WarpFunctionFactorBatch: factors from a residual function (every
argument type, autodiff / numeric / analytic Jacobians, slot checks, signatures, devices)."""

import numpy as np
import pytest
import warp as wp

import pycunls
from pycunls.catalog import slot_mismatch
from pycunls.warp import WarpFactorBatch, WarpFunctionFactorBatch, WarpStateBatch

vec5 = wp.types.vector(length=5, dtype=wp.float32)
mat12 = wp.types.matrix(shape=(1, 2), dtype=wp.float32)


@wp.func
def _world_point(X: wp.transform, c: wp.vec3) -> wp.vec3:
    return wp.transform_point(X, c)


@wp.func
def _rotated(q: wp.quat, c: wp.vec3) -> wp.vec3:
    return wp.quat_rotate(q, c)


@wp.func
def _scalar(x: float, a: float) -> float:
    return wp.sin(x) * a


@wp.func
def _vector5(x: vec5, a: float) -> wp.vec2:
    return wp.vec2(x[0] * x[1] + a, wp.exp(x[4]) - x[2] * x[3])


@wp.func
def _mixed(X: wp.transform, p: wp.vec3, v: wp.vec4, uv: wp.vec2) -> wp.vec3:
    q = wp.transform_point(wp.transform_inverse(X), p)  # smooth everywhere (random rotations)
    return wp.vec3(q[0] * uv[0] + q[1], q[2] * uv[1] + v[0] * q[0], v[3] * v[1] - v[2] * q[1])


@wp.func
def _cross(x: wp.vec3, y: wp.vec3) -> wp.vec3:
    return wp.cross(x, y)


@wp.func
def _reprojection(T: wp.transform, p: wp.vec3, uv: wp.vec2) -> wp.vec2:
    q = wp.transform_point(wp.transform_inverse(T), p)
    return wp.vec2(q[0] / q[2], q[1] / q[2]) - uv


@wp.func
def _product(x: wp.vec2, a: float) -> float:
    return a * x[0] * x[1]


@wp.func
def _product_jacobian(x: wp.vec2, a: float) -> mat12:
    return mat12(a * x[1], a * x[0])


@wp.func
def _product_jacobian_wrong(x: wp.vec2, a: float) -> mat12:
    return mat12(2.0 * a * x[1], a * x[0])


def _random_poses(rng, n, translation=0.5):
    """n row-major 4x4 world_from_body poses with random rotations."""
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q.T
    R = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                  2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                  2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                 axis=1).reshape(n, 3, 3)
    T = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    T[:, :3, :3] = R
    T[:, :3, 3] = rng.uniform(-translation, translation, (n, 3))
    return T.astype(np.float32)


def _states(kind, rng, n):
    """A state batch of n random states for an argument type, and its device data."""
    if kind == "se3":
        data = wp.array(_random_poses(rng, n).reshape(-1))
        batch = pycunls.SE3StateBatch(data, n)
    elif kind == "so3":
        data = wp.array(_random_poses(rng, n)[:, :3, :3].reshape(-1).copy())
        batch = pycunls.SO3StateBatch(data, n)
    else:
        dim = int(kind[3:])
        values = rng.uniform(-1.0, 1.0, (n, dim)).astype(np.float32)
        if kind == "vec3":
            values[:, 2] += 4.0  # in front of the cameras of the mixed residual
        data = wp.array(values.reshape(-1))
        batch = pycunls.VectorStateBatch(data, n, dim=dim)
    batch.set_num_active_states(n)
    return batch, data


def _problem(factor, batches, n):
    """Factor i reads state i of each batch."""
    factor.set_num_active_factors(n)
    problem = pycunls.Problem()
    for b in dict.fromkeys(batches):
        problem.add_state_batch(b)
    table = np.repeat(np.arange(n, dtype=np.int32)[:, None], len(batches), axis=1)
    problem.add_factor_batch(factor, list(batches), wp.array(table.reshape(-1)))
    wp.synchronize()
    return problem


CASES = {  # residual, the state kind of each slot, its data (dtype, values per factor)
    "se3": (_world_point, ["se3"], (wp.vec3, 3)),
    "so3": (_rotated, ["so3"], (wp.vec3, 3)),
    "float": (_scalar, ["vec1"], (wp.float32, 1)),
    "vec5": (_vector5, ["vec5"], (wp.float32, 1)),
    "mixed": (_mixed, ["se3", "vec3", "vec4"], (wp.vec2, 2)),
    "no data": (_cross, ["vec3", "vec3"], None),
}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("mode", ["auto", "autodiff", "numeric"])
def test_jacobians_match_central_differences_on_the_state_manifolds(case, mode, stream):
    """check_jacobians differentiates through each state batch's own Plus: the Jacobian of every
    argument type follows its state's update convention."""
    fn, kinds, item = CASES[case]
    rng, n = np.random.default_rng(1), 32
    batches = [_states(k, rng, n)[0] for k in kinds]
    data = None
    if item is not None:
        dtype, width = item
        values = rng.uniform(-1.0, 1.0, (n, width)).astype(np.float32)
        data = wp.array(values.reshape(n) if width == 1 else values, dtype=dtype)
    factor = WarpFunctionFactorBatch(fn, data=data, capacity=n, jacobian=mode)
    assert factor.slot_states == tuple(
        {"se3": "SE3StateBatch", "so3": "SO3StateBatch"}.get(k, "VectorStateBatch") for k in kinds)
    problem = _problem(factor, batches, n)
    for check in pycunls.check_jacobians(problem, stream):
        assert check.ok, (case, mode, check)
    assert factor.jacobian_mode == ("autodiff" if mode == "auto" else mode)
    assert not pycunls.diagnose(problem, stream)


def test_analytic_jacobian_function_is_used_as_given(stream):
    rng, n = np.random.default_rng(2), 16
    batch, _ = _states("vec2", rng, n)
    a = wp.array(rng.uniform(0.5, 2.0, n).astype(np.float32))
    for jacobian, ok in ((_product_jacobian, True), (_product_jacobian_wrong, False)):
        factor = WarpFunctionFactorBatch(_product, data=a, capacity=n, jacobian=jacobian)
        assert factor.jacobian_mode == "analytic"
        problem = _problem(factor, [batch], n)
        assert all(check.ok == ok for check in pycunls.check_jacobians(problem, stream))


@wp.func
def _stepped(x: float, a: float) -> float:
    return float(int(x * 1000.0)) * 0.001 - a  # autodiff: slope 0; central differences: about 1


def test_auto_falls_back_to_central_differences_when_autodiff_disagrees(stream):
    rng, n = np.random.default_rng(5), 8
    batch, _ = _states("vec1", rng, n)
    factor = WarpFunctionFactorBatch(_stepped, data=wp.zeros(n, dtype=float), capacity=n)
    problem = _problem(factor, [batch], n)
    with pytest.warns(RuntimeWarning, match="differs from central differences.*Fix:"):
        pycunls.check_jacobians(problem, stream)
    assert factor.jacobian_mode == "numeric"
    assert all(check.ok for check in pycunls.check_jacobians(problem, stream))


@wp.struct
class _Target:
    point: wp.vec3
    weight: float


@wp.func
def _weighted_offset(x: wp.vec3, item: _Target) -> wp.vec3:
    return item.weight * (x - item.point)


def test_struct_data_item(stream):
    rng, n = np.random.default_rng(6), 8
    batch, data = _states("vec3", rng, n)
    items = []
    for i in range(n):
        item = _Target()
        item.point, item.weight = wp.vec3(float(i), 1.0, 2.0), 2.0
        items.append(item)
    factor = WarpFunctionFactorBatch(_weighted_offset, data=wp.array(items, dtype=_Target),
                                     capacity=n)
    problem = _problem(factor, [batch], n)
    assert all(check.ok for check in pycunls.check_jacobians(problem, stream))
    assert pycunls.LevenbergMarquardtMinimizer().minimize(stream, problem).converged
    expected = np.stack([np.arange(n), np.ones(n), 2 * np.ones(n)], axis=1)
    np.testing.assert_allclose(data.numpy().reshape(n, 3), expected, atol=1e-5)


def test_bundle_adjustment_matches_the_builtin_reprojection_factor(stream):
    """The same small bundle adjustment with the built-in ReprojectionFactorBatch and with a
    residual function of the same formula: same iterations, same solution."""
    rng = np.random.default_rng(3)
    cams, points = 4, 40
    T = np.tile(np.eye(4, dtype=np.float32), (cams, 1, 1))
    T[:, 0, 3] = np.linspace(-0.6, 0.6, cams)
    P = rng.uniform([-1, -1, 4], [1, 1, 6], (points, 3)).astype(np.float32)
    pairs = np.array([(c, j) for c in range(cams) for j in range(points)], np.int32)
    local = P[pairs[:, 1]] - T[pairs[:, 0], :3, 3]
    uv = (local[:, :2] / local[:, 2:]).astype(np.float32)
    T0 = T.copy()
    T0[2:, :3, 3] += rng.normal(0, 0.03, (cams - 2, 3))
    P0 = P + rng.normal(0, 0.05, P.shape).astype(np.float32)
    results = []
    for builtin in (True, False):
        poses = wp.array(T0.reshape(-1))
        pts = wp.array(P0.reshape(-1))
        const = wp.array(np.array([0, 1], np.int32))  # two fixed cameras fix the gauge and scale
        cam_batch = pycunls.SE3StateBatch(poses, cams, const, 2)
        cam_batch.set_num_active_states(cams, 2)
        pt_batch = pycunls.VectorStateBatch3(pts, points)
        pt_batch.set_num_active_states(points)
        m = len(pairs)
        factor = (pycunls.ReprojectionFactorBatch(wp.array(uv.reshape(-1)), m) if builtin else
                  WarpFunctionFactorBatch(_reprojection, data=wp.array(uv, dtype=wp.vec2),
                                          capacity=m))
        factor.set_num_active_factors(m)
        problem = pycunls.Problem()
        problem.add_state_batch(cam_batch)
        problem.add_state_batch(pt_batch)
        problem.add_factor_batch(factor, [cam_batch, pt_batch], wp.array(pairs.reshape(-1)))
        wp.synchronize()
        summary = pycunls.LevenbergMarquardtMinimizer().minimize(stream, problem)
        assert summary.converged, summary.message
        results.append((summary.num_iterations, poses.numpy(), pts.numpy()))
    assert abs(results[0][0] - results[1][0]) <= 1
    np.testing.assert_allclose(results[0][1], results[1][1], atol=1e-4)
    np.testing.assert_allclose(results[0][2], results[1][2], atol=1e-3)
    np.testing.assert_allclose(results[1][2].reshape(-1, 3), P, atol=5e-3)  # float32, 5 m depth


def test_slot_connected_to_another_layout_is_reported(stream):
    """An SO3StateBatch and a VectorStateBatch3 both have tangent size 3: the slot types tell
    them apart, before any evaluation reads the wrong layout."""
    rng, n = np.random.default_rng(4), 8
    rotations, _ = _states("so3", rng, n)
    vectors, _ = _states("vec3", rng, n)
    data = wp.array(rng.uniform(-1, 1, (n, 3)).astype(np.float32), dtype=wp.vec3)
    point_factor = WarpFunctionFactorBatch(_cross, capacity=n)        # reads vectors
    rotation_factor = WarpFunctionFactorBatch(_rotated, data=data, capacity=n)  # reads rotations
    for factor, wrong, expected in ((point_factor, rotations, "VectorStateBatch"),
                                    (rotation_factor, vectors, "SO3StateBatch")):
        problem = _problem(factor, [wrong] * len(factor.slot_states), n)
        issues = pycunls.diagnose(problem, stream)
        assert issues and issues[0].severity == "error"
        assert f"needs a {expected}, got a {type(wrong).__name__}" in issues[0].message
        assert "Fix:" in issues[0].message
    assert slot_mismatch("f", 0, 3, rotations, "SO3StateBatch") is None
    assert slot_mismatch("f", 0, 3, vectors, "VectorStateBatch") is None
    assert "tangent size 6" in slot_mismatch("f", 0, 6, vectors, "VectorStateBatch")


def test_rejects_wrong_signatures():
    data = wp.zeros(4, dtype=float)
    with pytest.raises(TypeError, match="Fix:"):
        WarpFunctionFactorBatch(lambda x, a: 0.0, data=data, capacity=4)

    @wp.func
    def matrix_state(m: wp.mat33, a: float) -> float:
        return m[0, 0] - a

    @wp.func
    def integer_result(x: wp.vec3) -> int:
        return 0

    with pytest.raises(TypeError, match="wrong signature.*Fix:"):
        WarpFunctionFactorBatch(matrix_state, data=data, capacity=4)
    with pytest.raises(TypeError, match="wrong signature.*Fix:"):
        WarpFunctionFactorBatch(integer_result, capacity=4)
    with pytest.raises(TypeError, match="data must be a wp.array.*Fix:"):
        WarpFunctionFactorBatch(_scalar, data=wp.zeros(4, dtype=wp.vec2), capacity=4)
    with pytest.raises(ValueError, match="fewer items than the capacity.*Fix:"):
        WarpFunctionFactorBatch(_scalar, data=data, capacity=8)
    with pytest.raises(ValueError, match="jacobian='fd'.*Fix:"):
        WarpFunctionFactorBatch(_scalar, data=data, capacity=4, jacobian="fd")
    with pytest.raises(TypeError, match="Jacobian function.*Fix:"):
        WarpFunctionFactorBatch(_scalar, data=data, capacity=4, jacobian=_product_jacobian)


def test_device_defaults_to_the_current_cuda_device():
    device = wp.get_device()
    assert WarpFactorBatch(1, [1], 4)._device == device
    assert WarpStateBatch(wp.zeros(4, dtype=float), 1, 1, 4)._device == device
    assert WarpFunctionFactorBatch(_cross, capacity=4)._device == device
    with pytest.raises(ValueError, match="CUDA device.*Fix:"):
        WarpFactorBatch(1, [1], 4, device="cpu")
    if wp.get_cuda_device_count() < 2:
        pytest.skip("wp.ScopedDevice: needs two visible GPUs")
    with wp.ScopedDevice("cuda:1"):
        assert str(WarpFactorBatch(1, [1], 4)._device) == "cuda:1"
        assert str(WarpFunctionFactorBatch(_cross, capacity=4)._device) == "cuda:1"
        data = wp.zeros(4, dtype=float, device="cuda:0")
    assert str(WarpFunctionFactorBatch(_scalar, data=data, capacity=4)._device) == "cuda:0"
