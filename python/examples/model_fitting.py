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

"""Model fitting with ``pycunls.fit``: data and a model in, parameters and their quality out.

Six short scenes, each one call:

1. one curve, with measurement errors -> parameters ± standard errors, reduced chi-square;
2. 20,000 curves (one per pixel of an image) -> one batched solve on the GPU;
3. a line with 25% gross outliers -> a robust loss;
4. a parameter that must stay in a range -> bounds;
5. a sine whose frequency the initial guess gets wrong -> multiple starts;
6. the model check: reduced chi-square rejects a model that does not explain the data.

The models are plain NumPy-style functions of (x, p): p holds one row of parameters per sample.
A Warp ``@wp.func`` works too (autodiff Jacobians): see ``decay_wp`` below.

    python model_fitting.py [--quick]
"""

import argparse
import time

import numpy as np

import pycunls

RNG = np.random.default_rng(7)


def decay(x, p):
    """y = a exp(-b x) + c, p = [a, b, c]."""
    return p[:, 0] * np.exp(-p[:, 1] * x) + p[:, 2]


def line(x, p):
    """y = m x + q."""
    return p[:, 0] * x + p[:, 1]


def sine(x, p):
    """y = A sin(w x)."""
    return p[:, 0] * np.sin(p[:, 1] * x)


def show(title, result, truth=None):
    print(f"\n== {title}")
    print(result.report(max_fits=2))
    if truth is not None:
        print(f"  truth: {np.round(truth, 4).tolist()}")


def scene_single_curve():
    x = np.linspace(0, 5, 50)
    sigma = 0.02
    y = decay(x, np.array([[2.0, 1.3, 0.5]])) + RNG.normal(0, sigma, x.size)
    result = pycunls.fit(decay, x, y, p0=[1.0, 1.0, 0.0], sigma=sigma)
    show("1. one curve with measurement errors", result, [2.0, 1.3, 0.5])


def scene_many_curves(num_pixels):
    # Every pixel of a 2D detector records a decay over the same 16 time steps.
    x = np.linspace(0, 4, 16)
    truth = RNG.uniform([1.0, 0.5, 0.0], [2.0, 2.0, 0.5], (num_pixels, 3))
    a, b, c = truth[:, :1], truth[:, 1:2], truth[:, 2:]
    y = a * np.exp(-b * x) + c + RNG.normal(0, 0.005, (num_pixels, x.size))  # (pixels, 16)
    start = time.perf_counter()
    result = pycunls.fit(decay, x, y, p0=[1.0, 1.0, 0.0], sigma=0.005)
    elapsed = time.perf_counter() - start
    show(f"2. {num_pixels} curves in one call ({elapsed:.2f} s)", result)
    error = np.abs(result.params - truth).max(axis=0)
    print(f"  all converged: {result.all_converged}; largest error per parameter "
          f"{np.round(error, 4).tolist()}")


def scene_outliers():
    x = np.linspace(0, 10, 200)
    y = line(x, np.array([[2.0, -1.0]])) + RNG.normal(0, 0.1, x.size)
    bad = RNG.random(x.size) < 0.25
    y[bad] += RNG.uniform(3, 15, bad.sum())  # gross errors, all on one side
    plain = pycunls.fit(line, x, y, p0=[0.0, 0.0], sigma=0.1)
    robust = pycunls.fit(line, x, y, p0=[0.0, 0.0], sigma=0.1, loss="cauchy", loss_scale=2.0)
    print("\n== 3. a line with 25% outliers")
    print(f"  squared loss: m, q = {np.round(plain.params, 3).tolist()}  (pulled by the outliers)")
    print(f"  Cauchy loss:  m, q = {np.round(robust.params, 3).tolist()}  (truth [2.0, -1.0])")


def scene_bounds():
    # Few, noisy samples: the unconstrained offset c comes out negative, which is unphysical.
    x = np.linspace(0, 2, 8)
    y = decay(x, np.array([[1.0, 0.8, 0.02]])) + RNG.normal(0, 0.05, x.size)
    free = pycunls.fit(decay, x, y, p0=[1.0, 1.0, 0.1])
    bounded = pycunls.fit(decay, x, y, p0=[1.0, 1.0, 0.1], bounds=([0, 0, 0], [np.inf] * 3))
    print("\n== 4. bounds: c >= 0")
    print(f"  free:    {np.round(free.params, 3).tolist()}")
    print(f"  bounded: {np.round(bounded.params, 3).tolist()}  converged: {bounded.converged}")


def scene_starts():
    x = np.linspace(0, 10, 200)
    y = sine(x, np.array([[1.5, 3.7]])) + RNG.normal(0, 0.05, x.size)
    single = pycunls.fit(sine, x, y, p0=[1.0, 1.0], sigma=0.05)
    multi = pycunls.fit(sine, x, y, p0=[1.0, 1.0], sigma=0.05,
                        bounds=([0.5, 0.1], [3.0, 6.0]), starts=64)
    print("\n== 5. a local minimum and 64 starts")
    print(f"  one start:  A, w = {np.round(single.params, 3).tolist()}, "
          f"reduced chi2 {single.reduced_chi2:.1f}  (a local minimum)")
    print(f"  64 starts:  A, w = {np.round(multi.params, 3).tolist()}, "
          f"reduced chi2 {multi.reduced_chi2:.2f}  (start #{multi.start_index} won)")


def scene_model_check():
    # The data are two exponentials; one exponential cannot explain them.
    x = np.linspace(0, 5, 60)
    sigma = 0.01
    y = np.exp(-0.5 * x) + np.exp(-3.0 * x) + RNG.normal(0, sigma, x.size)
    one = pycunls.fit(decay, x, y, p0=[1.0, 1.0, 0.0], sigma=sigma)

    def two(x, p):
        return p[:, 0] * np.exp(-p[:, 1] * x) + p[:, 2] * np.exp(-p[:, 3] * x)

    both = pycunls.fit(two, x, y, p0=[1.0, 0.3, 1.0, 2.0], sigma=sigma)
    print("\n== 6. does the model explain the data? (reduced chi2 ~ 1 if it does)")
    print(f"  one exponential:  reduced chi2 {one.reduced_chi2:7.2f}  -> rejected")
    print(f"  two exponentials: reduced chi2 {both.reduced_chi2:7.2f}  -> "
          f"p = {np.round(both.params, 3).tolist()}")


def scene_warp():
    """The same decay as a Warp function: autodiff Jacobians, fastest for very large batches."""
    try:
        import warp as wp
    except ImportError:
        print("\n(warp-lang not installed: skipping the Warp model)")
        return
    wp.config.quiet = True

    @wp.func
    def decay_wp(x: float, p: wp.vec3) -> float:
        return p[0] * wp.exp(-p[1] * x) + p[2]

    x = np.linspace(0, 5, 50)
    y = decay(x, np.array([[2.0, 1.3, 0.5]])) + RNG.normal(0, 0.02, x.size)
    result = pycunls.fit(decay_wp, x, y, p0=[1.0, 1.0, 0.0], sigma=0.02)
    show("Warp model (same fit as scene 1)", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="smaller batch (for tests)")
    args = parser.parse_args()
    scene_single_curve()
    scene_many_curves(2_000 if args.quick else 20_000)
    scene_outliers()
    scene_bounds()
    scene_starts()
    scene_model_check()
    scene_warp()


if __name__ == "__main__":
    main()
