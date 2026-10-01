# RANSAC PnP Example

This example recovers a camera pose from 3D-2D correspondences of which a
large fraction are **gross outliers** (wrong matches), using
`RansacLevenbergMarquardtMinimizer` (`cunls/minimizer/ransac_minimizer.h`).
For the same problem it also runs a plain `LevenbergMarquardtMinimizer`,
which the outliers pull far from the truth.

```
RANSAC PnP example: 2000 correspondences, 50% outliers
  Initial guess      : rotation error 5.39 deg, translation error 0.996
  LevenbergMarquardt : rotation error 6.78 deg, translation error 8.36
  RansacLM           : rotation error 0.042 deg, translation error 0.0053
  RANSAC: 1 round(s), 256 hypotheses, 980 inliers (49%)
  Inlier mask: 980 of 983 true inliers kept, 0 outliers accepted
```

```bash
./ransac_pnp_example                                        # 2000 points, 50% outliers
./ransac_pnp_example --num-points 100000 --outlier-ratio 0.8
```

## What changes compared to a regular solve

Nothing in the problem: the states, the `PnPFactorBatch` and the `Problem`
are built exactly as in `examples/pnp`. Only the minimizer differs:

```cpp
cunls::RansacLevenbergMarquardtMinimizerOptions options;
cunls::RansacMinimizerOptions &ransac = options.base_options;
// One entry per residual batch, in the order they were added to the problem.
ransac.factor_batches = {{cunls::RansacRole::kSampled, /*inlier_threshold=*/0.01f}};
ransac.seed = 1;

cunls::RansacLevenbergMarquardtMinimizer minimizer(options);
cunls::RansacSummary summary = minimizer.Minimize(stream, problem);  // writes the pose back
const uint8_t *mask = minimizer.InlierMask(0);  // device, 1 byte per factor, 1 = inlier
```

- **`RansacRole::kSampled`** marks a factor batch whose factors may be
  outliers. Minimal samples are drawn from it and each of its factors is
  classified as inlier (`|r| <= inlier_threshold`) or outlier.
- **The inlier threshold** is in the residual's own units: here normalized
  image coordinates. With per-axis noise sigma = 3e-3, a threshold of
  0.01 (~3.3 sigma) keeps ~99.6% of the true inliers.
- **The estimate** is written back into the problem's state batch, and
  `InlierMask(i)` returns the classification of residual batch `i`.

See `docs/sphinx/ransac.rst` for how the algorithm works and how to tune it.

## Files

- `main.cpp`: the cuNLS workflow (`PnPProblem` builds the same problem for
  both minimizers).
- `../utils/`: header-only helpers kept out of `main` so it reads as the cuNLS
  workflow: `datasets.h` (synthetic scenes), `validation.h` (error metrics),
  `report.h` (printing and the quality verdict), `cli.h` (command-line flags),
  `se3_utils.h` / `camera_utils.h` (host SE(3) math and projection).

## Walkthrough

1. `examples::MakePnPScene` generates a ground-truth pose, random 3D points
   in front of the camera, their noisy normalized projections, and a
   perturbed initial guess. `--outlier-ratio` of the observations are
   replaced by random image points at least four thresholds away from the
   true projection.
2. Solve with `LevenbergMarquardtMinimizer` (no outlier handling) and with
   `RansacLevenbergMarquardtMinimizer`, each on a fresh copy of the problem.
3. Report the pose errors, the RANSAC summary, and the inlier mask against
   the generated ground truth.

## Build locally (all examples)

```bash
cmake -S examples -B build/examples/all \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUNLS_INSTALL_DIR=/path/to/cunls_install
cmake --build build/examples/all -j
```

Output binary: `ransac_pnp_example`.
