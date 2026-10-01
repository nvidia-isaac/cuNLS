/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// Printing helpers for the examples: solver summaries, before -> after
// metrics, pose errors, RANSAC results, and the final quality verdict.

#include <iostream>
#include <sstream>
#include <string>

#include "cunls/common/types.h"
#include "cunls/minimizer/gauss_newton_minimizer.h"
#include "cunls/minimizer/ransac_minimizer.h"
#include "utils/validation.h"

namespace examples {

// Concatenates values with default ostream formatting (e.g. 6.2e-06, not 0.000006).
template <typename... Args>
std::string Str(const Args &...args) {
  std::ostringstream out;
  (out << ... << args);
  return out.str();
}

// Prints a section title.
inline void PrintTitle(const std::string &title) { std::cout << title << "\n"; }

// "  <label>: <value>" with labels padded to a common width.
template <typename T>
void PrintValue(const std::string &label, const T &value) {
  std::cout << "  " << label << ":" << std::string(label.size() < 22 ? 22 - label.size() : 1, ' ')
            << value << "\n";
}

// Initial / final cost and iteration count of a solve.
inline void PrintSummary(const cunls::MinimizerSummary &summary) {
  PrintValue("Initial cost", summary.initial_cost);
  PrintValue("Final cost", summary.final_cost);
  PrintValue("Iterations", summary.num_iterations);
}

// "  <label>: <before> -> <after>" for an error metric.
inline void PrintChange(const std::string &label, float before, float after) {
  PrintValue(label, Str(before, " -> ", after));
}

// Rotation and translation error of `pose` with respect to `gt`.
inline void PrintPoseError(const std::string &label, const SE3Transform &pose,
                           const SE3Transform &gt) {
  PrintValue(label, Str("rotation error ", RotationErrorDeg(pose, gt), " deg, translation error ",
                        TranslationError(pose, gt)));
}

// Rounds / hypotheses / inliers of a RANSAC run, and its mask against the truth.
inline void PrintRansacResult(const cunls::RansacSummary &summary, const InlierMaskStats &mask) {
  PrintValue("RANSAC",
             Str(summary.num_rounds, " round(s), ", summary.num_hypotheses, " hypotheses, ",
                 summary.num_inliers, " inliers (", 100.f * summary.inlier_ratio, "%)"));
  PrintValue("Inlier mask",
             Str(mask.kept_inliers, " of ", mask.true_inliers, " true inliers kept, ",
                 mask.accepted_outliers, " outliers accepted"));
}

// Exit code of an example: 0 if `ok`, otherwise 2 after printing the failure.
inline int QualityExitCode(bool ok) {
  if (!ok) {
    std::cerr << "Optimization quality check failed.\n";
    return 2;
  }
  return 0;
}

}  // namespace examples
