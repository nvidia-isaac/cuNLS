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

#include <cstdint>
#include <vector>

namespace cunls {
namespace internal {

/** @brief Widest panel (columns) of the supernodal factor; one CTA factors a panel's diagonal. */
constexpr int kCholeskyPanelWidth = 64;

/** @brief Rows of a panel processed by one CTA (factor-solve and update tiles). */
constexpr int kCholeskyTileRows = 64;

/**
 * @brief Most rows (its diagonal block plus the rows below it) of a small panel: one warp,
 * a lane per row, factors it, applies its updates and solves it backward (many small
 * independent subproblems, e.g. batched inverse kinematics, and the small supernodes of
 * chains and graphs). Taller panels take the CTA kernels.
 */
constexpr int kCholeskySmallPanelRows = 32;

/**
 * @brief The warp width of a small panel of `width` columns: the warp kernels are compiled for at
 * most 8, 16 or 32 columns, so a launch of narrow panels needs fewer registers (more resident
 * warps). A launch is compiled for the widest panel of its level.
 */
inline int CholeskyWarpWidth(int width) { return width <= 8 ? 8 : width <= 16 ? 16 : 32; }

/**
 * @brief Small panels a level needs before they take the warp kernels: a warp works through
 * its panel serially, so it wins on throughput (many panels per SM), while with fewer panels
 * the CTA kernels finish them in about one wave with less latency each (e.g. the levels of a
 * single chain).
 */
constexpr int kCholeskyMinSmallPanelsPerLevel = 256;

/**
 * @brief The symbolic analysis of a sparse SPD matrix for BlockSparseCholeskySolver:
 * the fill-reducing ordering, the supernodal structure of the factor L, and
 * the program of the numeric phase with its index maps.
 *
 * L is stored as panels: groups of at most kCholeskyPanelWidth consecutive
 * columns (in the permuted order) with identical structure below their
 * diagonal block. Panel p is a dense column-major height_p x width_p block
 * whose first width_p rows are its own columns (the diagonal block, lower
 * triangle used) and whose remaining rows are its below rows (`below_rows`
 * from `rows_offset`, ascending, all after the panel's columns). A panel's
 * level is one more than the highest level among the panels that update it,
 * so the panels of a level are independent.
 *
 * The numeric phase runs `program`, a list of kernel launches: per level the
 * factor launches, then the update launches; then the backward launches,
 * levels in reverse. Every policy of the numeric phase is decided here, once:
 * which panels take a warp instead of CTAs (kCholeskySmallPanelRows,
 * kCholeskyMinSmallPanelsPerLevel), the warp width of a launch, and that a
 * root panel (no rows below) is solved backward by its factor task (it has no
 * backward task).
 */
struct CholeskyPlan {
  /** @brief A panel of L. */
  struct Panel {
    int col;          ///< First column (permuted order).
    int width;        ///< Columns, at most kCholeskyPanelWidth.
    int height;       ///< Rows: width plus the rows below the diagonal block.
    int rows_offset;  ///< Offset of its below rows in below_rows.
    /// No panel updates it (level 0): its diagonal block is still A's when it is factored, so
    /// the zero-pivot test reads A's diagonal there (no copy of it is needed).
    bool leaf;
    int64_t offset;  ///< Offset of its column-major values in the factor storage.
    /// Offset of its factored diagonal block L_D (width x width, column-major, the reciprocals
    /// of its diagonal on the diagonal), for the backward substitution; -1 for a root (no rows
    /// below), which its factor task solves backward.
    int64_t diagonal_offset;
  };

  /**
   * @brief A run of a panel's below rows that are the columns of one target
   * panel: the update L_b L_bᵀ of the source panel restricted to those
   * columns and to the rows from the run on goes to the target.
   */
  struct Run {
    int source;             ///< Source panel.
    int target;             ///< Target panel.
    int begin;              ///< First below-row position (of the source) in the run.
    int end;                ///< One past the last.
    int64_t relmap_offset;  ///< Offset in the relative map of the target positions of below
                            ///< rows begin... of the source (the map is built on the device).
  };

  /** @brief What a launch does with each of its tasks. */
  enum class Phase : char {
    kFactor,    ///< Factor a panel and take its step of the forward substitution (a root: also
                ///< its backward solve).
    kUpdate,    ///< Apply a run's update to its target panel.
    kBackward,  ///< Take a panel's step of the backward substitution.
  };

  /** @brief One task of a launch: for a CTA, a tile of at most kCholeskyTileRows below rows. */
  struct Task {
    int index;      ///< The panel (factor, backward) or the run (update).
    int row_begin;  ///< First below row (position) of the CTA's tile, at or after an update's
                    ///< run; 0 for a warp, which handles the whole panel or run.
  };

  /** @brief One kernel launch of the numeric phase: tasks [first_task, first_task + num_tasks). */
  struct Launch {
    Phase phase;     ///< What each task does.
    int warp_width;  ///< 0: a CTA per task; 8, 16 or 32: a warp per task, the kernel compiled for
                     ///< panels of at most that many columns (CholeskyWarpWidth of the widest).
    int first_task;  ///< Offset of its tasks in `tasks`.
    int num_tasks;   ///< Tasks (CTAs or warps).
  };

  int n = 0;                    ///< Matrix size.
  std::vector<int> perm;        ///< Permuted column -> original column.
  std::vector<Panel> panels;    ///< Panels in column order.
  std::vector<int> below_rows;  ///< Below rows (permuted indices, ascending) of all panels.
  int64_t factor_size = 0;      ///< Floats of the factor storage.
  int64_t diagonal_size = 0;    ///< Floats of the factored diagonal blocks.
  std::vector<Run> runs;        ///< Update runs of all panels.
  int64_t relmap_size = 0;      ///< Entries of the relative map (see Run).

  int num_levels = 0;           ///< Levels of the schedule.
  std::vector<Task> tasks;      ///< The tasks of all launches, launch by launch.
  std::vector<Launch> program;  ///< The numeric phase, in order (see the struct comment).

  // Statistics of the analysis (for logs and tests).
  int num_supervariables = 0;  ///< Groups of consecutive columns with identical structure.
  int num_supernodes = 0;      ///< Supernodes before splitting into panels.
  int num_small_panels = 0;    ///< Panels that take a warp.
  double flops = 0.0;          ///< Factorization flops (multiply-add counted as 2).
  const char *ordering = "";   ///< Name of the chosen ordering.
};

/**
 * @brief Analyzes the pattern of a symmetric matrix, given as its
 * supervariables (groups of consecutive columns with identical patterns: a
 * state's tangent block, or several states) and their graph.
 *
 * Two fill-reducing orderings of the supervariable graph are built and the
 * one with the lower estimated factorization time kept (a fixed latency per
 * sequential panel step plus the flops):
 *   - approximate minimum degree on the quotient graph (least fill on
 *     irregular graphs: covisibility, bundle adjustment, random loop
 *     closures);
 *   - nested dissection with balanced breadth-first level-set separators and
 *     minimum-degree leaves (short level schedules and less fill on chains
 *     and grids: trajectories, visual-inertial windows, pose graphs with
 *     local loop closures).
 * Fundamental supernodes are amalgamated with their parents where that saves
 * a panel step at no more than 50% explicit zeros, then cut into panels.
 *
 * @param n Matrix size.
 * @param supervariable_start First column of every supervariable, then n.
 * @param adjacency_offsets Per supervariable: offsets into @p adjacency (CSR).
 * @param adjacency Neighbor supervariables (without the supervariable itself).
 */
CholeskyPlan AnalyzeCholesky(int n, const std::vector<int> &supervariable_start,
                             const std::vector<int> &adjacency_offsets,
                             const std::vector<int> &adjacency);

}  // namespace internal
}  // namespace cunls
