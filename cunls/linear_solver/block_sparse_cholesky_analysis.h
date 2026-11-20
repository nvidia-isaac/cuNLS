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
 * @brief Width classes of small panels (at most 8, 16 or 32 columns): a level's warp kernels
 * are compiled for its widest class, so the registers of a batch of narrow panels stay few
 * (more resident warps).
 */
constexpr int kCholeskySmallWidthClasses = 3;

/** @brief The width class (kCholeskySmallWidthClasses) of a small panel of `width` columns. */
inline int CholeskySmallWidthClass(int width) { return width <= 8 ? 0 : width <= 16 ? 1 : 2; }

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
 * the level schedule and index maps of the numeric phase.
 *
 * L is stored as panels: groups of at most kCholeskyPanelWidth consecutive
 * columns (in the permuted order) with identical structure below their
 * diagonal block. Panel p is a dense column-major height_p x width_p block
 * whose first width_p rows are its own columns (the diagonal block, lower
 * triangle used) and whose remaining rows are its below rows (`below_rows`
 * from `rows_offset`, ascending, all after the panel's columns). The numeric
 * phase processes the panels level by level: a panel's level is one more than
 * the highest level among the panels that update it, so the panels of a level
 * are independent. Small panels (at most kCholeskySmallPanelRows rows) are
 * scheduled apart from the others, one warp each instead of factor and update
 * items, in the levels with enough of them (kCholeskyMinSmallPanelsPerLevel).
 */
struct CholeskyPlan {
  /** @brief A panel of L. */
  struct Panel {
    int col;          ///< First column (permuted order).
    int width;        ///< Columns, at most kCholeskyPanelWidth.
    int height;       ///< Rows: width plus the rows below the diagonal block.
    int rows_offset;  ///< Offset of its below rows in below_rows.
    int64_t offset;   ///< Offset of its column-major values in the factor storage.
    /// Offset of its factored diagonal block L_D (width x width, the reciprocals
    /// of its diagonal on the diagonal): row-major, column-major for the panels in
    /// small_panels (written and read by the warp kernels only).
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

  /**
   * @brief One CTA of the factor kernel: a panel not factored by a warp (not in small_panels)
   * and the first of its <= 64 below rows.
   */
  struct FactorItem {
    int panel;      ///< The panel.
    int row_begin;  ///< First of its below rows (position) handled by the CTA.
  };

  /** @brief One CTA of the update kernel: a run of a CTA panel and the first of its <= 64 rows. */
  struct UpdateItem {
    int panel;      ///< The source panel.
    int run;        ///< Its run (index into runs): the target and the columns.
    int row_begin;  ///< First below row (position, at or after the run) handled by the CTA.
  };

  int n = 0;                    ///< Matrix size.
  std::vector<int> perm;        ///< Permuted column -> original column.
  std::vector<Panel> panels;    ///< Panels in column order.
  std::vector<int> below_rows;  ///< Below rows (permuted indices, ascending) of all panels.
  int64_t factor_size = 0;      ///< Floats of the factor storage.
  int64_t diagonal_size = 0;    ///< Floats of the factored diagonal blocks.
  std::vector<Run> runs;        ///< Update runs of all panels.
  int64_t relmap_size = 0;      ///< Entries of the relative map (see Run).

  int num_levels = 0;                    ///< Levels of the schedule.
  std::vector<FactorItem> factor_items;  ///< Factor CTAs by level.
  std::vector<int> factor_offsets;       ///< num_levels + 1 offsets into factor_items.
  std::vector<UpdateItem> update_items;  ///< Update CTAs by level.
  std::vector<int> update_offsets;       ///< num_levels + 1 offsets into update_items.
  /** @brief Per level: whether no two panels of the level update the same target (no atomics). */
  std::vector<char> update_exclusive;
  /// The small panels (at most kCholeskySmallPanelRows rows) by level that take one warp each in
  /// the factor and backward kernels (no factor items; see kCholeskyMinSmallPanelsPerLevel).
  /// Within a level those with rows below come before the roots (no rows below, e.g. a whole
  /// subproblem of one block), which the factor kernel also solves backward.
  std::vector<int> small_panels;
  std::vector<int> small_panel_offsets;  ///< num_levels + 1 offsets into small_panels.
  std::vector<int> small_root_offsets;   ///< Per level: its first root in small_panels.
  /// Per level: the widest CholeskySmallWidthClass among its small panels (the kernels' size).
  std::vector<int> small_width_class;
  /// The runs of the small panels by level: one warp each in the update kernel (no update items).
  std::vector<int> small_runs;
  std::vector<int> small_run_offsets;  ///< num_levels + 1 offsets into small_runs.

  // Statistics of the analysis (for logs and tests).
  int num_supervariables = 0;  ///< Groups of consecutive columns with identical structure.
  int num_supernodes = 0;      ///< Supernodes before splitting into panels.
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
