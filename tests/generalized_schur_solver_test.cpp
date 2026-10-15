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

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/linear_solver/sparse_linear_solver.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/vector_state_batch.h"

namespace cunls {
namespace {
// Exercise existing fixed-size states without adding a public state type.
template <int Dimension = 1>
std::unique_ptr<StateBatch> MakeVectorState(float *state_data, int dimension) {
  if (dimension == Dimension) {
    return std::make_unique<VectorStateBatch<Dimension>>(state_data, 1);
  }
  if constexpr (Dimension < 22) {
    return MakeVectorState<Dimension + 1>(state_data, dimension);
  }
  throw std::invalid_argument("Unsupported test state dimension");
}

// Construct a block graph with dense blocks, then make it strictly diagonally
// dominant SPD. A known solution and double host residual provide an oracle
// independent of the Schur implementation and the competing GPU solvers.
void CheckGraphSolution(const std::vector<int> &state_dimensions, int tile_size, int graph_shape,
                        bool use_state_metadata = true) {
  int num_blocks = state_dimensions.size(),
      num_rows = std::accumulate(state_dimensions.begin(), state_dimensions.end(), 0);
  std::vector<int> state_offsets{0}, row_state_ids;
  for (int state_dimension : state_dimensions) {
    state_offsets.push_back(state_offsets.back() + state_dimension);
  }
  for (int block_id = 0; block_id < num_blocks; ++block_id) {
    for (int component = 0; component < state_dimensions[block_id]; ++component) {
      row_state_ids.push_back(block_id);
    }
  }
  std::vector<float> dense_matrix(num_rows * num_rows, 0), expected_solution(num_rows),
      host_rhs(num_rows, 0);
  std::mt19937 random_generator(87 + graph_shape);
  std::uniform_real_distribution<float> random_value(-.2, .2);
  for (int row = 0; row < num_rows; ++row) {
    expected_solution[row] = random_value(random_generator);
  }
  for (int row = 0; row < num_rows; ++row) {
    for (int column = row + 1; column < num_rows; ++column) {
      int row_state = row_state_ids[row], column_state = row_state_ids[column];
      bool edge = row_state == column_state ||
                  (graph_shape == 0 && abs(row_state - column_state) == 1) ||
                  (graph_shape == 1 && (row_state < 2 || column_state < 2)) || graph_shape == 2;
      if (edge) {
        dense_matrix[row * num_rows + column] = dense_matrix[column * num_rows + row] =
            random_value(random_generator);
      }
    }
  }
  for (int row = 0; row < num_rows; ++row) {
    double sum = 1;
    for (int column = 0; column < num_rows; ++column) {
      sum += fabs(dense_matrix[row * num_rows + column]);
    }
    dense_matrix[row * num_rows + row] = sum;
  }
  if (graph_shape == 4) {
    int unobserved = num_rows / 2;
    for (int column = 0; column < num_rows; ++column) {
      dense_matrix[unobserved * num_rows + column] = dense_matrix[column * num_rows + unobserved] =
          0.f;
    }
    expected_solution[unobserved] = 0.f;
  }
  for (int row = 0; row < num_rows; ++row) {
    double sum = 0;
    for (int column = 0; column < num_rows; ++column) {
      sum += (double)dense_matrix[row * num_rows + column] * expected_solution[column];
    }
    host_rhs[row] = sum;
  }
  std::vector<int> row_offsets{0}, column_ids;
  std::vector<float> matrix_values;
  for (int row = 0; row < num_rows; row += tile_size) {
    for (int column = 0; column < num_rows; column += tile_size) {
      bool nonzero = false;
      for (int tile_row = 0; tile_row < tile_size; ++tile_row) {
        for (int tile_column = 0; tile_column < tile_size; ++tile_column) {
          nonzero |= dense_matrix[(row + tile_row) * num_rows + column + tile_column] != 0;
        }
      }
      if (tile_size == 1 && row_state_ids[row] == row_state_ids[column]) {
        nonzero = true;
      }
      if (nonzero) {
        column_ids.push_back(column / tile_size);
        for (int tile_row = 0; tile_row < tile_size; ++tile_row) {
          for (int tile_column = 0; tile_column < tile_size; ++tile_column) {
            matrix_values.push_back(
                dense_matrix[(row + tile_row) * num_rows + column + tile_column]);
          }
        }
      }
    }
    row_offsets.push_back(column_ids.size());
  }
  CudaStream stream;
  auto cuda_stream = stream.GetStream();
  Problem problem;
  dvector<float> state_data(num_rows), rhs(host_rhs), solution(num_rows);
  std::vector<std::unique_ptr<StateBatch>> states;
  if (use_state_metadata) {
    for (int row = 0; row < num_blocks; ++row) {
      states.push_back(
          MakeVectorState(state_data.data() + state_offsets[row], state_dimensions[row]));
      states.back()->SetNumActiveStates(1);
      problem.AddStateBatch(states.back().get());
    }
  }
  auto solver = CreateSparseLinearSolver(SparseLinearSolverType::GeneralizedSchur, {});
  auto verify = [&]() {
    std::vector<float> actual_solution(num_rows);
    solution.CopyToHost(actual_solution.data(), num_rows);
    double residual_squared_norm = 0, rhs_squared_norm = 0;
    for (int row = 0; row < num_rows; ++row) {
      double residual = host_rhs[row];
      for (int column = 0; column < num_rows; ++column) {
        residual -= (double)dense_matrix[row * num_rows + column] * actual_solution[column];
      }
      residual_squared_norm += residual * residual;
      rhs_squared_norm += (double)host_rhs[row] * host_rhs[row];
      EXPECT_NEAR(actual_solution[row], expected_solution[row], 5e-4);
    }
    EXPECT_LT(sqrt(residual_squared_norm / rhs_squared_norm), 1.05e-3);
  };
  if (tile_size == 1) {
    CSRSparseMatrix sparse_matrix;
    sparse_matrix.row_offsets = dvector<int>(row_offsets);
    sparse_matrix.col_ids = dvector<int>(column_ids);
    sparse_matrix.values = dvector<float>(matrix_values);
    ASSERT_TRUE(solver->Initialize(cuda_stream, problem, sparse_matrix, rhs, solution));
    ASSERT_TRUE(solver->Solve(cuda_stream, sparse_matrix, rhs, solution));
    verify();
    // Reuse the symbolic plan for changed values and RHS, then zero RHS.
    for (float &value : matrix_values) {
      value *= 2;
    }
    for (float &value : host_rhs) {
      value *= 2;
    }
    for (float &value : dense_matrix) {
      value *= 2;
    }
    sparse_matrix.values.CopyFromHost(matrix_values.data(), matrix_values.size());
    rhs.CopyFromHost(host_rhs.data(), num_rows);
    ASSERT_TRUE(solver->Solve(cuda_stream, sparse_matrix, rhs, solution));
    verify();
    std::fill(host_rhs.begin(), host_rhs.end(), 0);
    rhs.CopyFromHost(host_rhs.data(), num_rows);
    ASSERT_TRUE(solver->Solve(cuda_stream, sparse_matrix, rhs, solution));
  } else {
    BSRSparseMatrix sparse_matrix;
    sparse_matrix.block_size = tile_size;
    sparse_matrix.num_block_rows = num_rows / tile_size;
    sparse_matrix.row_offsets = dvector<int>(row_offsets);
    sparse_matrix.col_ids = dvector<int>(column_ids);
    sparse_matrix.values = dvector<float>(matrix_values);
    ASSERT_TRUE(solver->Initialize(cuda_stream, problem, sparse_matrix, rhs, solution));
    ASSERT_TRUE(solver->Solve(cuda_stream, sparse_matrix, rhs, solution));
    verify();
  }
}

TEST(GeneralizedSchur, MixedDimensionsAndGeneralGraphs) {
  for (int graph_shape = 0; graph_shape < 5; ++graph_shape) {
    CheckGraphSolution({1, 3, 6, 7, 15, 16, 19, 2, 4}, 1, graph_shape);
  }
}

TEST(GeneralizedSchur, BlockStorage) {
  for (int graph_shape = 0; graph_shape < 5; ++graph_shape) {
    CheckGraphSolution({6, 6, 3, 3, 3, 3, 3, 3}, 3, graph_shape);
  }
}

TEST(GeneralizedSchur, RawMatrices) {
  CheckGraphSolution({1, 1, 1, 1, 1, 1, 1}, 1, 0, false);
  CheckGraphSolution({3, 3, 3, 3}, 3, 1, false);
}

TEST(GeneralizedSchur, OnlyLargeBlocks) { CheckGraphSolution({19, 20, 21, 22}, 1, 0); }

TEST(GeneralizedSchur, SingleState) { CheckGraphSolution({7}, 1, 3); }

TEST(GeneralizedSchur, WideConnectedGraph) {
  // A small pivot connected to more than 64 scalar columns exercises the
  // generic clique path; rows wider than 128 use the cooperative residual.
  std::vector<int> state_dimensions(12, 16);
  state_dimensions.push_back(3);
  CheckGraphSolution(state_dimensions, 1, 2);
}

TEST(GeneralizedSchur, LargeImplicitChainAndReinitialization) {
  CudaStream stream;
  auto solver = CreateSparseLinearSolver(SparseLinearSolverType::GeneralizedSchur, {});
  for (int num_rows : {10000, 31, 12000}) {
    std::vector<int> row_offsets{0}, column_ids;
    std::vector<float> values, expected_solution(num_rows), host_rhs(num_rows);
    for (int row = 0; row < num_rows; ++row) {
      expected_solution[row] = std::sin(.017 * row);
    }
    for (int row = 0; row < num_rows; ++row) {
      if (row) {
        column_ids.push_back(row - 1);
        values.push_back(-1);
      }
      column_ids.push_back(row);
      values.push_back(4);
      if (row + 1 < num_rows) {
        column_ids.push_back(row + 1);
        values.push_back(-1);
      }
      row_offsets.push_back(column_ids.size());
      host_rhs[row] = 4 * expected_solution[row] - (row ? expected_solution[row - 1] : 0) -
                      (row + 1 < num_rows ? expected_solution[row + 1] : 0);
    }
    CSRSparseMatrix matrix;
    matrix.row_offsets = dvector<int>(row_offsets);
    matrix.col_ids = dvector<int>(column_ids);
    matrix.values = dvector<float>(values);
    dvector<float> rhs(host_rhs), solution(num_rows);
    Problem problem;
    ASSERT_TRUE(solver->Initialize(stream.GetStream(), problem, matrix, rhs, solution));
    ASSERT_TRUE(solver->Solve(stream.GetStream(), matrix, rhs, solution));
    std::vector<float> actual_solution(num_rows);
    solution.CopyToHost(actual_solution.data(), num_rows);
    double residual_squared_norm = 0, rhs_squared_norm = 0;
    for (int row = 0; row < num_rows; ++row) {
      double residual = host_rhs[row] - 4 * actual_solution[row] +
                        (row ? actual_solution[row - 1] : 0) +
                        (row + 1 < num_rows ? actual_solution[row + 1] : 0);
      residual_squared_norm += residual * residual;
      rhs_squared_norm += double(host_rhs[row]) * host_rhs[row];
    }
    EXPECT_LT(std::sqrt(residual_squared_norm / rhs_squared_norm), 1e-3);
  }
}

TEST(GeneralizedSchur, RejectsIndefinitePivot) {
  CSRSparseMatrix matrix;
  matrix.row_offsets = dvector<int>(std::vector<int>{0, 1});
  matrix.col_ids = dvector<int>(std::vector<int>{0});
  matrix.values = dvector<float>(std::vector<float>{-1});
  dvector<float> host_rhs(std::vector<float>{1}), solution(1);
  CudaStream stream;
  Problem problem;
  auto solver = CreateSparseLinearSolver(SparseLinearSolverType::GeneralizedSchur, {});
  ASSERT_TRUE(solver->Initialize(stream.GetStream(), problem, matrix, host_rhs, solution));
  EXPECT_FALSE(solver->Solve(stream.GetStream(), matrix, host_rhs, solution));
}

}  // namespace

}  // namespace cunls
