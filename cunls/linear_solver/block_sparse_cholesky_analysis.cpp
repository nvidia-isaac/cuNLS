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

#include "cunls/linear_solver/block_sparse_cholesky_analysis.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace cunls {
namespace internal {
namespace {

/** Supervariable graph: consecutive columns with identical patterns, as one vertex. */
struct Graph {
  std::vector<int> start;             ///< Per vertex: first column; size + 1 entries.
  std::vector<int> weight;            ///< Per vertex: columns.
  std::vector<std::vector<int>> adj;  ///< Per vertex: neighbors (no self).
  int Size() const { return static_cast<int>(weight.size()); }
};

/** A 64-bit mix (splitmix64 finalizer) for order-independent set hashes. */
uint64_t Mix(uint64_t x) {
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

/**
 * Approximate minimum degree ordering of the subgraph induced by `nodes`, on
 * the quotient graph: an eliminated vertex becomes an element standing for
 * the clique of its neighbors, so the fill is never formed. Each variable
 * keeps its adjacent elements and the variables it still reaches directly;
 * its external degree (in columns) is bounded from above by its direct
 * neighbors, the new element and, for every other adjacent element e, the
 * part of e outside the new element. Elements inside the new element are
 * absorbed, and variables with identical adjacency are merged
 * (indistinguishable: they are eliminated together). `local` is scratch of
 * size g.Size() filled with -1, and left so.
 */
std::vector<int> MinimumDegree(const Graph &g, const std::vector<int> &nodes,
                               std::vector<int> &local) {
  const int m = static_cast<int>(nodes.size());
  for (int i = 0; i < m; ++i) local[nodes[i]] = i;
  std::vector<std::vector<int>> vars(m), elems(m);  // Per variable: adjacency.
  std::vector<std::vector<int>> members(m);  // Per element: its variables (dead ones skipped).
  std::vector<int64_t> nv(m), degree(m, 0), element_weight(m, 0), outside(m, 0);
  int64_t remaining = 0;  // Columns of the variables not eliminated yet.
  size_t edges = 0;
  for (int i = 0; i < m; ++i) {
    nv[i] = g.weight[nodes[i]];
    remaining += nv[i];
    for (int u : g.adj[nodes[i]]) {
      if (local[u] >= 0) {
        vars[i].push_back(local[u]);
        degree[i] += g.weight[u];
      }
    }
    edges += vars[i].size();
  }
  // Variables whose lists grow far beyond the average degree are updated lazily (below).
  const size_t long_lists = std::max<size_t>(64, 16 * edges / std::max(m, 1));
  for (int i = 0; i < m; ++i) local[nodes[i]] = -1;
  // State per index: variable (principal), merged into another variable, or element.
  enum : char { kVariable, kMerged, kElement };
  std::vector<char> state(m, kVariable);
  std::vector<int> merged_next(m, -1);  // Variables merged into a principal one (a list).
  // Degree buckets (doubly linked), degrees capped at the remaining columns.
  std::vector<int> head(remaining + 1, -1), next(m, -1), prev(m, -1);
  int64_t min_degree = 0;
  auto insert = [&](int i) {
    const int64_t d = degree[i];
    next[i] = head[d];
    prev[i] = -1;
    if (head[d] >= 0) prev[head[d]] = i;
    head[d] = i;
    min_degree = std::min(min_degree, d);
  };
  auto remove = [&](int i) {
    if (prev[i] >= 0) {
      next[prev[i]] = next[i];
    } else {
      head[degree[i]] = next[i];
    }
    if (next[i] >= 0) prev[next[i]] = prev[i];
  };
  for (int i = 0; i < m; ++i) {
    degree[i] = std::min(degree[i], remaining - nv[i]);
    insert(i);
  }
  std::vector<int> mark(m, -1), stamp(m, -1);
  std::vector<int> order;
  order.reserve(m);
  std::vector<int> lp;
  std::vector<std::pair<uint64_t, int>> hashes;
  std::vector<char> heavy;
  std::vector<size_t> pruned(m, 0);  // Per variable: list lengths after its last pruning.
  for (int pass = 0; static_cast<int>(order.size()) < m; ++pass) {
    while (head[min_degree] < 0) ++min_degree;
    const int p = head[min_degree];
    remove(p);
    // The new element: variables of p's elements (absorbed) and its own variables.
    lp.clear();
    mark[p] = pass;
    int64_t lp_weight = 0;
    auto take = [&](int v) {
      if (state[v] == kVariable && mark[v] != pass) {
        mark[v] = pass;
        lp.push_back(v);
        lp_weight += nv[v];
      }
    };
    for (int e : elems[p]) {
      if (state[e] != kElement) continue;  // absorbed already (stale entry)
      for (int v : members[e]) take(v);
      members[e].clear();
      members[e].shrink_to_fit();
      state[e] = kMerged;  // absorbed into p
    }
    for (int v : vars[p]) take(v);
    vars[p].clear();
    vars[p].shrink_to_fit();
    elems[p].clear();
    elems[p].shrink_to_fit();
    state[p] = kElement;
    members[p] = lp;
    element_weight[p] = lp_weight;
    remaining -= nv[p];
    for (int v = p; v >= 0; v = merged_next[v]) order.push_back(nodes[v]);
    // Prune the adjacency of the element's variables and take them out of the
    // buckets. A variable with lists far longer than the average degree (a
    // pose of bundle adjustment collects an element per eliminated landmark)
    // is pruned only when its lists doubled, keeps the cheap degree bound and
    // is not merged: scanning its lists at every elimination would be
    // quadratic.
    heavy.assign(lp.size(), 0);
    for (size_t k = 0; k < lp.size(); ++k) {
      const int i = lp[k];
      remove(i);
      auto &e = elems[i];
      auto &a = vars[i];
      heavy[k] = e.size() + a.size() > long_lists;
      if (!heavy[k] || e.size() + a.size() >= 2 * pruned[i]) {
        e.erase(std::remove_if(e.begin(), e.end(), [&](int x) { return state[x] != kElement; }),
                e.end());
        a.erase(std::remove_if(a.begin(), a.end(),
                               [&](int x) { return state[x] != kVariable || mark[x] == pass; }),
                a.end());
        pruned[i] = e.size() + a.size() + 1;
      }
      e.push_back(p);
    }
    // |e \ Lp| for every other element adjacent to the new one (without the
    // heavy variables: an overestimate, which only bounds degrees from above
    // and skips absorptions).
    for (size_t k = 0; k < lp.size(); ++k) {
      if (heavy[k]) continue;
      const int i = lp[k];
      for (int e : elems[i]) {
        if (e == p) continue;
        if (stamp[e] != pass) {
          stamp[e] = pass;
          outside[e] = element_weight[e];
        }
        outside[e] -= nv[i];
      }
    }
    // Approximate external degrees; elements inside Lp are absorbed.
    for (size_t k = 0; k < lp.size(); ++k) {
      const int i = lp[k];
      const int64_t bound = std::min(degree[i] + lp_weight - nv[i], remaining - nv[i]);
      if (heavy[k]) {
        degree[i] = bound;
        continue;
      }
      int64_t d = lp_weight - nv[i];
      auto &e = elems[i];
      size_t keep = 0;
      for (int x : e) {
        if (x != p && (state[x] != kElement || outside[x] <= 0)) {
          state[x] = kMerged;  // aggressive absorption: x ⊂ Lp
          continue;
        }
        if (x != p) d += outside[x];
        e[keep++] = x;
      }
      e.resize(keep);
      for (int v : vars[i]) d += nv[v];
      degree[i] = std::min(d, bound);
    }
    // Indistinguishable variables: identical elements and variables (an
    // order-independent hash, then an exact comparison).
    hashes.clear();
    for (size_t k = 0; k < lp.size(); ++k) {
      if (heavy[k]) continue;
      const int i = lp[k];
      uint64_t h = elems[i].size() * 0x9E3779B97F4A7C15ull + vars[i].size();
      for (int x : elems[i]) h += Mix(static_cast<uint64_t>(x));
      for (int x : vars[i]) h += Mix(static_cast<uint64_t>(x) + (1ull << 40));
      hashes.push_back({h, i});
    }
    std::sort(hashes.begin(), hashes.end());
    for (size_t a = 0; a < hashes.size();) {
      size_t b = a + 1;
      while (b < hashes.size() && hashes[b].first == hashes[a].first) ++b;
      if (b - a > 1) {
        for (size_t x = a; x < b; ++x) {
          const int i = hashes[x].second;
          std::sort(elems[i].begin(), elems[i].end());
          std::sort(vars[i].begin(), vars[i].end());
        }
      }
      for (size_t x = a; x < b; ++x) {
        const int i = hashes[x].second;
        if (state[i] != kVariable) continue;
        for (size_t y = x + 1; y < b; ++y) {
          const int j = hashes[y].second;
          if (state[j] != kVariable || elems[i] != elems[j] || vars[i] != vars[j]) continue;
          // Merge j into i: j is eliminated right after i.
          int tail = i;
          while (merged_next[tail] >= 0) tail = merged_next[tail];
          merged_next[tail] = j;
          nv[i] += nv[j];
          degree[i] -= nv[j];
          nv[j] = 0;
          state[j] = kMerged;
          elems[j].clear();
          vars[j].clear();
        }
      }
      a = b;
    }
    for (int i : lp) {
      if (state[i] != kVariable) continue;
      degree[i] = std::max<int64_t>(0, std::min(degree[i], remaining - nv[i]));
      insert(i);
    }
    // Drop the merged variables from the element.
    auto &mp = members[p];
    mp.erase(std::remove_if(mp.begin(), mp.end(), [&](int x) { return state[x] != kVariable; }),
             mp.end());
  }
  return order;
}

/** Nested dissection with breadth-first level-set separators and minimum-degree leaves. */
class Dissection {
 public:
  Dissection(const Graph &g, int leaf_weight)
      : g_(g), leaf_(leaf_weight), set_(g.Size(), -1), level_(g.Size(), -1), local_(g.Size(), -1) {}

  std::vector<int> Order() {
    std::vector<int> all(g_.Size());
    for (int v = 0; v < g_.Size(); ++v) all[v] = v;
    Recurse(std::move(all));
    return std::move(order_);
  }

 private:
  int64_t Weight(const std::vector<int> &nodes) const {
    int64_t w = 0;
    for (int v : nodes) w += g_.weight[v];
    return w;
  }

  /** Breadth-first search from `root` within the set `id`; levels in level_, returns the visit
   * order. */
  std::vector<int> Bfs(int root, int id, int visit) {
    std::vector<int> visited{root};
    level_[root] = 0;
    set_[root] = visit;
    for (size_t head = 0; head < visited.size(); ++head) {
      const int u = visited[head];
      for (int v : g_.adj[u]) {
        if (set_[v] == id) {
          set_[v] = visit;
          level_[v] = level_[u] + 1;
          visited.push_back(v);
        }
      }
    }
    for (int v : visited) set_[v] = id;
    return visited;
  }

  /**
   * Orders a part that is not split further: minimum degree, unless the part
   * is nearly complete (any order then gives a dense factor, and minimum
   * degree would cost the cube of its size).
   */
  void Leaf(const std::vector<int> &nodes) {
    const int m = static_cast<int>(nodes.size());
    for (int v : nodes) local_[v] = 0;
    int64_t edges = 0;
    for (int v : nodes) {
      for (int u : g_.adj[v]) edges += local_[u] == 0;
    }
    for (int v : nodes) local_[v] = -1;
    if (edges >= static_cast<int64_t>(m) * (m - 1) * 3 / 4) {
      std::vector<int> o = nodes;
      std::sort(o.begin(), o.end());
      order_.insert(order_.end(), o.begin(), o.end());
      return;
    }
    std::vector<int> o = MinimumDegree(g_, nodes, local_);
    order_.insert(order_.end(), o.begin(), o.end());
  }

  void Recurse(std::vector<int> nodes) {
    if (nodes.empty()) return;
    const int64_t total = Weight(nodes);
    if (total <= leaf_) {
      Leaf(nodes);
      return;
    }
    const int id = next_id_++;
    for (int v : nodes) set_[v] = id;
    // Connected components are ordered independently.
    std::vector<int> first = Bfs(nodes[0], id, next_id_++);
    if (first.size() < nodes.size()) {
      const int done = next_id_++;
      std::vector<std::vector<int>> components;
      for (int v : nodes) {
        if (set_[v] != id) continue;
        components.push_back(Bfs(v, id, next_id_++));
        for (int u : components.back()) set_[u] = done;
      }
      for (auto &c : components) Recurse(std::move(c));
      return;
    }
    // Pseudo-peripheral root: restart from the farthest vertex while the depth grows.
    std::vector<int> visit = std::move(first);
    for (int attempt = 0; attempt < 4; ++attempt) {
      const int depth = level_[visit.back()];
      std::vector<int> again = Bfs(visit.back(), id, next_id_++);
      if (level_[again.back()] <= depth) {
        visit = Bfs(visit.front(), id, next_id_++);  // restore the deeper levels
        break;
      }
      visit = std::move(again);
    }
    const int depth = level_[visit.back()];
    if (depth < 2) {
      Leaf(nodes);
      return;
    }
    std::vector<int64_t> level_weight(depth + 1, 0);
    for (int v : visit) level_weight[level_[v]] += g_.weight[v];
    // The separator level minimizing its weight relative to the product of
    // the halves (a ratio cut): small separators near the root would peel
    // the part off a few vertices at a time.
    int64_t before = 0;
    double best = std::numeric_limits<double>::max();
    int cut = -1;
    for (int l = 1; l < depth; ++l) {
      before += level_weight[l - 1];
      const int64_t sep = level_weight[l], after = total - before - sep;
      if (after <= 0) break;
      double score =
          static_cast<double>(sep) / (static_cast<double>(before) * static_cast<double>(after));
      if (8 * std::min(before, after) < total) score *= 1e6;  // unbalanced: a last resort
      if (score < best) {
        best = score;
        cut = l;
      }
    }
    if (2 * level_weight[cut] > total) {
      Leaf(nodes);
      return;
    }
    std::vector<int> a, b, sep;
    for (int v : visit) {
      const int l = level_[v];
      (l < cut ? a : l > cut ? b : sep).push_back(v);
    }
    Recurse(std::move(a));
    Recurse(std::move(b));
    Leaf(sep);
  }

  const Graph &g_;
  const int leaf_;
  std::vector<int> set_, level_, local_;
  std::vector<int> order_;
  int next_id_ = 0;
};

/** Block (supervariable) structure of L for an ordering. */
struct Symbolic {
  std::vector<int> order;              ///< Position -> vertex.
  std::vector<std::vector<int>> rows;  ///< Per position: positions below the diagonal, ascending.
  std::vector<int> parent;             ///< Per position: elimination-tree parent, or -1.
  std::vector<std::pair<int, int>> supernodes;  ///< [first, last] positions.
  double flops = 0.0;
  int chain = 0;  ///< Longest etree path counted in panels (sequential steps).
};

Symbolic Factorize(const Graph &g, std::vector<int> order) {
  const int nb = g.Size();
  Symbolic s;
  std::vector<int> pos(nb);
  for (int k = 0; k < nb; ++k) pos[order[k]] = k;
  s.rows.resize(nb);
  s.parent.assign(nb, -1);
  std::vector<std::vector<int>> children(nb);
  std::vector<int> mark(nb, -1);
  for (int k = 0; k < nb; ++k) {
    std::vector<int> &r = s.rows[k];
    mark[k] = k;
    for (int u : g.adj[order[k]]) {
      const int p = pos[u];
      if (p > k && mark[p] != k) {
        mark[p] = k;
        r.push_back(p);
      }
    }
    for (int c : children[k]) {
      for (int p : s.rows[c]) {
        if (mark[p] != k) {
          mark[p] = k;
          r.push_back(p);
        }
      }
    }
    std::sort(r.begin(), r.end());
    if (!r.empty()) {
      s.parent[k] = r.front();
      children[r.front()].push_back(k);
    }
  }
  // Fundamental supernodes, then relaxed amalgamation: a supernode merges
  // into its parent supernode (the next one, when it is its parent) if that
  // saves a panel step and the explicit zeros stay under kRelaxedZeros of the
  // merged entries. Latency per step dominates on the GPU, flops are cheap.
  constexpr double kRelaxedZeros = 0.5;
  std::vector<int> wo(nb);
  for (int k = 0; k < nb; ++k) wo[k] = g.weight[order[k]];
  auto below_of = [&](int k) {
    double b = 0;
    for (int p : s.rows[k]) b += wo[p];
    return b;
  };
  auto panels = [](double width) {
    return static_cast<int>((width + kCholeskyPanelWidth - 1) / kCholeskyPanelWidth);
  };
  struct Node {
    int first, last;
    double width, below, zeros;
  };
  std::vector<Node> nodes;
  for (int k = 0; k < nb; ++k) {
    const bool extends =
        k > 0 && s.parent[k - 1] == k && s.rows[k - 1].size() == s.rows[k].size() + 1;
    if (extends) {
      nodes.back().last = k;
      nodes.back().width += wo[k];
      nodes.back().below = below_of(k);
    } else {
      nodes.push_back({k, k, static_cast<double>(wo[k]), below_of(k), 0.0});
    }
  }
  std::vector<Node> merged;
  for (const Node &node : nodes) {
    if (!merged.empty()) {
      Node &a = merged.back();
      if (s.parent[a.last] == node.first) {
        const double width = a.width + node.width;
        const double zeros = a.zeros + node.zeros + a.width * (node.width + node.below - a.below);
        const double entries = width * (width + 1) / 2 + width * node.below;
        if (panels(a.width) + panels(node.width) > panels(width) &&
            zeros <= kRelaxedZeros * entries) {
          a = {a.first, node.last, width, node.below, zeros};
          continue;
        }
      }
    }
    merged.push_back(node);
  }
  for (const Node &node : merged) {
    s.supernodes.push_back({node.first, node.last});
    for (int i = 0; i < node.width; ++i) {
      const double c = node.below + node.width - i;
      s.flops += c * c;
    }
  }
  std::vector<int> supernode_of(nb);
  for (size_t i = 0; i < s.supernodes.size(); ++i) {
    for (int k = s.supernodes[i].first; k <= s.supernodes[i].second; ++k) supernode_of[k] = i;
  }
  std::vector<int> critical(s.supernodes.size(), 0);
  for (size_t i = 0; i < s.supernodes.size(); ++i) {
    critical[i] += panels(merged[i].width);
    const int p = s.parent[s.supernodes[i].second];
    if (p >= 0) {
      int &c = critical[supernode_of[p]];
      c = std::max(c, critical[i]);
    }
    s.chain = std::max(s.chain, critical[i]);
  }
  s.order = std::move(order);
  return s;
}

/** Estimated factorization time [us]: a fixed latency per sequential panel step plus the flops. */
double EstimatedMicroseconds(const Symbolic &s) {
  constexpr double kStepLatency = 40.0;  // the dependent kernels of a level (measured)
  constexpr double kFlopsPerMicrosecond = 5e6;
  return s.chain * kStepLatency + s.flops / kFlopsPerMicrosecond;
}

}  // namespace

namespace {

/** The supervariable graph from its CSR adjacency. */
Graph MakeGraph(const std::vector<int> &supervariable_start,
                const std::vector<int> &adjacency_offsets, const std::vector<int> &adjacency) {
  Graph g;
  g.start = supervariable_start;
  const int nb = static_cast<int>(g.start.size()) - 1;
  g.weight.resize(nb);
  g.adj.resize(nb);
  for (int v = 0; v < nb; ++v) {
    g.weight[v] = g.start[v + 1] - g.start[v];
    g.adj[v].assign(adjacency.begin() + adjacency_offsets[v],
                    adjacency.begin() + adjacency_offsets[v + 1]);
  }
  return g;
}

/**
 * The symbolic factorization of the cheaper of two orderings by
 * EstimatedMicroseconds(): nested dissection, and minimum degree unless the
 * graph is nearly complete (any ordering then gives a dense factor). Sets the
 * ordering's name in `name`.
 */
Symbolic ChooseOrdering(const Graph &g, int64_t num_edges, const char **name) {
  const int nb = g.Size();
  Symbolic best = Factorize(g, Dissection(g, kCholeskyPanelWidth).Order());
  *name = "nested dissection";
  if (num_edges < static_cast<int64_t>(nb) * nb / 4) {
    std::vector<int> all(nb), local(nb, -1);
    for (int v = 0; v < nb; ++v) all[v] = v;
    Symbolic md = Factorize(g, MinimumDegree(g, all, local));
    if (EstimatedMicroseconds(md) < EstimatedMicroseconds(best)) {
      best = std::move(md);
      *name = "minimum degree";
    }
  }
  return best;
}

/**
 * The scalar permutation (supervariables in order, their columns consecutive)
 * and the panels: supernodes cut into kCholeskyPanelWidth-column pieces.
 * Returns the panel of each (permuted) column.
 */
std::vector<int> BuildPanels(const Graph &g, const Symbolic &s, CholeskyPlan &plan) {
  const int nb = g.Size();
  std::vector<int> first(nb + 1, 0);  // per position: first permuted column
  for (int k = 0; k < nb; ++k) first[k + 1] = first[k] + g.weight[s.order[k]];
  plan.perm.resize(plan.n);
  for (int k = 0; k < nb; ++k) {
    const int v = s.order[k];
    for (int t = 0; t < g.weight[v]; ++t) plan.perm[first[k] + t] = g.start[v] + t;
  }
  std::vector<int> col_panel(plan.n);
  for (const auto &[k0, k1] : s.supernodes) {
    const int c0 = first[k0], c1 = first[k1 + 1];
    std::vector<int> below;  // the rows below the supernode
    for (int p : s.rows[k1]) {
      for (int c = first[p]; c < first[p + 1]; ++c) below.push_back(c);
    }
    for (int c = c0; c < c1; c += kCholeskyPanelWidth) {
      CholeskyPlan::Panel panel;
      panel.col = c;
      panel.width = std::min(kCholeskyPanelWidth, c1 - c);
      panel.rows_offset = static_cast<int>(plan.below_rows.size());
      for (int r = c + panel.width; r < c1; ++r) plan.below_rows.push_back(r);
      plan.below_rows.insert(plan.below_rows.end(), below.begin(), below.end());
      panel.height = panel.width + static_cast<int>(plan.below_rows.size()) - panel.rows_offset;
      panel.offset = plan.factor_size;
      panel.diagonal_offset = plan.diagonal_size;
      plan.factor_size += static_cast<int64_t>(panel.height) * panel.width;
      plan.diagonal_size += static_cast<int64_t>(panel.width) * panel.width;
      std::fill(col_panel.begin() + c, col_panel.begin() + c + panel.width,
                static_cast<int>(plan.panels.size()));
      plan.panels.push_back(panel);
    }
  }
  if (plan.factor_size > INT_MAX) {
    throw std::runtime_error("BlockSparseCholeskySolver: the factor exceeds 2^31 entries");
  }
  return col_panel;
}

/**
 * The level schedule: a panel's level is one more than the highest level of
 * the panels that update it. Then per level the factor items, the runs of
 * each panel's below rows by target panel with their update items, and
 * whether no two panels of the level update the same target.
 */
void Schedule(const std::vector<int> &col_panel, CholeskyPlan &plan) {
  const int np = static_cast<int>(plan.panels.size());
  auto below_of = [&](int p) {
    const auto &panel = plan.panels[p];
    return std::make_pair(plan.below_rows.data() + panel.rows_offset, panel.height - panel.width);
  };
  std::vector<int> level(np, 0);
  for (int p = 0; p < np; ++p) {
    const auto [rows, count] = below_of(p);
    if (count > 0) {
      int &l = level[col_panel[rows[0]]];
      l = std::max(l, level[p] + 1);
    }
  }
  plan.num_levels = np > 0 ? *std::max_element(level.begin(), level.end()) + 1 : 0;

  std::vector<std::vector<CholeskyPlan::FactorItem>> factor_by_level(plan.num_levels);
  std::vector<std::vector<CholeskyPlan::UpdateItem>> update_by_level(plan.num_levels);
  std::vector<std::tuple<int, int, int>> targets;  // (level, target, source)
  int64_t relmap_size = 0;
  for (int p = 0; p < np; ++p) {
    const int l = level[p];
    const auto [rows, count] = below_of(p);
    if (count == 0) {
      factor_by_level[l].push_back({p, 0});
      continue;
    }
    for (int r = 0; r < count; r += kCholeskyTileRows) factor_by_level[l].push_back({p, r});
    for (int a = 0; a < count;) {
      const int q = col_panel[rows[a]];
      int b = a + 1;
      while (b < count && col_panel[rows[b]] == q) ++b;
      const int run = static_cast<int>(plan.runs.size());
      if (relmap_size + (count - a) > INT_MAX) {
        throw std::runtime_error(
            "BlockSparseCholeskySolver: the relative map exceeds 2^31 entries");
      }
      plan.runs.push_back({p, q, a, b, static_cast<int>(relmap_size)});
      relmap_size += count - a;
      for (int r = a; r < count; r += kCholeskyTileRows) update_by_level[l].push_back({p, run, r});
      targets.emplace_back(l, q, p);
      a = b;
    }
  }
  plan.relmap_size = static_cast<int>(relmap_size);
  std::sort(targets.begin(), targets.end());
  plan.update_exclusive.assign(plan.num_levels, 1);
  for (size_t i = 1; i < targets.size(); ++i) {
    const auto &[l0, q0, p0] = targets[i - 1];
    const auto &[l1, q1, p1] = targets[i];
    if (l0 == l1 && q0 == q1 && p0 != p1) plan.update_exclusive[l1] = 0;
  }
  for (int l = 0; l < plan.num_levels; ++l) {
    plan.factor_items.insert(plan.factor_items.end(), factor_by_level[l].begin(),
                             factor_by_level[l].end());
    plan.factor_offsets.push_back(static_cast<int>(plan.factor_items.size()));
    plan.update_items.insert(plan.update_items.end(), update_by_level[l].begin(),
                             update_by_level[l].end());
    plan.update_offsets.push_back(static_cast<int>(plan.update_items.size()));
  }
}

}  // namespace

CholeskyPlan AnalyzeCholesky(int n, const std::vector<int> &supervariable_start,
                             const std::vector<int> &adjacency_offsets,
                             const std::vector<int> &adjacency) {
  CholeskyPlan plan;
  plan.n = n;
  plan.factor_offsets = {0};
  plan.update_offsets = {0};
  if (n == 0) return plan;
  const Graph g = MakeGraph(supervariable_start, adjacency_offsets, adjacency);
  plan.num_supervariables = g.Size();
  const Symbolic s = ChooseOrdering(g, static_cast<int64_t>(adjacency.size()), &plan.ordering);
  plan.flops = s.flops;
  plan.num_supernodes = static_cast<int>(s.supernodes.size());
  Schedule(BuildPanels(g, s, plan), plan);
  return plan;
}

}  // namespace internal
}  // namespace cunls
