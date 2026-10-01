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

// Minimal command-line parsing for the examples: `--key value` pairs from a
// fixed list of keys, plus `--help`.
//
//   examples::CommandLine cli(argc, argv, {"--num-points", "--outlier-ratio"});
//   const size_t n = cli.GetSize("--num-points", 2000);

#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace examples {

class CommandLine {
 public:
  // Parses argv. Prints usage and exits with 0 on --help; prints an error and
  // exits with 1 on an unknown key or a missing value.
  CommandLine(int argc, char **argv, std::vector<std::string> keys)
      : program_(argv[0]), keys_(std::move(keys)) {
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--help") {
        PrintUsage(std::cout);
        std::exit(0);
      }
      if (!IsKey(arg)) Fail("Unknown argument '" + arg + "'");
      if (i + 1 >= argc) Fail("Missing value for " + arg);
      values_[arg] = argv[++i];
    }
  }

  // Value of `key` as a positive integer, or `fallback` when absent.
  size_t GetSize(const std::string &key, size_t fallback) const {
    const auto it = values_.find(key);
    if (it == values_.end()) return fallback;
    char *end = nullptr;
    const long long v = std::strtoll(it->second.c_str(), &end, 10);
    if (end == it->second.c_str() || *end != '\0' || v <= 0) {
      Fail("Invalid " + key + " value '" + it->second + "' (expected a positive integer)");
    }
    return static_cast<size_t>(v);
  }

  // Value of `key` as a float, or `fallback` when absent.
  float GetFloat(const std::string &key, float fallback) const {
    const auto it = values_.find(key);
    if (it == values_.end()) return fallback;
    char *end = nullptr;
    const float v = std::strtof(it->second.c_str(), &end);
    if (end == it->second.c_str() || *end != '\0') {
      Fail("Invalid " + key + " value '" + it->second + "' (expected a number)");
    }
    return v;
  }

  // Value of `key` as a string, or `fallback` when absent.
  std::string GetString(const std::string &key, const std::string &fallback) const {
    const auto it = values_.find(key);
    return it == values_.end() ? fallback : it->second;
  }

  // Prints an error and the usage, then exits with 1.
  [[noreturn]] void Fail(const std::string &message) const {
    std::cerr << message << "\n";
    PrintUsage(std::cerr);
    std::exit(1);
  }

 private:
  bool IsKey(const std::string &arg) const {
    for (const auto &k : keys_) {
      if (k == arg) return true;
    }
    return false;
  }

  void PrintUsage(std::ostream &out) const {
    out << "Usage: " << program_;
    for (const auto &k : keys_) out << " [" << k << " VALUE]";
    out << "\n";
  }

  std::string program_;
  std::vector<std::string> keys_;
  std::map<std::string, std::string> values_;
};

}  // namespace examples
