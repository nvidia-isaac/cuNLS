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

"""Printing and pass/fail checks for the pycunls examples."""


def print_summary(title, summary, **extra):
    """Prints a minimizer summary followed by `extra` as "label: value" lines."""
    print(title)
    print(f"  Initial cost : {summary.initial_cost:.6f}")
    print(f"  Final cost   : {summary.final_cost:.6f}")
    print(f"  Iterations   : {summary.num_iterations}")
    for label, value in extra.items():
        print(f"  {label.replace('_', ' '):<13}: {value}")


def print_pose_error(label, rotation_deg, translation):
    print(f"  {label:<20}: rotation error {rotation_deg:.4f} deg, "
          f"translation error {translation:.4f}")


def check(ok, message):
    """Exits with an error when a quality check fails."""
    if not ok:
        raise SystemExit(f"Quality check failed: {message}")
