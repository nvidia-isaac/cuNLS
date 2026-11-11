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

"""Small Warp helpers for custom pycunls factors: reading states through the state pointer table.

``WarpFactorBatch.evaluate`` receives ``state_pointers_ptr``: a device table of ``float*``, one
per (item, state), item-major (item t's k-th state is ``state_pointers[t * K + k]``). Wrap it once
with ``self.wrap_array(state_pointers_ptr, wp.uint64, num_items * K)`` and read the values inside
the kernel with :func:`state_value`; no gather pass or extra copy is needed.
"""

import warp as wp


@wp.func
def state_value(state_pointers: wp.array(dtype=wp.uint64), slot: int, component: int) -> float:
    """Component ``component`` of the state that ``state_pointers[slot]`` points to."""
    state = wp.array(ptr=state_pointers[slot], shape=(component + 1,), dtype=float)
    return state[component]
