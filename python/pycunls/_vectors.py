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

"""The numbered vector batch names (``VectorStateBatch3``, ``PriorVectorFactorBatch6``, ...).

Each is the runtime-sized class with its size fixed: ``VectorStateBatch3(data, capacity)`` is
``VectorStateBatch(data, capacity, dim=3)``. Any other size uses the runtime-sized class.
"""

import types

from pycunls import _pycunls_core as _core
from pycunls.catalog import VECTOR_DIMS

BASES = (_core.VectorStateBatch, _core.PriorVectorFactorBatch, _core.VectorBetweenFactorBatch,
         _core.BoundFactorBatch, _core.HalfspaceFactorBatch)


def _numbered(base: type, dim: int) -> type:
    name = f"{base.__name__}{dim}"

    def __init__(self, *args, **kwargs):
        if kwargs.setdefault("dim", dim) != dim:
            raise ValueError(f"{name}: got dim={kwargs['dim']} (expected {dim}). Fix: use "
                             f"pycunls.{base.__name__}(..., dim={kwargs['dim']}).")
        base.__init__(self, *args, **kwargs)

    __init__.__doc__ = f"The arguments of {base.__name__} without dim (fixed to {dim})."
    namespace = {"__init__": __init__, "__module__": "pycunls", "__qualname__": name,
                 "__doc__": f"{base.__name__} with dim={dim}: {name}(...) is "
                            f"{base.__name__}(..., dim={dim}). See help(pycunls.{base.__name__})."}
    return types.new_class(name, (base,), exec_body=lambda ns: ns.update(namespace))


NUMBERED = {f"{base.__name__}{dim}": _numbered(base, dim)
            for base in BASES for dim in VECTOR_DIMS}
