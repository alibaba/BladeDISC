# Copyright 2023 The BladeDISC Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import hashlib
from torch_blade._torch_blade import _backends
HASH_SEED = 1

def data_hash(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16)

def hash_combine(a: int, b: int) -> int:
    return a ^ (b + 0x9e3779b9 + (a << 6) + (a >> 2))

def get_graph_hash(graph):
    nodes = []
    for node in graph.nodes():
        if node.kind() != 'prim::Constant':
            nodes.append(node)
    hash_value = HASH_SEED
    # hash input tensor info
    for input in graph.inputs():
        val_info = _backends.TensorInfo(input)
        rank = len(val_info.sizes)
        hash_value = hash_combine(hash_value, data_hash(val_info.dtype))
        hash_value = hash_combine(hash_value, data_hash(val_info.device))
        hash_value = hash_combine(hash_value, data_hash(str(rank)))
    # hash node info
    for n in nodes:
        hash_value = hash_combine(hash_value, data_hash(n.kind()))
    return hash_value