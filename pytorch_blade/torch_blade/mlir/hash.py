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
from torch_blade import hash_data, hash_combine
HASH_SEED = 1

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
        hash_value = hash_combine(hash_value, hash_data(val_info.dtype))
        hash_value = hash_combine(hash_value, hash_data(val_info.device))
        hash_value = hash_combine(hash_value, hash_data(str(rank)))
    # hash node info
    for n in nodes:
        hash_value = hash_combine(hash_value, hash_data(n.kind()))
        for input in n.inputs():
            hash_value = hash_combine(hash_value, hash_data(input.type().str()))
    return hash_value