# Copyright 2021 The BladeDISC Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import torch
import torch_blade.tools as tools

from torch.testing import FileCheck
from torch_blade.testing.common_utils import TestCase

class TestFreezeModule(TestCase):

    def test_freeze_module(self):
        class TestModel(torch.nn.Module):
            def forward(self, x):
                return x.size()

        x = torch.ones(1, 3, 224, 224)
        m = TestModel().eval()
        traced_m = torch.jit.trace(m, x)
        frozen_traced0 = torch._C._freeze_module(traced_m._c)
        frozen_traced1 = tools.freeze_module(traced_m._c, disableShapePeephole=False)
        frozen_traced2 = tools.freeze_module(traced_m._c, disableShapePeephole=True)

        shape_peephole_str = """
        graph(%self : __torch__.___torch_mangle_13.TestModel,
            %x : Float(1:150528, 3:50176, 224:224, 224:1)):
        # CHECK-COUNT-1: prim::Constant
        %23 : (Long(), Long(), Long(), Long()) = prim::Constant[value=({1}, {3}, {224}, {224})]()
        # CHECK: return
        return (%23)
        """ 
        non_shape_peephole_str = """graph(%self : __torch__.___torch_mangle_11.TestModel,
                     %x.1 : Tensor):
                 # CHECK-COUNT-4: prim::Constant
                 %3 : int = prim::Constant[value=0]() # <ipython-input-7-54fe5e9e31ae>:6:0
                 %7 : int = prim::Constant[value=1]() # <ipython-input-7-54fe5e9e31ae>:6:0
                 %11 : int = prim::Constant[value=2]() # <ipython-input-7-54fe5e9e31ae>:6:0
                 %15 : int = prim::Constant[value=3]() # <ipython-input-7-54fe5e9e31ae>:6:0
                 # CHECK: aten::size
                 %4 : int = aten::size(%x.1, %3) # <ipython-input-7-54fe5e9e31ae>:6:0
                 # CHECK: prim::NumToTensor
                 %5 : Tensor = prim::NumToTensor(%4) # :0:0
                 # CHECK: aten::size
                 %8 : int = aten::size(%x.1, %7) # <ipython-input-7-54fe5e9e31ae>:6:0
                 # CHECK: prim::NumToTensor
                 %9 : Tensor = prim::NumToTensor(%8) # :0:0
                 # CHECK: aten::size
                 %12 : int = aten::size(%x.1, %11) # <ipython-input-7-54fe5e9e31ae>:6:0
                 # CHECK: prim::NumToTensor
                 %13 : Tensor = prim::NumToTensor(%12) # :0:0
                 # CHECK: aten::size
                 %16 : int = aten::size(%x.1, %15) # <ipython-input-7-54fe5e9e31ae>:6:0
                 # CHECK: prim::NumToTensor
                 %17 : Tensor = prim::NumToTensor(%16) # :0:0
                 # CHECK: prim::TupleConstruct
                 %22 : (Tensor, Tensor, Tensor, Tensor) = prim::TupleConstruct(%5, %9, %13, %17)
                 # CHECK: return
                 return (%22)
                 """

        FileCheck().run(shape_peephole_str, str(frozen_traced0.forward.graph))
        FileCheck().run(shape_peephole_str, str(frozen_traced1.forward.graph))
        FileCheck().run(non_shape_peephole_str, str(frozen_traced2.forward.graph))

if __name__ == '__main__':
    unittest.main()
