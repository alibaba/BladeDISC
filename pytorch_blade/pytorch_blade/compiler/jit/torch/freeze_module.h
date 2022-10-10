// Copyright 2021 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/** \brief This file defines freezing Torchscript module API.
 *
 * This API has python-binding and can be invoked directly or as a part of
 * general optimization pipeline.
 */
#pragma once

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>

/** \brief Freeze Module, i.e., Assume all atrributes are constants.
 *
 * Freezing module is a functionality that allows the JIT to internalize
 * imutable attributes. Combined with inlinig, the module is aggressively
 * optimized and significant overhead is optimized away. The freezeModule API
 * produces a cloned frozen module.
 */

namespace torch {
namespace blade {
// This method was added because torch::jit::freeze_module would do
// peepholes shape optimization by default, which is not what our desired.
//
// For example,
// ```
// import torch
// class TestModel(torch.nn.Module):
//
//     def forward(self, x):
//         return x.size()
//
// x = torch.ones([1, 3, 224, 224])
// m = TestModel().eval()
// traced_m = torch.jit.trace(m, x)
// frozen_traced = torch._C._freeze_module(traced_m._c)
// print(traced_m.forward.graph)
// print(frozen_traced.forward.graph)
// ```
// Outputs:
// ```
// graph(%self : __torch__.___torch_mangle_11.TestModel,
//       %x : Float(1:150528, 3:50176, 224:224, 224:1)):
//   %3 : int = prim::Constant[value=0]() # <ipython-input-7-54fe5e9e31ae>:6:0
//   %4 : int = aten::size(%x, %3) # <ipython-input-7-54fe5e9e31ae>:6:0
//   %5 : Long() = prim::NumToTensor(%4)
//   %6 : int = prim::Constant[value=1]() # <ipython-input-7-54fe5e9e31ae>:6:0
//   %7 : int = aten::size(%x, %6) # <ipython-input-7-54fe5e9e31ae>:6:0
//   %8 : Long() = prim::NumToTensor(%7)
//   %9 : int = prim::Constant[value=2]() # <ipython-input-7-54fe5e9e31ae>:6:0
//   %10 : int = aten::size(%x, %9) # <ipython-input-7-54fe5e9e31ae>:6:0
//   %11 : Long() = prim::NumToTensor(%10)
//   %12 : int = prim::Constant[value=3]() # <ipython-input-7-54fe5e9e31ae>:6:0
//   %13 : int = aten::size(%x, %12) # <ipython-input-7-54fe5e9e31ae>:6:0
//   %14 : Long() = prim::NumToTensor(%13)
//   %15 : (Long(), Long(), Long(), Long()) = prim::TupleConstruct(%5, %8, %11,
//   %14) return (%15)
//
// graph(%self : __torch__.___torch_mangle_13.TestModel,
//       %x : Float(1:150528, 3:50176, 224:224, 224:1)):
//   %23 : (Long(), Long(), Long(), Long()) = prim::Constant[value=({1}, {3},
//   {224}, {224})]() return (%23)
// ```
// Like the example above, the original torch:::jit::freeze_module would do
// const folding on shapes. But we want the frozen module to run correctly on
// another shape.
//
// Therefore, we add another parameter disable_shape_peephole to the
// freeze_module, so that the user of freeze_module could control the behavior
// of shape relevance optimation.
//
// TODO: this function could be removed once torch::jit::freeze_module has the
// corresponding functionality.
torch::jit::Module freeze_module(
    const torch::jit::Module& module,
    std::vector<std::string> preservedAttrs = std::vector<std::string>(),
    bool freezeInterfaces = true,
    bool preserveParameters = false,
    bool disableShapePeephole = true);
} // namespace blade
} // namespace torch
