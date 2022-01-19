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

#include "mlir/mhlo/builder/gru_cell.h"

#include "mlir/mhlo/builder/activation.h"
#include "mlir/mhlo/builder/constant.h"
#include "mlir/mhlo/builder/element_wise_binary.h"
#include "mlir/mhlo/builder/slice.h"

namespace mlir {
namespace mhlo {

/*
The BuildGRUCell builds the following computation graph:
```
def gru_cell(
  inp_gates: Tensor,
  h_gates: Tensor,
  h_x: Tensor,
  inp_bias: Tensor,
  h_bias: Tensor) -> Tensor:
    biased_input = inp_gates + inp_bias
    biased_hidden = h_gates + h_bias
    r_x, z_x, n_x = torch.chunk(biased_input, 3, 1)
    r_h, z_h, n_h = torch.chunk(biased_hidden, 3, 1)
    sigmoid_r = torch.sigmoid(r_x + r_h)
    sigmoid_z = torch.sigmoid_(z_x + z_h)
    tanh_n = torch.tanh(n_x + sigmoid_r * n_h)
    h = (h_x - tanh_n) * sigmoid_z + tanh_n
    return h
```
 */

mlir::Value BuildGRUCell(mlir::OpBuilder& builder, const mlir::Location& loc,
                         const mlir::Value& inp_gates,
                         const mlir::Value& h_gates, const mlir::Value& h_x,
                         const mlir::Value& inp_bias,
                         const mlir::Value& h_bias) {
  MHLO_CHECK(IsHloConstant(inp_bias), "The input bias must be constant");
  MHLO_CHECK(IsHloConstant(h_bias), "The hidden bias must be constant");
  auto inp_bias_ranked_type = GetMilrRankedTensorType(inp_bias);
  auto h_bias_ranked_type = GetMilrRankedTensorType(h_bias);
  auto inp_bias_rank = inp_bias_ranked_type.getRank();
  auto h_bias_rank = h_bias_ranked_type.getRank();

  MHLO_CHECK(inp_bias_rank > 0 && h_bias_rank > 0,
             "The ranks of biases must greater than 0");
  mlir_dim_t cell_dim_size_x3 =
      inp_bias_ranked_type.getDimSize(inp_bias_rank - 1);
  MHLO_CHECK(cell_dim_size_x3 == h_bias_ranked_type.getDimSize(h_bias_rank - 1),
             "The biases dim sizes mis-match");
  MHLO_CHECK((cell_dim_size_x3 % 3) == 0, "The cell dim is illegal");

  mlir_dim_t cell_dim_size_x1 = cell_dim_size_x3 / 3;
  auto elem_type = mlir::mhlo::GetMlirTensorElemType(inp_gates);

  // math: biased_input = inp_gates + inp_bias
  auto biased_input = BuildMlirBinaryOp<mlir::chlo::BroadcastAddOp>(
      builder, loc, inp_gates, inp_bias, elem_type,
      /* no_implicit_broadcast */ true);
  // math: biased_hidden = hidden_gates + h_bias
  auto biased_hidden = BuildMlirBinaryOp<mlir::chlo::BroadcastAddOp>(
      builder, loc, h_gates, h_bias, elem_type,
      /* no_implicit_broadcast */ true);

  auto std_zero = BuildStdConstForI64(builder, loc, 0);
  auto std_one = BuildStdConstForI64(builder, loc, 1);
  auto std_cell_size_x1 = BuildStdConstForI64(builder, loc, cell_dim_size_x1);
  auto std_cell_size_x2 =
      BuildStdConstForI64(builder, loc, cell_dim_size_x1 * 2);
  auto std_cell_size_x3 = BuildStdConstForI64(builder, loc, cell_dim_size_x3);

  mlir_dim_t dim_index = 1;
  // math: r_x, z_x, n_x = torch.chunk(biased_input, 3, 1)
  auto r_x = BuildDynamicSliceInternal(builder, loc, biased_input, std_zero,
                                       std_cell_size_x1, std_one, dim_index);
  auto z_x =
      BuildDynamicSliceInternal(builder, loc, biased_input, std_cell_size_x1,
                                std_cell_size_x2, std_one, dim_index);
  auto n_x =
      BuildDynamicSliceInternal(builder, loc, biased_input, std_cell_size_x2,
                                std_cell_size_x3, std_one, dim_index);

  // math: r_h, z_h, n_h = torch.chunk(biased_hidden, 3, 1)
  auto r_h = BuildDynamicSliceInternal(builder, loc, biased_hidden, std_zero,
                                       std_cell_size_x1, std_one, dim_index);
  auto z_h =
      BuildDynamicSliceInternal(builder, loc, biased_hidden, std_cell_size_x1,
                                std_cell_size_x2, std_one, dim_index);
  auto n_h =
      BuildDynamicSliceInternal(builder, loc, biased_hidden, std_cell_size_x2,
                                std_cell_size_x3, std_one, dim_index);

  // math: sigmoid_r = sigmoid(r_x + r_h)
  auto r_x_add_h = BuildMlirBinaryOp<mlir::chlo::BroadcastAddOp>(
      builder, loc, r_x, r_h, elem_type, /* no_implicit_broadcast */ true);
  auto sigmoid_r = BuildSigmoid(builder, loc, r_x_add_h);

  // math: sigmoid_z = sigmoid(z_x + z_h)
  auto z_x_add_h = BuildMlirBinaryOp<mlir::chlo::BroadcastAddOp>(
      builder, loc, z_x, z_h, elem_type, /* no_implicit_broadcast */ true);
  auto sigmoid_z = BuildSigmoid(builder, loc, z_x_add_h);

  // math: tanh_n = tanh(n_x + sigmoid_r * n_h)
  auto n_h_carry = BuildMlirBinaryOp<mlir::chlo::BroadcastMulOp>(
      builder, loc, sigmoid_r, n_h, elem_type,
      /* no_implicit_broadcast */ true);
  auto n_x_add_h_carry = BuildMlirBinaryOp<mlir::chlo::BroadcastAddOp>(
      builder, loc, n_x, n_h_carry, elem_type,
      /* no_implicit_broadcast */ true);
  auto tanh_n = builder.create<mlir::mhlo::TanhOp>(loc, n_x_add_h_carry);

  // math: h = (h_x - tanh_n) * sigmoid_z + tanh_n
  auto sub_hx_tanh = BuildMlirBinaryOp<mlir::chlo::BroadcastSubOp>(
      builder, loc, h_x, tanh_n, elem_type, /* no_implicit_broadcast */ true);
  auto mul_sigmoid_z = BuildMlirBinaryOp<mlir::chlo::BroadcastMulOp>(
      builder, loc, sub_hx_tanh, sigmoid_z, elem_type,
      /* no_implicit_broadcast */ true);
  return BuildMlirBinaryOp<mlir::chlo::BroadcastAddOp>(
      builder, loc, mul_sigmoid_z, tanh_n, elem_type,
      /* no_implicit_broadcast */ true);
}
}  // namespace mhlo
}  // namespace mlir
