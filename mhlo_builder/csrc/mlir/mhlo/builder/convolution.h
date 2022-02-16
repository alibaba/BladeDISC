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

#pragma once
#include "mlir/mhlo/builder/mlir_type_utils.h"

namespace mlir {
namespace mhlo {

// Build Convolution with the following operands layouts:
// input layout: N, IC, Spatial dims
// output layout: N, OC, Spatial dims
// kenerl layout: OC, IC, Spatial dims
mlir::Value BuildConvolution(mlir::OpBuilder& builder,
                             const mlir::Location& loc,
                             const mlir::Value& input,
                             const mlir::Value& weight,
                             const mlir::Value& padding,
                             mlir::ArrayRef<int64_t> strides,
                             mlir::ArrayRef<int64_t> dilations);

}  // namespace mhlo
}  // namespace mlir
