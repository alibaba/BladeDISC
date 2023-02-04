// Copyright 2022 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TORCH_DISC_PDLL_UTILS_H_
#define TORCH_DISC_PDLL_UTILS_H_

#include "mlir/disc/transforms/disc_pdl_utils.h"

namespace mlir {
class PDLPatternModule;
namespace torch {

extern const std::string kDefaultHelperFunctionDeclarations;

// Register some pre-defined helper functions for torch pdl patterns.
void registerPredefinedHelperFunctions(PDLPatternModule& pdlPatterns);

} // namespace torch
} // end namespace mlir

#endif // TORCH_DISC_PDLL_UTILS_H_
