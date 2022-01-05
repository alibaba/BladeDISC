/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef DISC_TRANSFORMS_REWRITERS_H_
#define DISC_TRANSFORMS_REWRITERS_H_

namespace mlir {

class LLVMTypeConverter;
class SymbolTable;
class RewritePatternSet;
namespace bufferization {
class BufferizeTypeConverter;
}
class MLIRContext;

namespace disc_ral {

void populateDiscToLLVMConversionPatterns(LLVMTypeConverter* converter,
                                          SymbolTable* symbol_table,
                                          RewritePatternSet* patterns);
}  // namespace disc_ral

namespace mhlo_disc {

void populateDiscHLOToLHLOConversionPattern(
    MLIRContext* context, bufferization::BufferizeTypeConverter* converter,
    RewritePatternSet* patterns);

}  // namespace mhlo_disc

}  // namespace mlir

#endif  // DISC_TRANSFORMS_REWRITERS_H_
