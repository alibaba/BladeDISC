/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_MHLO_TRANSFORMS_DISC_PDL_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_MHLO_TRANSFORMS_DISC_PDL_UTILS_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

// This file implements some helper functions and classes used to support disc
// custom call.

namespace mlir {

// forward declaration
class DialectRegistry;
class RewritePatternSet;
class PDLPatternModule;

namespace disc_ral {

// Returns a unique `SmallVector<Value>` instance per thread per tag.
llvm::SmallVector<Value>& getThreadLocalValueRangeStorage(llvm::StringRef tag);

// Adds related depedent dialects (e.g. PDL dialect).
void getPDLDependentDialects(DialectRegistry& registry);

std::vector<std::string> ParseFileString(const std::string& str);

using RegisterPDLFunctionsCallback = std::function<void(PDLPatternModule&)>;

// Parses pdll patterns from string, compile them and then add to `patterns`.
LogicalResult populateDiscPdlPatternsFromString(
    RewritePatternSet* patterns, llvm::StringRef pdlPatterns,
    const std::vector<std::string>& includeDirs = {},
    const std::string& customPredefinedFunctionPrototypes = {},
    RegisterPDLFunctionsCallback callback = {});

// Parse pdll patterns from files, compile them and then add to `patterns`.
LogicalResult populateDiscPdlPatternsFromFiles(
    RewritePatternSet* patterns, const std::vector<std::string>& pdlFiles,
    const std::vector<std::string>& includeDirs = {},
    const std::string& customPredefinedFunctionPrototypes = {},
    RegisterPDLFunctionsCallback callback = {});

}  // namespace disc_ral
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_MHLO_TRANSFORMS_DISC_PDL_UTILS_H_
