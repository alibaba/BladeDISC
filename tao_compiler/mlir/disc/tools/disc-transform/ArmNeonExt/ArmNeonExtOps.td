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

#ifndef DISC_ARMNEONEXT_OPS
#define DISC_ARMNEONEXT_OPS

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// ArmNeonExt dialect definition
//===----------------------------------------------------------------------===//

def DISCArmNeonExt_Dialect : Dialect {
  let name = "disc_arm_neon_ext";
  let cppNamespace = "::mlir::disc_ral::disc_arm_neon_ext";
  let useDefaultAttributePrinterParser = 0;
  let dependentDialects = ["arm_neon::ArmNeonDialect"];


  // Note: this does not need to depend on LLVMDialect as long as functions in
  // this dialect (such as canonicalization) do not produce entities belonging
  // to the LLVMDialect (ops or types).
}

//===----------------------------------------------------------------------===//
// ArmNeonExt op definitions
//===----------------------------------------------------------------------===//

// ArmNeonExt dialect op that corresponds (and is convertible to) an LLVM IR
// intrinsic.
class DISCArmNeonExt_IntrOp<string mnemonic, list<int> overloadedResults,
                     list<int> overloadedOperands, int numResults,
                     list<Trait> traits = [], bit requiresAccessGroup = 0,
                     bit requiresAliasAnalysis = 0>
    : LLVM_IntrOpBase</*dialect=*/DISCArmNeonExt_Dialect,
                      /*opName=*/"intr." # mnemonic,
                      /*enumName=*/"aarch64_neon_" # !subst(".", "_", mnemonic),
                      /*overloadedResults=*/overloadedResults,
                      /*overloadedOperands=*/overloadedOperands,
                      /*traits=*/traits,
                      /*numResults=*/numResults,
                      /*requiresAccessGroup=*/requiresAccessGroup,
                      /*requiresAliasAnalysis=*/requiresAliasAnalysis>;

// ArmNeon dialect op that corresponds to an LLVM IR intrinsic with one
// overloaded result.
class DISCArmNeonExt_OverloadedOneResultIntrOp<string mnemonic,
                                        list<Trait> traits = []>
  : DISCArmNeonExt_IntrOp<mnemonic, [0], [], 1, traits>;

// ArmNeon dialect op that corresponds to an LLVM IR intrinsic with one
// overloaded result and overloaded operands list.
class DISCArmNeonExt_OverloadedOperandsWithOneResultIntrOp<string mnemonic,
                                                    list<int> overloadedOperands,
                                                    list<Trait> traits = []>
  : DISCArmNeonExt_IntrOp<mnemonic, [0], overloadedOperands, 1, traits>;

def DISCArmNeonExt_BFMMLAOp : DISCArmNeonExt_OverloadedOperandsWithOneResultIntrOp<"bfmmla", [1], [
      Pure,
      AllTypesMatch<["a", "b"]>,
      AllTypesMatch<["c", "res"]>,
      TypesMatchWith<"res has the same number of elements as operand b",
                     "b", "res",
                     "VectorType::get({::llvm::cast<VectorType>($_self).getShape()[0] / 2},"
                     "FloatType::getF32($_self.getContext()))">]> {
  let summary = "bfmmla op";
  let description = [{"res = c + a * b"}];
  // Supports
  //   (vector<4xf32>, vector<8xbf16>, vector<8xbf16>) -> vector<4xf32>

  let arguments = (ins VectorOfLengthAndType<[4], [F32]>:$c,
                       VectorOfLengthAndType<[8], [BF16]>:$a,
                       VectorOfLengthAndType<[8], [BF16]>:$b);
  let results = (outs VectorOfLengthAndType<[4],  [F32]>:$res);
  let assemblyFormat =
    "$c `,` $a `,` $b `,` attr-dict `:` type($c) `,` type($a) `,` type($b) `,` `to` type($res)";
}

def DISCArmNeonExt_UZPOp : DISCArmNeonExt_OverloadedOperandsWithOneResultIntrOp<"uzp", [1], [
      Pure,
      AllTypesMatch<["a", "b"]>,
      AllTypesMatch<["a", "res"]>]> {
  let summary = "bfmmla op";
  let description = [{"res = uzp1(a, b) or uzp2(a, b)"}];

  let arguments = (ins AnyVectorOfAnyRank:$a,
                       AnyVectorOfAnyRank:$b,
                       BoolAttr:$isEven);
  let results = (outs AnyShaped:$res);
  let assemblyFormat =
    "$a `,` $b `,` attr-dict `:` type($a) `,` type($b) `,` `to` type($res)";
}

#endif  // DISC_ARMNEONGEXT_OPS
