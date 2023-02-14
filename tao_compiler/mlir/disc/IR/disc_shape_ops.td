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

// This is the operation definition file for DISC shape related ops.

#ifndef DISC_SHAPE_OPS
#define DISC_SHAPE_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def DISCShape_Dialect : Dialect {
  let name = "disc_shape";
  let cppNamespace = "::mlir::disc_shape";
}

class DISCShape_Op<string mnemonic, list<Trait> traits> :
    Op<DISCShape_Dialect, mnemonic, traits> {
  let hasVerifier = 1;
}

def DISCShape_LinearizeOp: DISCShape_Op<"linearize", [Pure, AttrSizedOperandSegments]> {
  let summary = "calculate the linear index from multidim index";
  let description = [{
    Calculate the linear index using provided multidim indexes and shape.
  }];
  let arguments = (ins
    Arg<Variadic<Index>, "">:$multiDimIndexes,
    Arg<Variadic<Index>, "">:$shapeDimIndexes
  );
  let results = (outs Index);

  let hasCanonicalizer = 1;
}

def DISCShape_DelinearizeOp: DISCShape_Op<"delinearize", [Pure]> {
  let summary = "calculate the multidim index from linear index";
  let description = [{
    Calculate the multidim index using provided linear index and shape.
  }];
  let arguments = (ins
    Index:$linearIndex,
    Arg<Variadic<Index>, "">:$shapeDimIndexes
  );
  let results = (outs Variadic<Index>);

  let hasCanonicalizer = 1;
}

def DISCShape_TieShapeOp: DISCShape_Op<"tie_shape", [Pure]> {
  let summary = "binding a set of multidim index with a tensor value";
  let description = [{
    Binding a set of multidim index with a tensor value.
  }];
  let arguments = (ins
    TensorOf<[AnyType]>:$value,
    Arg<Variadic<Index>, "">:$shapeDimIndexes
  );
  let results = (outs TensorOf<[AnyType]>);

  let hasCanonicalizer = 1;
}

def DISCShape_SymbolicDimOp:
      DISCShape_Op<"SymbolicDim", [IsolatedFromAbove, Symbol]> {
  let summary = "representing a symbolic dimension size";
  let description = [{
    Representing a symbolic dimension size.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    DefaultValuedAttr<I64Attr, "ShapedType::kDynamic">:$value,
    DefaultValuedAttr<BoolAttr, "false">:$knownNonNegative,
    DefaultValuedAttr<BoolAttr, "false">:$knownNegativeOne,
    DefaultValuedAttr<BoolAttr, "false">:$knownNonSizeOne,
    DefaultValuedAttr<BoolAttr, "false">:$knownNonSizeZero
    // TODO(disc): add likely values
  );
  let extraClassDeclaration = [{
    // Returns the dimension value this SymbolicDim op represents.
    int64_t getDimSize();

    // Sets the dimension value this SymbolicDim op represents to `val`
    void setDimSize(int64_t val);

    // Returns true if this SymbolicDim is not known at compile-time.
    bool isDynamic();

    // Returns true if we know that the dimension value this SymbolicDim op
    // represents is size one.
    bool knownSizeOne() { return getDimSize() == 1; }

    // Sets `knownNonNegative` to the value of `flag`
    void updateKnownNonNegative(bool flag);

    // Sets `knownNegativeOne` to the value of `flag`
    void updateKnownNegativeOne(bool flag);

    // Sets `knownNonSizeOne` to the value of `flag`
    void updateKnownNonSizeOne(bool flag);

    // Sets `knownNonSizeZero` to the value of `flag`
    void updateKnownNonSizeZero(bool flag);

    // Try to merge two SymbolicDim ops.
    // Returns failure if fail to merge.
    LogicalResult Merge(SymbolicDimOp other);

    // Returns the name of attribute used for store the symbolicDim op reference attribute.
    static StringRef getSymbolicDimAttrName() { return "kDiscSymbolicDimAttr"; }
  }];

  let hasCanonicalizer = 1;
}

def DISCShape_DimOp:
      DISCShape_Op<"dim", [Pure]> {
  let summary = "binding to a symbolic dimension size";
  let description = [{
    Binding to a symbolic dimension size.
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$name
  );
  let results = (outs Index);
}

def SymbolicDimPredicateStructuredInterface : OpInterface<"PredicateOp"> {
  let cppNamespace = "::mlir::disc_shape";

  let methods = [
  ];
}

class DISCShape_PredicateOp<string mnemonic, list<Trait> traits> :
    DISCShape_Op<mnemonic, !listconcat(
      [SymbolicDimPredicateStructuredInterface], traits)> {
}

def DISCShape_TieProductEqualOp: DISCShape_Op<"tie_product_equal", [AttrSizedOperandSegments]> {
  let summary = "mark two groups x and y satisfy: product(x) == product(y)";
  let description = [{
    Mark two groups x and y satisfy: product(x) == product(y)
  }];
  let arguments = (ins
    Arg<Variadic<Index>, "">:$lhs,
    Arg<Variadic<Index>, "">:$rhs
  );
}

#endif // DISC_RAL_OPS
