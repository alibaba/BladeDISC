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
include "mlir/Interfaces/SideEffectInterfaces.td"

def DISCShape_Dialect : Dialect {
  let name = "disc_shape";
  let cppNamespace = "::mlir::disc_shape";
}

class DISCShape_Op<string mnemonic, list<Trait> traits> :
    Op<DISCShape_Dialect, mnemonic, traits> {
  let hasVerifier = 1;
}

def DISCShape_LinearizeOp: DISCShape_Op<"linearize", [NoSideEffect, AttrSizedOperandSegments]> {
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

def DISCShape_DelinearizeOp: DISCShape_Op<"delinearize", [NoSideEffect]> {
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

def DISCShape_TieShapeOp: DISCShape_Op<"tie_shape", [NoSideEffect]> {
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

#endif // DISC_RAL_OPS
