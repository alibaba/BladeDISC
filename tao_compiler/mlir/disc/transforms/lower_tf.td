/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

include "mlir/IR/OpBase.td"
include "mhlo/IR/hlo_ops.td"
include "mlir/disc/IR/hlo_disc_ops.td"
include "external/org_tensorflow/tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"

// Treated as passthrough IdentityOp
def : Pat<(TF_CheckNumericsOp $arg, $msg), (TF_IdentityOp $arg)>;
def : Pat<(TF_SnapshotOp $arg), (TF_IdentityOp $arg)>;
def : Pat<(TF_StopGradientOp $arg), (TF_IdentityOp $arg)>;

def IsShapedTensor
  : Constraint<CPred<"$0.getType().isa<RankedTensorType>()">>;

def PackRandomUniformInputs : NativeCodeCall<
  "PackRandomUniformInputs($0, $1, $2)">;

def PackRandomUniformBackendConfig : NativeCodeCall<
  "PackRandomUniformBackendConfig($0, $1, &$_builder)">;
