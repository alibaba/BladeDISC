// Copyright 2023 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- ArmNeonDialect.h - MLIR Dialect forArmNeon ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Target dialect for ArmNeon in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef DISC_TOOLS_DISC_TRANSFORM_ARMNEONEXT_DIALECT_EXT_
#define DISC_TOOLS_DISC_TRANSFORM_ARMNEONEXT_DIALECT_EXT_

#include "mlir/Dialect/ArmNeon/ArmNeonDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/disc/tools/disc-transform/ArmNeonExt/ArmNeonExtDialect.h.inc"

//#define GET_OP_CLASSES
#include "mlir/disc/tools/disc-transform/ArmNeonExt/ArmNeonExtOps.h.inc"

#endif  // DISC_TOOLS_DISC_TRANSFORM_ARMNEONEXT_DIALECT_EXT_
