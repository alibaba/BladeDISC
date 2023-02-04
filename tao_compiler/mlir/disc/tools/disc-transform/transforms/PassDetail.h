/* Copyright 2022 The BladeDISC Authors. All Rights Reserved.

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

#ifndef DISC_TOOLS_DISC_TRANSFORM_TRANSFORMS_PASSDETAIL_H_
#define DISC_TOOLS_DISC_TRANSFORM_TRANSFORMS_PASSDETAIL_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

class AffineDialect;

namespace NVVM {
class NVVMDialect;
}

namespace ROCDL {
class ROCDLDialect;
}

namespace math {
class MathDialect;
}

namespace memref {
class MemRefDialect;
}

namespace scf {
class SCFDialect;
}

namespace shape {
class ShapeDialect;
}

namespace mhlo {
class MhloDialect;
}

namespace arith {
class ArithDialect;
}

namespace linalg {
class LinalgDialect;
}

namespace disc_ral {

namespace disc_linalg_ext {
class DISCLinalgExtDialect;
}

#define GEN_PASS_CLASSES
#include "mlir/disc/tools/disc-transform/transforms/transform_passes.h.inc"

}  // namespace disc_ral
}  // namespace mlir

#endif  // DISC_TOOLS_DISC_TRANSFORM_TRANSFORMS_PASSDETAIL_H_
