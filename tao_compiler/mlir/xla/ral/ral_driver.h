//===- ral_driver.h ----------------------===//
//
// Copyright 2020 The PAI Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef RAL_RAL_DRIVER_H_
#define RAL_RAL_DRIVER_H_

#include "mlir/xla/ral/ral_base.h"

namespace tao {
namespace ral {

// An abstraction of a core device driver apis.
class Driver {
 public:
  virtual ~Driver() = default;
};

}  // namespace ral
}  // namespace tao

#endif  // RAL_RAL_DRIVER_H_