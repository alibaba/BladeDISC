// Copyright 2021 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include <string>
#include <vector>

namespace mlir_test {

enum class BackendType;

bool feature_test_main(const std::string& mlir_file_path,
                       const std::vector<BackendType>& backend_types,
                       int num_inputs, int num_outputs,
                       const std::vector<std::string>& input_descriptors,
                       const std::vector<std::string>& output_descriptors,
                       const std::vector<std::vector<float>>& input_vals = {},
                       bool profiling = false, bool multi_cc_mode = false,
                       bool multi_cc_mode_dbg_ptx_only = false);

}  // namespace mlir_test
