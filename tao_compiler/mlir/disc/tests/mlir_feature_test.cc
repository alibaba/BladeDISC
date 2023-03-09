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

#include "mlir/disc/tests/mlir_feature_test.h"

#include <stdio.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

using tensorflow::Env;
using tensorflow::ReadFileToString;
using tensorflow::WriteStringToFile;

namespace mlir_test {

std::string GenerateStrSeq(int num, const std::string& prefix,
                           const std::string& sep = ",") {
  std::vector<std::string> str_vec;
  str_vec.reserve(num);
  for (int i = 0; i < num; ++i) {
    str_vec.push_back(absl::StrCat(prefix, i));
  }
  return absl::StrJoin(str_vec, sep);
}

std::string DeviceTypeToStr(DeviceType dt) {
  if (dt == DeviceType::kCPU) {
    return "cpu";
  } else if (dt == DeviceType::kGPU) {
    return "gpu";
  } else {
    LOG(FATAL) << "unknown device type";
  }
}

std::string ReplaceTemplateValue(
    const std::string& tf_code, int num_inputs, int num_outputs,
    const std::vector<DeviceType>& input_placements,
    const std::vector<DeviceType>& output_placements) {
  std::string inputs_str = GenerateStrSeq(num_inputs, "input");
  std::string outputs_str = GenerateStrSeq(num_outputs, "output");

  std::vector<std::string> input_placement_vec;
  std::vector<std::string> output_placement_vec;
  std::for_each(input_placements.begin(), input_placements.end(),
                [&](DeviceType dt) {
                  input_placement_vec.push_back(DeviceTypeToStr(dt));
                });
  std::for_each(output_placements.begin(), output_placements.end(),
                [&](DeviceType dt) {
                  output_placement_vec.push_back(DeviceTypeToStr(dt));
                });

  std::string input_placements_str = absl::StrJoin(input_placement_vec, ",");
  std::string output_placements_str = absl::StrJoin(output_placement_vec, ",");

  return absl::StrReplaceAll(
      tf_code, {{"{{INPUTS}}", inputs_str},
                {"{{OUTPUTS}}", outputs_str},
                {"{{INPUT_PLACEMENTS}}", input_placements_str},
                {"{{OUTPUT_PLACEMENTS}}", output_placements_str}});
}

uint64_t getNextTestInstanceId() {
  static uint64_t next_index = 0;
  return next_index++;
}

bool feature_test_main(
    const std::string& mlir_file_path, BackendType backend_type, int num_inputs,
    int num_outputs, const std::vector<std::string>& input_descriptors,
    const std::vector<std::string>& output_descriptors,
    const std::vector<std::vector<float>>& input_vals,
    const std::vector<tensorflow::Tensor>& expected_output_vals,
    bool profiling = false, bool multi_cc_mode = false,
    bool multi_cc_mode_dbg_ptx_only = false) {
  std::vector<buffer_shape_t> input_shapes(num_inputs);
  std::vector<DataType> input_elem_types(num_inputs, tensorflow::DT_INVALID);
  std::vector<DeviceType> input_placement(num_inputs);
  std::vector<DataType> output_elem_types(num_outputs, tensorflow::DT_INVALID);
  std::vector<DeviceType> output_placement(num_outputs);
  for (int i = 0; i < num_inputs; ++i) {
    input_shapes[i] =
        ParseInputDescriptor(input_descriptors[i], backend_type,
                             &input_elem_types[i], &input_placement[i]);
  }
  for (int i = 0; i < num_outputs; ++i) {
    ParseOutputDescriptor(output_descriptors[i], backend_type,
                          &output_elem_types[i], &output_placement[i]);
  }
  const std::string tmp_dir = tensorflow::testing::TmpDir();
  std::string test_name =
      ::testing::UnitTest::GetInstance()->current_test_info()->name();
  test_name = absl::StrCat(test_name, "_", getNextTestInstanceId());

  if (input_descriptors.size() != num_inputs) {
    LOG(ERROR) << "The size of input_descriptor must be equal to num_inputs";
    return false;
  }
  if (output_descriptors.size() != num_outputs) {
    LOG(ERROR) << "The size of output_descriptor must be equal to num_outputs";
    return false;
  }
  if ((input_vals.size() != 0) && (input_vals.size() != num_inputs)) {
    LOG(ERROR)
        << "If assigned, the size of input_vals must be equal to num_inputs";
    return false;
  }
  if ((expected_output_vals.size() != 0) &&
      (expected_output_vals.size() != num_outputs)) {
    LOG(ERROR) << "If assigned, the size of expected_output_vals must be equal "
                  "to num_outputs";
    return false;
  }

  // Read tf code.
  std::string tf_code;
  auto tf_code_status =
      ReadFileToString(Env::Default(), mlir_file_path, &tf_code);
  if (!tf_code_status.ok()) {
    LOG(ERROR) << "failed to load mlir file from " << mlir_file_path << ": "
               << tf_code_status.error_message();
    return false;
  }
  LOG(INFO) << "Original TF code: " << tf_code;

  tf_code = ReplaceTemplateValue(tf_code, num_inputs, num_outputs,
                                 input_placement, output_placement);
  LOG(INFO) << "New TF code: " << tf_code;

  std::string new_mlir_file;
  if (!Env::Default()->LocalTempFilename(&new_mlir_file)) {
    LOG(ERROR) << "failed to create new temp file";
    return false;
  }

  if (!WriteStringToFile(Env::Default(), new_mlir_file, tf_code).ok()) {
    LOG(ERROR) << "failed to write file " << new_mlir_file;
    return false;
  }

  MlirTestImpl test(new_mlir_file, tmp_dir, test_name, num_inputs, num_outputs,
                    input_shapes, input_elem_types, input_placement, input_vals,
                    output_elem_types, output_placement, expected_output_vals,
                    profiling, multi_cc_mode, multi_cc_mode_dbg_ptx_only);
  auto status = test.Run();
  if (status != tsl::OkStatus()) {
    VLOG(0) << "[[FAILED]]: " << status.error_message();
  }
  return (status == tsl::OkStatus());
}

bool feature_test_read_input_from_file(const std::string& mlir_file_path,
                                       BackendType backend_type,
                                       const std::string& test_config_file,
                                       bool profiling) {
  int num_inputs;
  std::ifstream fin(test_config_file);
  fin >> num_inputs;
  std::vector<std::string> input_descriptors;
  std::vector<std::vector<float>> input_vals;
  for (int i = 0; i < num_inputs; ++i) {
    int nelem = 0;
    std::string dscrpt;
    fin >> dscrpt >> nelem;
    input_descriptors.push_back(dscrpt);
    input_vals.emplace_back();
    while (nelem-- > 0) {
      if (dscrpt.find("i32") != std::string::npos) {
        int v;
        fin >> v;
        input_vals.back().push_back(v);
      } else if (dscrpt.find("i1") != std::string::npos) {
        bool v;
        std::string v_str;
        fin >> v_str;
        std::transform(v_str.begin(), v_str.end(), v_str.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        v = (v_str == "1" || v_str == "true");
        input_vals.back().push_back(v);
      } else if (dscrpt.find("f32") != std::string::npos) {
        float v;
        std::string v_str;
        fin >> v_str;
        sscanf(v_str.c_str(), "%f", &v);
        if (std::isinf(v)) {
          VLOG(0) << "read a inf";
        }

        input_vals.back().push_back(v);
      } else {
        LOG(ERROR) << "unsupported type: " << dscrpt;
        return false;
      }
    }
  }

  int num_outputs;
  std::vector<std::string> output_descriptors;
  fin >> num_outputs;
  for (int i = 0; i < num_outputs; ++i) {
    std::string dscrpt;
    fin >> dscrpt;
    output_descriptors.push_back(dscrpt);
  }
  return feature_test_main(mlir_file_path, backend_type, num_inputs,
                           num_outputs, input_descriptors, output_descriptors,
                           input_vals, /*expected_output_vals*/ {}, profiling);
}

void addBoolFlags(EnvSettings& envSettings, const std::string& key) {
  char* value = getenv(key.c_str());
  if (value) {
    for (auto& setting : envSettings) {
      setting[key].first = value;
      setting[key].second = true;
    }
  } else {
    size_t original_size = envSettings.size();
    for (int i = 0; i < original_size; ++i) {
      envSettings[i][key].first = "false";
      envSettings.push_back(envSettings[i]);
      envSettings[i][key].first = "true";
    }
  }
}

EnvSettings getEnvironmentSettings() {
  EnvSettings envSettings{{}};
  addBoolFlags(envSettings, "DISC_ENABLE_STITCH");
  addBoolFlags(envSettings, "DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL");
  return envSettings;
}

bool feature_test_main(
    const std::string& mlir_file_path,
    const std::vector<BackendType>& backend_types, int num_inputs,
    int num_outputs, const std::vector<std::string>& input_descriptors,
    const std::vector<std::string>& output_descriptors,
    const std::vector<std::vector<float>>& input_vals,
    const std::vector<tensorflow::Tensor>& expected_output_vals, bool profiling,
    bool multi_cc_mode, bool multi_cc_mode_dbg_ptx_only) {
  bool pass = true;
  auto envSettings = getEnvironmentSettings();
  for (const auto& setting : envSettings) {
    EnvSettingContext ctx(setting);

    for (auto backend_type : backend_types) {
      if (backend_type == BackendType::kCuda) {
#if (GOOGLE_CUDA) || (TENSORFLOW_USE_ROCM)
        VLOG(0) << "Testing for CUDA backend";
        pass = pass && feature_test_main(
                           mlir_file_path, backend_type, num_inputs,
                           num_outputs, input_descriptors, output_descriptors,
                           input_vals, expected_output_vals, profiling,
                           multi_cc_mode, multi_cc_mode_dbg_ptx_only);
#endif
      } else if (backend_type == BackendType::kX86) {
#if TAO_CPU_ONLY and defined(TAO_X86)
        VLOG(0) << "Testing for X86 backend";
        pass = pass && feature_test_main(
                           mlir_file_path, backend_type, num_inputs,
                           num_outputs, input_descriptors, output_descriptors,
                           input_vals, expected_output_vals, profiling,
                           multi_cc_mode, multi_cc_mode_dbg_ptx_only);
#endif
      } else if (backend_type == BackendType::kAArch64) {
#if TAO_CPU_ONLY and defined(TAO_AARCH64)
        VLOG(0) << "Testing for AArch64 backend";
        pass = pass && feature_test_main(
                           mlir_file_path, backend_type, num_inputs,
                           num_outputs, input_descriptors, output_descriptors,
                           input_vals, expected_output_vals, profiling,
                           multi_cc_mode, multi_cc_mode_dbg_ptx_only);
#endif
      } else {
        LOG(ERROR) << "unknown backend type";
        return false;
      }
    }
  }
  return pass;
}
}  //  namespace mlir_test
