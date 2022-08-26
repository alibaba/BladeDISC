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

#include "tensorrt_onnx_parser.h"
#include <cmath>
#include <iterator>
#include "common_utils/utils.h"
#include "compiler/tensorrt/bridge/tensorrt_flags.h"

namespace torch {
namespace blade {
namespace {
bool GetSegmentList(
    const std::string& proto_bytes,
    SubGraphCollection_t& nodes_segment_list) {
  // the proto_bytes must be serialized from a valid onnx model
  // Get supported node list
  OnnxParserContext context;
  bool supported = context.parser->supportsModel(
      proto_bytes.data(), proto_bytes.size(), nodes_segment_list);

  if (!supported) {
    auto error_num = context.parser->getNbErrors();
    LOG(WARNING) << "TensorRT unsupported model, because of:";
    for (int i = 0; i < error_num; i++) {
      auto error_msg = context.parser->getError(i);
      LOG(WARNING) << "Reason " << i << ": " << error_msg->desc();
    }
  }

  if (supported && nodes_segment_list.size() == 0) {
    // assert the onnx model has graph nodes
    LOG(WARNING) << "The onnx version may be mis-match between torch_blade "
                    "and onnx-tenosrrt";
    return false;
  }
  return supported;
}
} // namespace

OnnxParserContext::OnnxParserContext() {
  TensorrtLogger& trt_logger = GetTensorrtLogger();
  builder = TrtUniquePtr<nvinfer1::IBuilder>(
      nvinfer1::createInferBuilder(trt_logger));
  const auto explicitBatch =
      1U << static_cast<uint32_t>(
          nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  network = TrtUniquePtr<nvinfer1::INetworkDefinition>(
      builder->createNetworkV2(explicitBatch));
  parser = TrtUniquePtr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network, trt_logger));
}
bool platformHasFastInt8() {
  OnnxParserContext context;
  return context.builder->platformHasFastInt8();
}

TrtUniquePtr<nvinfer1::ICudaEngine> TensorrtOnnxParser::BuildEngine(
    const std::string& proto_bytes,
    const std::shared_ptr<backends::EngineState>& state,
    const std::vector<DynamicRanges> dynamic_ranges,
    const QValType& q_val) {
  // Get supported node list
  OnnxParserContext context;
  bool supported =
      context.parser->parse(proto_bytes.data(), proto_bytes.size());

  if (supported) {
    CHECK_EQ(context.parser->getNbErrors(), 0);

    auto config = TrtUniquePtr<nvinfer1::IBuilderConfig>(
        context.builder->createBuilderConfig());
    config->setMaxWorkspaceSize(max_workspace_size_);
    if (dynamic_ranges.size() > 0) {
      auto changeDims = [](nvinfer1::Dims dims,
                           const std::vector<int64_t> shape) {
        CHECK_EQ(dims.nbDims, shape.size());
        // We traverse and modify the original dim instead of creating a new
        // dim. Because trt only has Dims2, Dims3 and Dims4
        for (int j = 0; j < dims.nbDims; j++) {
          dims.d[j] = shape[j];
        }
        return dims;
      };
      LOG(INFO) << "Building TRT engine with dynamic shapes.";
      int n_trt_input = context.network->getNbInputs();
      for (auto each_dynamic_range : dynamic_ranges) {
        if (!each_dynamic_range.Validate(n_trt_input)) {
          LOG(ERROR) << "Get a invalid dynamic setting, skip this:";
          LOG(ERROR) << each_dynamic_range.GetShapeString();
          return nullptr;
        }
        nvinfer1::IOptimizationProfile* profile =
            context.builder->createOptimizationProfile();
        for (int i = 0; i < n_trt_input; i++) {
          auto inp_name = context.network->getInput(i)->getName();
          auto inp_dims_trt = context.network->getInput(i)->getDimensions();
          auto inp_dims_min =
              changeDims(inp_dims_trt, each_dynamic_range.min_shape[i]);
          auto inp_dims_max =
              changeDims(inp_dims_trt, each_dynamic_range.max_shape[i]);
          profile->setDimensions(
              inp_name, nvinfer1::OptProfileSelector::kMIN, inp_dims_min);
          profile->setDimensions(
              inp_name, nvinfer1::OptProfileSelector::kMAX, inp_dims_max);
          for (auto opt : each_dynamic_range.opt_shapes) {
            auto inp_dims_opt = changeDims(inp_dims_trt, opt[i]);
            profile->setDimensions(
                inp_name, nvinfer1::OptProfileSelector::kOPT, inp_dims_opt);
          }
        }
        TORCH_CHECK(profile->isValid());
        config->addOptimizationProfile(profile);
      }
    }
    LOG(INFO) << "Creating TensorRT engine with BuilderFlags: "
              << GetBuilderFlags();
    config->setFlags(GetBuilderFlags());
    // enable tensorrt TF32 by default
    config->setFlag(nvinfer1::BuilderFlag::kTF32);
    auto calib_data = state->get_calib_data();
    // calibrator life time needs to last until after the engine is built.
    std::unique_ptr<nvinfer1::IInt8Calibrator> calibrator;
    if (!calib_data.empty()) {
      LOG(INFO) << "Building INT8 TensorRT engine with calibration data";
      config->setFlag(nvinfer1::BuilderFlag::kINT8);
      config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
      auto grp_calibrator = new Int8EntropyCalibrator2(calib_data);
      calibrator.reset(grp_calibrator);
      config->setInt8Calibrator(calibrator.get());
    }
    if (!q_val.empty()) {
      LOG(INFO) << "Building INT8 TensorRT engine. ";
      config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
      config->setFlag(nvinfer1::BuilderFlag::kINT8);
      if (!setDynamicRange(context.network, q_val)) {
        LOG(WARNING) << "Unable to set per tensor dynamic range";
      }
    }
    auto trt_engine = TrtUniquePtr<nvinfer1::ICudaEngine>(
        context.builder->buildEngineWithConfig(*context.network, *config));
    bool debug_log_flag =
        env::ReadBoolFromEnvVar("TORCH_BLADE_DEBUG_LOG", false);
    if (trt_engine == nullptr && debug_log_flag) {
      auto error_recorder = context.builder->getErrorRecorder();
      if (error_recorder != nullptr) {
        LOG(ERROR) << "Failed to build the engine, error message are:";
        auto nb_error = error_recorder->getNbErrors();
        for (int i = 0; i < nb_error; i++) {
          auto error_msg = error_recorder->getErrorDesc(i);
          LOG(ERROR) << error_msg;
        }
      } else {
        // Keep quiet when `error_recorder` is nullptr, error's will be written
        // to log stream said TensorRT doc:
        //    https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_builder.html#aa83591ea175b212a9d05ad938ebc7e24
      }
    }
    return trt_engine;
  } else {
    int nerror = context.parser->getNbErrors();
    for (int i = 0; i < nerror; ++i) {
      nvonnxparser::IParserError const* error = context.parser->getError(i);
      LOG(ERROR) << error->file() << ":" << error->line() << " In function "
                 << error->func() << ":\n"
                 << "[" << static_cast<int>(error->code()) << "] "
                 << error->desc();
    }
    return nullptr;
  }
}

bool TensorrtOnnxParser::IsSupportedModel(const std::string& proto_bytes) {
  SubGraphCollection_t nodes_segment_list;
  return GetSegmentList(proto_bytes, nodes_segment_list);
}

bool TensorrtOnnxParser::setDynamicRange(
    TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
    const QValType& q_val) {
  auto setTensorRange = [&](nvinfer1::ITensor* t, nvinfer1::ILayer* lyr) {
    const std::string tName = t->getName();
    auto layer_type = lyr->getType();
    auto ret = q_val.find(tName);
    if (ret != q_val.end()) {
      auto target_val = ret->second;
      return t->setDynamicRange(target_val.first, target_val.second);
    } else if (layer_type == nvinfer1::LayerType::kCONSTANT) {
      nvinfer1::IConstantLayer* cLyr =
          static_cast<nvinfer1::IConstantLayer*>(lyr);
      LOG(WARNING) << "Computing missing dynamic range for tensor " << tName
                   << " from weights.";
      auto wts = cLyr->getWeights();
      double max = std::numeric_limits<double>::min();
      for (size_t wb = 0, we = wts.count; wb < we; ++wb) {
        double val = 0;
        switch (wts.type) {
          case nvinfer1::DataType::kFLOAT:
            val = static_cast<const float*>(wts.values)[wb];
            break;
          case nvinfer1::DataType::kBOOL:
            val = static_cast<const bool*>(wts.values)[wb];
            break;
          case nvinfer1::DataType::kINT8:
            val = static_cast<const int8_t*>(wts.values)[wb];
            break;
          case nvinfer1::DataType::kINT32:
            val = static_cast<const int32_t*>(wts.values)[wb];
            break;
        }
        max = std::max(max, std::abs(val));
      }
      return t->setDynamicRange(-max, max);
    } else {
      LOG(WARNING) << "Missing dynamic range for tensor: " << tName;
    }
    return true;
  };

  for (int i = 0; i < network->getNbLayers(); ++i) {
    auto lyr = network->getLayer(i);
    for (int j = 0, e = lyr->getNbOutputs(); j < e; ++j) {
      auto output_tensor = lyr->getOutput(j);
      if (!setTensorRange(output_tensor, lyr)) {
        return false;
      }
    }

    auto type = lyr->getType();
    int num_input = lyr->getNbInputs();
    // getNbInputs will return 3 for some ISliceLayers but actually they only
    // have 1 input. In such case, it will raise runtime crash when look up the
    // second or later input. So here we only look up the first input for all
    // ISliceLayers. todo: find out this bug is caused by onnx or trt.
    if (type == nvinfer1::LayerType::kSLICE) {
      num_input = 1;
    }
    for (int j = 0; j < num_input; ++j) {
      auto input_tensor = lyr->getInput(j);
      if (!setTensorRange(input_tensor, lyr)) {
        return false;
      }
    }
  }
  return true;
}
} // namespace blade
} // namespace torch
