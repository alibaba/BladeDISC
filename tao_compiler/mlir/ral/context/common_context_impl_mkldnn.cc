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

#if defined(TAO_CPU_ONLY) && defined(TAO_ENABLE_MKLDNN)

#include <sstream>

#if defined(TAO_X86)
#include "mkl.h"
#endif

#if defined(TAO_AARCH64)
#include "arm_compute/runtime/Scheduler.h"
#endif

#include "mlir/ral/context/common_context_impl_mkldnn.h"
#include "mlir/ral/context/mkldnn/ideep/ideep_pin_singletons.hpp"

namespace tao {
namespace ral {

namespace {

DiscCpuMathKernelMode initDiscCpuMathKernelMode() {
#if defined(TAO_AARCH64)
  // MKL is not supported on AArch64
  return kDiscPreferOneDNN;
#endif
  const char* env = getenv("DISC_CPU_MATH_KERNEL_MODE");
  std::string str = (env ? env : "");
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (str == "mkl") {
    TAO_VLOG(1) << "Use MKL as blas by default.";
    return kDiscPreferMKL;
  } else if (str == "onednn") {
    TAO_VLOG(1) << "Use onednn as blas by default.";
    return kDiscPreferOneDNN;
  } else if (str == "autotune") {
    TAO_VLOG(1) << "Use auto-tuning strategy for blas.";
    return kDiscPreferTuningBasedSelection;
  } else {
    // default use mkl
    TAO_VLOG(1) << "Use MKL as blas by default.";
    return kDiscPreferMKL;
  }
}

bool initPromoteConv1DToConv2D() {
  const char* env = getenv("DISC_CPU_PROMOTE_CONV_1D_TO_2D");
  // default is to do the promotion, same as TensorFlow.
  // Furthermore, it seems that oneDNN also has better performance when
  // we do the promotion.
  if (!env) return true;
  std::string envStr = env;
  std::transform(envStr.begin(), envStr.end(), envStr.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return envStr == "true" || envStr == "1";
}

bool initInOutLayoutTuningFlag() {
  const char* env = getenv("DISC_CPU_ENABLE_IN_OUT_LAYOUT_TUNING");
  if (!env) return false;
  std::string envStr = env;
  std::transform(envStr.begin(), envStr.end(), envStr.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return envStr == "true" || envStr == "1";
}

bool initEnableWeightPrePacking() {
  const char* env = getenv("DISC_CPU_ENABLE_WEIGHT_PRE_PACKING");
  // TODO(disc): change this to true once we can limit the max number of cache
  // memory used.
  if (!env) return false;
  std::string envStr = env;
  std::transform(envStr.begin(), envStr.end(), envStr.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return envStr == "true" || envStr == "1";
}

int initWeightPrePackingCacheCapacity() {
  const char* env = getenv("DISC_CPU_WEIGHT_PRE_PACKING_CACHE_CAPACITY");
  if (!env) return 1000;
  return std::atoi(env);
}

}  // namespace

#if defined(TAO_AARCH64)
bool applyACLThreadPoolConfigIfNotSet() {
  static bool status = []() {
    // make sure oneDNN ACL thread pool is configured first.
    // It'll be initialized inside the constructor of the engine.
    (void)ideep::engine::cpu_engine();
    // override ACL thread pool config if necessary.
    if (getNumAvailableCores() == 1) {
      // using single thread scheduler
      arm_compute::Scheduler::set(arm_compute::Scheduler::Type::ST);
      TAO_VLOG(1)
          << "Enforce to use single thread schedule due to `OMP_NUM_THREADS=1`";
    } else if (const char* scheduler_type = getenv("DISC_ACL_SCHEDULER_TYPE")) {
      if (std::strcmp(scheduler_type, "OMP") == 0) {
        arm_compute::Scheduler::set(arm_compute::Scheduler::Type::OMP);
        arm_compute::Scheduler::get().set_num_threads(getNumAvailableCores());
        TAO_VLOG(1) << "Use OMP thread schedule with " << getNumAvailableCores()
                    << " threads";
      }
    }
    return true;
  }();
  return status;
}
#endif

const std::thread::id kDiscCpuDefaultThreadId{};

DiscCpuMathKernelMode GetDiscCpuMathKernelMode() {
  static DiscCpuMathKernelMode mode = initDiscCpuMathKernelMode();
  return mode;
}

bool promoteConv1DToConv2D() {
  static bool enabled = initPromoteConv1DToConv2D();
  return enabled;
}

bool isWeightPrePackingEnabled() {
  static bool enabled = initEnableWeightPrePacking();
  return enabled;
}

bool isWeightPrePackingForMatMulEnabled() {
  static bool enabled = []() {
    const char* env = getenv("DISC_CPU_ENABLE_WEIGHT_PRE_PACKING_FOR_MATMUL");
    if (!env) return true;
    std::string envStr = env;
    std::transform(envStr.begin(), envStr.end(), envStr.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return envStr == "true" || envStr == "1";
  }();
  return isWeightPrePackingEnabled() && enabled;
}

int getWeightPrePackingCacheCapacity() {
  static int capacity = initWeightPrePackingCacheCapacity();
  return capacity;
}

bool enableInOutLayoutTuning() {
  static bool enabled = initInOutLayoutTuningFlag();
  return enabled;
}

format_tag str2format(const std::string& fmt) {
  if (fmt == "abcd") {
    return format_tag::abcd;
  } else if (fmt == "acdb") {
    return format_tag::acdb;
  } else if (fmt == "cdba") {
    return format_tag::cdba;
  } else if (fmt == "abc") {
    return format_tag::abc;
  } else if (fmt == "acb") {
    return format_tag::acb;
  } else if (fmt == "cba") {
    return format_tag::cba;
  } else if (fmt == "abcde") {
    return format_tag::abcde;
  } else if (fmt == "acdeb") {
    return format_tag::acdeb;
  } else if (fmt == "cdeba") {
    return format_tag::cdeba;
  }
  return format_tag::undef;
}

struct MkldnnConvState : public Context::Resource {
  std::mutex mu;
  std::unordered_map<opaque_t, std::vector<ideep::tensor>> packed_weight_cache;
};

#if defined(TAO_AARCH64)

// Returns true is ACL AMP is enabled (`DISC_CPU_ACL_USE_AMP`)
bool useAclAMP() {
  static bool flag = []() {
    const char* env = getenv("DISC_CPU_ACL_USE_AMP");
    if (!env) return false;
    std::string envStr = env;
    std::transform(envStr.begin(), envStr.end(), envStr.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return envStr == "true" || envStr == "1";
  }();
  return flag;
}

template <typename Tinput, int N, typename Tfilter = Tinput,
          typename Toutput = Tinput>
bool isAclSupportedDepthwiseConv(
    ExecutionContext* ctx, opaque_t /*stream_handle*/,
    MemRefType<Tinput, N> input, MemRefType<Tfilter, N> kernel,
    MemRefType<int32_t, 1> padding, MemRefType<Toutput, N> output,
    MemRefType<int32_t, 1> metadata, const ConvParams& params) {
  // TODO(disc): support other types.
  if (!std::is_same<Tinput, float>::value ||
      !std::is_same<Tfilter, float>::value ||
      !std::is_same<Toutput, float>::value) {
    return false;
  }
  // ACL currently do not support num_groups != 1
  if (params.groups != 1) {
    return false;
  }
  // NHWC + HWIO
  // TODO(disc): support other formats
  if (params.input_format == format_tag::acdb &&
      params.output_format == format_tag::acdb &&
      params.filter_format == format_tag::cdba && params.is_depthwise) {
    return true;
  }
  return false;
}

template <typename Tinput, int NDims, typename Tfilter = Tinput,
          typename Toutput = Tinput>
void runAclDepthwiseKernel(ExecutionContext* ctx, opaque_t /*stream_handle*/,
                           MemRefType<Tinput, NDims> input,
                           MemRefType<Tfilter, NDims> kernel,
                           MemRefType<int32_t, 1> padding,
                           MemRefType<Toutput, NDims> output,
                           MemRefType<int32_t, 1> metadata,
                           const ConvParams& params, CpuTimer& timer) {
  arm_compute::DataLayout data_layout = arm_compute::DataLayout::NHWC;

  auto src_dims = params.src.get_dims();
  auto dst_dims = params.dst.get_dims();
  auto weight_dims = params.weight.get_dims();
  int N = src_dims[0];
  int Ci = src_dims[1];
  int Ih = src_dims[2];
  int Iw = src_dims[3];
  int Co = dst_dims[1];
  int Oh = dst_dims[2];
  int Ow = dst_dims[3];
  int Kh = weight_dims[2];
  int Kw = weight_dims[3];

  if (TAO_VLOG_IS_ON(1)) {
    TAO_VLOG(0) << "N = " << N;
    TAO_VLOG(0) << "Ih = " << Ih;
    TAO_VLOG(0) << "Iw = " << Iw;
    TAO_VLOG(0) << "Ci = " << Ci;
    TAO_VLOG(1) << "Oh = " << Oh;
    TAO_VLOG(1) << "Ow = " << Ow;
    TAO_VLOG(0) << "Co = " << Co;
    TAO_VLOG(0) << "Kh = " << Kh;
    TAO_VLOG(0) << "Kw = " << Kw;
  }

  auto src_shape = arm_compute::TensorShape(Ci, Iw, Ih, N);
  auto weights_shape = arm_compute::TensorShape(Co, Kw, Kh, 1);
  auto dst_shape = arm_compute::TensorShape(Co, Ow, Oh, N);

  arm_compute::DataType data_type = arm_compute::DataType::F32;
  auto src_info = arm_compute::TensorInfo(src_shape, 1, data_type, data_layout);
  auto weights_info =
      arm_compute::TensorInfo(weights_shape, 1, data_type, data_layout);
  auto dst_info = arm_compute::TensorInfo(dst_shape, 1, data_type, data_layout);

  arm_compute::Tensor src, weights, dst;
  src.allocator()->init(src_info);
  weights.allocator()->init(weights_info);
  dst.allocator()->init(dst_info);
  src.allocator()->import_memory(input.data);
  weights.allocator()->import_memory(kernel.data);
  dst.allocator()->import_memory(output.data);

  auto AclDepthwiseConvCreator = [&](const arm_compute::ITensorPack* pack) {
    std::shared_ptr<AclDepthwiseConvInfo> info(new AclDepthwiseConvInfo);
    if (!info->op.validate(
            &src_info, &weights_info, nullptr, &dst_info,
            arm_compute::PadStrideInfo{
                params.strides[1], params.strides[0], params.padding_l[1],
                params.padding_r[1], params.padding_l[0], params.padding_r[0],
                arm_compute::DimensionRoundingType::FLOOR},
            /* multiplier */ Co / Ci, arm_compute::ActivationLayerInfo{},
            arm_compute::Size2D{params.dilates[1], params.dilates[0]})) {
      ctx->signalError(Context::FAILURE, "fail to validate acl depthwise conv");
    } else {
      info->op.configure(
          &src, &weights, nullptr, &dst,
          arm_compute::PadStrideInfo{params.strides[1], params.strides[0],
                                     params.padding_l[1], params.padding_r[1],
                                     params.padding_l[0], params.padding_r[0],
                                     arm_compute::DimensionRoundingType::FLOOR},
          /* multiplier */ Co / Ci, arm_compute::ActivationLayerInfo{},
          arm_compute::Size2D{params.dilates[1], params.dilates[0]});
    }
    if (pack) info->op.reuse_packed_weight(*pack);
    info->op.prepare(&src, &weights, nullptr, &dst);

    return info;
  };

  std::shared_ptr<AclDepthwiseConvInfo> info;
  std::shared_ptr<AclDepthwiseConvThreadSafeInfo> thread_safe_info;
  if (isWeightPrePackingEnabled() && params.weight_is_const) {
    std::string unique_name = "disc.ral.cpu.acl_depthwise_conv";
    auto state = ctx->getOrCreateResource<AclDepthwiseConvState>(
        unique_name, []() { return new AclDepthwiseConvState; });
    auto key = makeConvParamsKey(input, kernel, padding, output, metadata,
                                 kDiscCpuDefaultThreadId);
    auto dynamicKey = makeDynamicShapeConvParamsKey(
        input, kernel, padding, output, metadata, kDiscCpuDefaultThreadId);
    thread_safe_info = state->getOrCreate(dynamicKey);
    info = thread_safe_info->getOrCreate(key, AclDepthwiseConvCreator);
  } else {
    info = AclDepthwiseConvCreator(nullptr);
  }

  info->op.run(&src, &weights, nullptr, &dst);

  timer.Stop();
  if (isProfilingEnabled()) {
    dumpConvLikeKernelProflingInfo<Tinput, Tfilter, Toutput>(
        params, timer.GetNanoSeconds(), "ral_cpu_acl_depthwise_conv");
  }
}

template <typename Tinput, int N, typename Tfilter = Tinput,
          typename Toutput = Tinput>
bool isAclSupportedConv(ExecutionContext* ctx, opaque_t /*stream_handle*/,
                        MemRefType<Tinput, N> input,
                        MemRefType<Tfilter, N> kernel,
                        MemRefType<int32_t, 1> padding,
                        MemRefType<Toutput, N> output,
                        MemRefType<int32_t, 1> metadata,
                        const ConvParams& params) {
  // TODO(disc): support other types.
  if (!std::is_same<Tinput, float>::value ||
      !std::is_same<Tfilter, float>::value ||
      !std::is_same<Toutput, float>::value) {
    return false;
  }
  // ACL currently do not support num_groups != 1
  if (params.groups != 1) {
    return false;
  }
  // NHWC + OHWI
  // TODO(disc): support other formats
  if (params.input_format == format_tag::acdb &&
      params.output_format == format_tag::acdb &&
      params.filter_format == format_tag::acdb) {
    return true;
  }
  return false;
}

/**
 * Valid data type configurations:
 * |src0         |src1        |src2      |dst            |
 * |:------------|:-----------|:---------|:--------------|
 * |F16          |F16         |F16       |F16            |
 * |BFLOAT16     |BFLOAT16    |BFLOAT16  |FP32           |
 */
template <typename Tinput, int NDims = 2, typename Tweight = Tinput,
          typename Toutput = Tinput>
void runAclAMPGemmKernel(ExecutionContext* ctx, opaque_t /*stream_handle*/,
                         MemRefType<Tinput, NDims> input,
                         MemRefType<Tweight, NDims> weight,
                         MemRefType<Toutput, NDims> result, bool tp_a,
                         bool tp_b, bool weight_is_const) {
  CpuTimer timer("ral_acl_amp_gemm");
  if (isEmptyMemref(input) || isEmptyMemref(weight) || isEmptyMemref(result)) {
    TAO_VLOG(1) << "ral_cpu_amp_gemm: early return for empty tensor";
    return;
  }

  // ACL GEMM kernel does not support transpose.
  // TODO(disc): use standalone ACL transpose kernel to imitate.
  if (tp_a || tp_b) {
    ctx->signalError(Context::FAILURE,
                     "not supported ral_amp_gemm with transpose");
  }

  int64_t m = input.sizes[0];
  int64_t k = input.sizes[1];
  if (k != weight.sizes[0]) {
    ctx->signalError(Context::FAILURE, "mismatch contraction dim for gemm");
    return;
  }
  int64_t n = weight.sizes[1];
  bool is_bf16_gemm = toDataType<Tinput>() == data_type::bf16 ? true : false;
  auto AclGemmCreator = [&](const arm_compute::ITensorPack* pack) {
    std::shared_ptr<AclGemmInfo> info(new AclGemmInfo);
    auto src_shape = arm_compute::TensorShape(k, m);
    auto weight_shape = arm_compute::TensorShape(n, k);
    auto dst_shape = arm_compute::TensorShape(n, m);
    arm_compute::DataType input_data_type, res_data_type;
    if (is_bf16_gemm) {
      input_data_type = arm_compute::DataType::BFLOAT16;
      res_data_type = arm_compute::DataType::F32;
    } else {
      input_data_type = arm_compute::DataType::F16;
      res_data_type = arm_compute::DataType::F16;
    }
    arm_compute::TensorInfo src_info =
        arm_compute::TensorInfo(src_shape, 1, input_data_type);
    arm_compute::TensorInfo weight_info =
        arm_compute::TensorInfo(weight_shape, 1, input_data_type);
    arm_compute::TensorInfo dst_info =
        arm_compute::TensorInfo(dst_shape, 1, res_data_type);

    info->src.allocator()->init(src_info);
    info->weights.allocator()->init(weight_info);
    info->dst.allocator()->init(dst_info);
    info->src.allocator()->import_memory(
        reinterpret_cast<uint16_t*>(input.data));
    info->weights.allocator()->import_memory(
        reinterpret_cast<uint16_t*>(weight.data));
    info->dst.allocator()->import_memory(
        reinterpret_cast<uint16_t*>(result.data));
    auto status = info->op.validate(&src_info, &weight_info, nullptr, &dst_info,
                                    1.0, 0.0);
    if (!status) {
      ctx->signalError(Context::FAILURE, "fail to validate acl negemm");
    } else {
      info->op.configure(&info->src, &info->weights, nullptr, &info->dst, 1.0,
                         0.0);
    }
    if (pack) info->op.reuse_packed_weight(*pack);
    info->op.prepare(&info->src, &info->weights, nullptr, &info->dst);
    return info;
  };
  // TODO: support weight prepacking
  std::shared_ptr<AclGemmInfo> info;
  info = AclGemmCreator(nullptr);
  info->op.run(&info->src, &info->weights, nullptr, &info->dst);
  timer.Stop();
  if (isProfilingEnabled()) {
    int64_t bytes = sizeof(uint16_t) * m * k + sizeof(uint16_t) * k * n +
                    sizeof(uint16_t) * m * n;
    std::string amp_gemm_type = is_bf16_gemm ? "ral_cpu_amp_gemm_bf16_bf16_fp32"
                                             : "ral_cpu_amp_gemm_f16_f16_f16";

    TAO_VLOG(0) << amp_gemm_type << ":\n"
                << "\tpa = " << input.data << "\n"
                << "\tpb = " << weight.data << "\n"
                << "\tpc = " << result.data << "\n"
                << "\tm = " << m << "\n"
                << "\tn = " << n << "\n"
                << "\tk = " << k << "\n"
                << "\ttp_a = " << tp_a << "\n"
                << "\ttp_b = " << tp_b << "\n"
                << "\tweight_is_const = " << weight_is_const << "\n"
                << "\tMath Ops = " << 2 * m * n * k << "\n"
                << "\tBytes = " << bytes << "\n"
                << "\tBandwidth = "
                << double(bytes) / double(timer.GetNanoSeconds()) << " GB\n"
                << "\tGFLOPS = "
                << double(2 * m * n * k) / double(timer.GetNanoSeconds())
                << "\n";
  }
}

template <typename Tinput, int NDims, typename Tfilter = Tinput,
          typename Toutput = Tinput>
void runAclConvKernel(ExecutionContext* ctx, opaque_t /*stream_handle*/,
                      MemRefType<Tinput, NDims> input,
                      MemRefType<Tfilter, NDims> kernel,
                      MemRefType<int32_t, 1> padding,
                      MemRefType<Toutput, NDims> output,
                      MemRefType<int32_t, 1> metadata, const ConvParams& params,
                      CpuTimer& timer) {
  auto src_dims = params.src.get_dims();
  auto dst_dims = params.dst.get_dims();
  auto weight_dims = params.weight.get_dims();
  int N = src_dims[0];
  int Ci = src_dims[1];
  int Ih = src_dims[2];
  int Iw = src_dims[3];
  int Co = dst_dims[1];
  int Oh = dst_dims[2];
  int Ow = dst_dims[3];
  int Kh = weight_dims[2];
  int Kw = weight_dims[3];

  if (TAO_VLOG_IS_ON(1)) {
    TAO_VLOG(1) << "N = " << N;
    TAO_VLOG(1) << "Ih = " << Ih;
    TAO_VLOG(1) << "Iw = " << Iw;
    TAO_VLOG(1) << "Ci = " << Ci;
    TAO_VLOG(1) << "Oh = " << Oh;
    TAO_VLOG(1) << "Ow = " << Ow;
    TAO_VLOG(1) << "Co = " << Co;
    TAO_VLOG(1) << "Kh = " << Kh;
    TAO_VLOG(1) << "Kw = " << Kw;
    TAO_VLOG(0) << "params.strides[1] = " << params.strides[1];
    TAO_VLOG(0) << "params.strides[0] = " << params.strides[0];
    TAO_VLOG(0) << "params.padding_l[1] = " << params.padding_l[1];
    TAO_VLOG(0) << "params.padding_l[0] = " << params.padding_l[0];
    TAO_VLOG(0) << "params.padding_r[1] = " << params.padding_r[1];
    TAO_VLOG(0) << "params.padding_r[0] = " << params.padding_r[0];
    TAO_VLOG(0) << "params.dilates[1] = " << params.dilates[1];
    TAO_VLOG(0) << "params.dilates[0] = " << params.dilates[0];
  }

  arm_compute::DataType data_type = arm_compute::DataType::F32;
  arm_compute::DataLayout data_layout = arm_compute::DataLayout::NHWC;
  auto src_shape = arm_compute::TensorShape(Ci, Iw, Ih, N);
  auto weights_shape = arm_compute::TensorShape(Ci, Kw, Kh, Co);
  auto dst_shape = arm_compute::TensorShape(Co, Ow, Oh, N);

  auto src_info = arm_compute::TensorInfo(src_shape, 1, data_type, data_layout);
  auto weights_info =
      arm_compute::TensorInfo(weights_shape, 1, data_type, data_layout);
  auto dst_info = arm_compute::TensorInfo(dst_shape, 1, data_type, data_layout);

  arm_compute::Tensor src, weights, dst;
  src.allocator()->init(src_info);
  weights.allocator()->init(weights_info);
  dst.allocator()->init(dst_info);
  src.allocator()->import_memory(input.data);
  weights.allocator()->import_memory(kernel.data);
  dst.allocator()->import_memory(output.data);

  auto AclConvCreator = [&](const arm_compute::ITensorPack* pack) {
    std::shared_ptr<AclConvInfo> info(new AclConvInfo);

    bool use_fast_math =
        useAclAMP() &&
        info->op.validate(
            &src_info, &weights_info, nullptr, &dst_info,
            arm_compute::PadStrideInfo{
                params.strides[1], params.strides[0], params.padding_l[1],
                params.padding_r[1], params.padding_l[0], params.padding_r[0],
                arm_compute::DimensionRoundingType::FLOOR},
            arm_compute::WeightsInfo{},
            arm_compute::Size2D{params.dilates[1], params.dilates[0]},
            arm_compute::ActivationLayerInfo{}, /*enable_fast_math*/ true);

    if (!info->op.validate(
            &src_info, &weights_info, nullptr, &dst_info,
            arm_compute::PadStrideInfo{
                params.strides[1], params.strides[0], params.padding_l[1],
                params.padding_r[1], params.padding_l[0], params.padding_r[0],
                arm_compute::DimensionRoundingType::FLOOR},
            arm_compute::WeightsInfo{},
            arm_compute::Size2D{params.dilates[1], params.dilates[0]},
            arm_compute::ActivationLayerInfo{}, use_fast_math)) {
      ctx->signalError(Context::FAILURE, "fail to validate acl depthwise conv");
    } else {
      info->op.configure(
          &src, &weights, nullptr, &dst,
          arm_compute::PadStrideInfo{params.strides[1], params.strides[0],
                                     params.padding_l[1], params.padding_r[1],
                                     params.padding_l[0], params.padding_r[0],
                                     arm_compute::DimensionRoundingType::FLOOR},
          arm_compute::WeightsInfo{},
          arm_compute::Size2D{params.dilates[1], params.dilates[0]},
          arm_compute::ActivationLayerInfo{}, use_fast_math);
    }
    if (pack) info->op.reuse_packed_weight(*pack);
    info->op.prepare(&src, &weights, nullptr, &dst);

    return info;
  };

  std::shared_ptr<AclConvInfo> info;
  std::shared_ptr<AclConvThreadSafeInfo> thread_safe_info;
  if (isWeightPrePackingEnabled() && params.weight_is_const) {
    std::string unique_name = "tao_ral.cpu.acl_conv";
    auto state = ctx->getOrCreateResource<AclConvState>(
        unique_name, []() { return new AclConvState; });
    auto key = makeConvParamsKey(input, kernel, padding, output, metadata,
                                 kDiscCpuDefaultThreadId);
    auto dynamicKey = makeDynamicShapeConvParamsKey(
        input, kernel, padding, output, metadata, kDiscCpuDefaultThreadId);
    thread_safe_info = state->getOrCreate(dynamicKey);
    info = thread_safe_info->getOrCreate(key, AclConvCreator);
  } else {
    info = AclConvCreator(nullptr);
  }

  info->op.run(&src, &weights, nullptr, &dst);

  timer.Stop();
  if (isProfilingEnabled()) {
    dumpConvLikeKernelProflingInfo<Tinput, Tfilter, Toutput>(
        params, timer.GetNanoSeconds(), "ral_cpu_acl_conv");
  }
}

#endif

template <typename Tinput, int N, typename Tfilter = Tinput,
          typename Toutput = Tinput>
void ral_conv(ExecutionContext* ctx, opaque_t stream_handle,
              MemRefType<Tinput, N> input, MemRefType<Tfilter, N> kernel,
              MemRefType<int32_t, 1> padding, MemRefType<Toutput, N> output,
              MemRefType<int32_t, 1> metadata) {
  CpuTimer timer("ral_cpu_conv");
  if (isEmptyMemref(input) || isEmptyMemref(kernel) || isEmptyMemref(output)) {
    TAO_VLOG(1) << "ral_conv: early return for empty tensor";
    return;
  }

  ConvParams params;
  if (!parseConvParams(ctx, input, kernel, padding, output, metadata,
                       &params)) {
    ctx->signalError(Context::FAILURE, "invalid conv params");
  }

#if defined(TAO_AARCH64)
  applyACLThreadPoolConfigIfNotSet();
  if (isAclSupportedDepthwiseConv(ctx, stream_handle, input, kernel, padding,
                                  output, metadata, params)) {
    return runAclDepthwiseKernel(ctx, stream_handle, input, kernel, padding,
                                 output, metadata, params, timer);
  }
  if (isAclSupportedConv(ctx, stream_handle, input, kernel, padding, output,
                         metadata, params)) {
    return runAclConvKernel(ctx, stream_handle, input, kernel, padding, output,
                            metadata, params, timer);
  }
#endif

  // TODO(disc): use context-specific stream/engine
  if (enableInOutLayoutTuning()) {
    ideep::tensor y;
    ideep::convolution_forward::compute(
        params.src, params.weight, params.dst_dims, y, params.strides,
        params.dilates, params.padding_l, params.padding_r, params.groups);
    // reorder to dst format
    y.reorder_to(params.dst);
  } else {
    if (params.weight_is_const && isWeightPrePackingEnabled()) {
      ideep::convolution_forward_params conv_params;
      ideep::convolution_forward::prepare<true>(
          conv_params, params.src, params.weight, params.dst_dims, params.dst,
          params.strides, params.dilates, params.padding_l, params.padding_r,
          params.groups);

      ideep::tensor packed_weight;
      std::string unique_name = "tao_ral.cpu.mkldnn_conv_" +
                                tao::ral::TaoTypeNameHelper<Tinput>::Invoke();
      auto state = ctx->getOrCreateResource<MkldnnConvState>(
          unique_name, []() { return new MkldnnConvState; });
      {
        std::lock_guard<std::mutex> l(state->mu);
        auto& packed_weights = state->packed_weight_cache[kernel.data];
        for (auto& tensor : packed_weights) {
          if (conv_params.pd.weights_desc() == tensor.get_desc()) {
            packed_weight = tensor;
            break;
          }
        }
        if (packed_weight.is_empty()) {
          packed_weight =
              params.weight.make_grouped_weights(params.groups)
                  .reorder_if_differ_in(conv_params.pd.weights_desc());
          packed_weights.push_back(packed_weight);
        }
      }
      ideep::convolution_forward::compute(conv_params, params.src,
                                          packed_weight, params.dst);
    } else {
      ideep::convolution_forward::compute<true>(
          params.src, params.weight, params.dst_dims, params.dst,
          params.strides, params.dilates, params.padding_l, params.padding_r,
          params.groups);
    }
  }

  timer.Stop();

  if (isProfilingEnabled()) {
    dumpConvLikeKernelProflingInfo<Tinput, Tfilter, Toutput>(
        params, timer.GetNanoSeconds(), "ral_cpu_conv");
  }
}

TAO_RAL_API("ral_conv", "cpu", ral_conv<float, 3>);
TAO_RAL_API("ral_conv", "cpu", ral_conv<float, 4>);
TAO_RAL_API("ral_conv", "cpu", ral_conv<float, 5>);

template <typename TKey, typename TValue>
using CpuGemmKeyMap = std::unordered_map<TKey, TValue, typename TKey::Hasher>;

#if defined(TAO_X86)
class MklPackedWeight {
 public:
  MklPackedWeight(ExecutionContext* ctx, const GEMMParamsKey& key);
  ~MklPackedWeight();

  opaque_t packed_weight() const { return packed_weight_; }

 private:
  Context* ctx_;
  cpu::CPUDriver* driver_;
  opaque_t packed_weight_ = nullptr;
};

MklPackedWeight::MklPackedWeight(ExecutionContext* ctx,
                                 const GEMMParamsKey& key)
    : ctx_(ctx->getContext()),
      driver_(ctx->getDriver<cpu::CPUDriver>(cpu::CPUDriver::name())) {
  size_t packed_weight_size =
      cblas_sgemm_pack_get_size(CblasBMatrix, key.m, key.n, key.k);
  packed_weight_ = driver_->raw_alloc(ctx_, packed_weight_size);
  cblas_sgemm_pack(
      CblasRowMajor, CblasBMatrix, key.transpose_b ? CblasTrans : CblasNoTrans,
      key.m, key.n, key.k, 1.0, static_cast<float*>(key.const_weight_ptr),
      key.transpose_b ? key.k : key.n, static_cast<float*>(packed_weight_));
}

MklPackedWeight::~MklPackedWeight() {
  if (!packed_weight_) return;
  driver_->raw_dealloc(ctx_, packed_weight_);
}

using MklGemmCache =
    ideep::utils::lru_cache<GEMMParamsKey, std::shared_ptr<MklPackedWeight>,
                            CpuGemmKeyMap>;

struct MklGemmState : public Context::Resource {
  std::mutex mu;
  MklGemmCache cache{getWeightPrePackingCacheCapacity()};
};

#endif

template <typename Tinput, int N = 2, typename Tweight = Tinput,
          typename Toutput = Tinput>
void mkl_ral_gemm(ExecutionContext* ctx, void* stream_handle,
                  MemRefType<Tinput, N> A, MemRefType<Tweight, N> B,
                  MemRefType<Toutput, N> C, bool tp_a, bool tp_b,
                  bool weight_is_const) {
#if not defined(TAO_X86)
  ctx->signalError(Context::FAILURE, "mkl_ral_gemm not impl");
#else
  int m = tp_a ? A.sizes[1] : A.sizes[0];
  int k = tp_a ? A.sizes[0] : A.sizes[1];
  int n = tp_b ? B.sizes[0] : B.sizes[1];

  if (!isWeightPrePackingForMatMulEnabled() || !weight_is_const) {
    cblas_sgemm(CblasRowMajor, tp_a ? CblasTrans : CblasNoTrans,
                tp_b ? CblasTrans : CblasNoTrans, m, n, k, 1.0,
                reinterpret_cast<Tinput*>(A.data), A.strides[0],
                reinterpret_cast<Tweight*>(B.data), B.strides[0], 0.0,
                reinterpret_cast<Toutput*>(C.data), C.strides[0]);
    return;
  }

  std::string unique_name =
      "tao_ral.cpu.mkl_gemm_" + tao::ral::TaoTypeNameHelper<Tinput>::Invoke();
  auto state = ctx->getOrCreateResource<MklGemmState>(
      unique_name, []() { return new MklGemmState; });
  opaque_t packed_weight;
  {
    GEMMParamsKey key{
        m, n, k, 1, tp_a, tp_b, B.data, nullptr, kDiscCpuDefaultThreadId};
    std::lock_guard<std::mutex> l(state->mu);
    auto& cache = state->cache;
    auto it = cache.find(key);
    if (it == cache.end()) {
      std::shared_ptr<MklPackedWeight> packed_weight_ptr(
          new MklPackedWeight(ctx, key));
      it = cache.insert(std::make_pair(key, packed_weight_ptr)).first;
    }
    packed_weight = it->second->packed_weight();
  }
  cblas_sgemm_compute(CblasRowMajor, tp_a ? CblasTrans : CblasNoTrans,
                      CblasPacked, m, n, k, reinterpret_cast<Tinput*>(A.data),
                      A.strides[0], reinterpret_cast<Tweight*>(packed_weight),
                      B.strides[0], 0.0, reinterpret_cast<Toutput*>(C.data),
                      C.strides[0]);
#endif
}

using MatmulPrimitive = ideep::matmul_forward::super;
using OnednnAclGemmCache =
    ideep::utils::lru_cache<GEMMParamsKey, std::shared_ptr<MatmulPrimitive>,
                            CpuGemmKeyMap>;

struct OnednnAclGemmState : public Context::Resource {
  std::mutex mu;
  OnednnAclGemmCache cached_primitive{getWeightPrePackingCacheCapacity()};
};

std::shared_ptr<MatmulPrimitive> getOrCreateMatmulPrimitive(
    OnednnAclGemmCache& cached_primitive, const GEMMParamsKey& key,
    const tensor& src, const tensor& weight, tensor& output) {
  auto it = cached_primitive.find(key);
  if (it == cached_primitive.end()) {
    auto pb =
        ideep::matmul_forward::get_primitive_desc<true>(src, weight, output);
    std::shared_ptr<MatmulPrimitive> primitive(new MatmulPrimitive(pb));
    it = cached_primitive.insert(std::make_pair(key, std::move(primitive)))
             .first;
  }
  return it->second;
}

template <typename Tinput, int N = 2, typename Tweight = Tinput,
          typename Toutput = Tinput>
void onednn_ral_gemm(ExecutionContext* ctx, void* stream_handle,
                     MemRefType<Tinput, N> A, MemRefType<Tweight, N> B,
                     MemRefType<Toutput, N> C, bool tp_a, bool tp_b,
                     bool weight_is_const) {
  int m = tp_a ? A.sizes[1] : A.sizes[0];
  int k = tp_a ? A.sizes[0] : A.sizes[1];
  int n = tp_b ? B.sizes[0] : B.sizes[1];

#if defined(TAO_X86)
  if (!isWeightPrePackingForMatMulEnabled() || !weight_is_const) {
    dnnl::sgemm(tp_a ? 'T' : 'N', tp_b ? 'T' : 'N', m, n, k, 1.0,
                reinterpret_cast<const float*>(A.data), A.strides[0],
                reinterpret_cast<const float*>(B.data), B.strides[0], 0.0,
                reinterpret_cast<float*>(C.data), C.strides[0]);
    return;
  }
#endif

  data_type input_dtype = toDataType<Tinput>();
  tensor src{dims{m, k}, input_dtype, tp_a ? format_tag::ba : format_tag::ab,
             A.data};
  data_type weight_dtype = toDataType<Tweight>();
  tensor weight{dims{k, n}, weight_dtype,
                tp_b ? format_tag::ba : format_tag::ab, B.data};
  data_type output_dtype = toDataType<Toutput>();
  tensor output{dims{m, n}, output_dtype, format_tag::ab, C.data};

#if defined(TAO_AARCH64)
  // not using pre-packing path
  if (!isWeightPrePackingForMatMulEnabled() || !weight_is_const) {
    ideep::matmul_forward::compute<true>(src, weight, output);
    return;
  }

  std::string unique_name = "tao_ral.cpu.onednn_acl_gemm_" +
                            tao::ral::TaoTypeNameHelper<Tinput>::Invoke();
  auto state = ctx->getOrCreateResource<OnednnAclGemmState>(
      unique_name, []() { return new OnednnAclGemmState; });
  std::shared_ptr<MatmulPrimitive> primitive;
  {
    GEMMParamsKey key{
        m, n, k, 1, tp_a, tp_b, B.data, nullptr, std::this_thread::get_id()};
    std::lock_guard<std::mutex> l(state->mu);
    primitive = getOrCreateMatmulPrimitive(state->cached_primitive, key, src,
                                           weight, output);
  }
  ideep::matmul_forward::compute(*primitive, src, weight, output);
#elif defined(TAO_X86)
  auto weights_desc =
      ideep::matmul_forward::expected_weights_desc(src, weight, output);

  ideep::tensor packed_weight;
  std::string unique_name = "tao_ral.cpu.onednn_gemm_" +
                            tao::ral::TaoTypeNameHelper<Tinput>::Invoke();
  auto state = ctx->getOrCreateResource<OnednnGemmState>(
      unique_name, []() { return new OnednnGemmState; });
  packed_weight =
      state->get_or_create_packed_weight(B.data, weight, weights_desc);
  ideep::matmul_forward::compute</* keep_format */ true,
                                 /* weight_format_any */ true>(
      src, packed_weight, output);
#else
  ctx->signalError(Context::FAILURE, "onednn_ral_gemm not impl");
#endif
}

template <typename Tinput, int N = 2, typename Tweight = Tinput,
          typename Toutput = Tinput>
void ral_gemm(ExecutionContext* ctx, void* stream_handle,
              MemRefType<Tinput, N> A, MemRefType<Tweight, N> B,
              MemRefType<Toutput, N> C, bool tp_a, bool tp_b,
              bool weight_is_const) {
  CpuTimer timer("ral_cpu_gemm");
  if (isEmptyMemref(A) || isEmptyMemref(B) || isEmptyMemref(C)) {
    TAO_VLOG(1) << "ral_gemm: early return for empty tensor";
    return;
  }

#if defined(TAO_AARCH64)
  applyACLThreadPoolConfigIfNotSet();
#endif

  int64_t m = tp_a ? A.sizes[1] : A.sizes[0];
  int64_t k = tp_a ? A.sizes[0] : A.sizes[1];
  if (k != (tp_b ? B.sizes[1] : B.sizes[0])) {
    ctx->signalError(Context::FAILURE, "mismatch contraction dim for gemm");
    return;
  }
  int64_t n = (tp_b ? B.sizes[0] : B.sizes[1]);

#if defined(TAO_AARCH64)
  data_type input_dtype = toDataType<Tinput>();
  data_type weight_dtype = toDataType<Tweight>();
  data_type result_dtype = toDataType<Toutput>();
  if (((input_dtype == data_type::bf16 && weight_dtype == data_type::bf16 &&
        result_dtype == data_type::f32) ||
       (input_dtype == data_type::f16 && weight_dtype == data_type::f16 &&
        result_dtype == data_type::f16)) &&
      N == 2) {
    runAclAMPGemmKernel(ctx, stream_handle, A, B, C, tp_a, tp_b,
                        weight_is_const);
    return;
  }
#endif

  DiscCpuMathKernelMode mode = GetDiscCpuMathKernelMode();
  if (mode == kDiscPreferOneDNN) {
    onednn_ral_gemm(ctx, stream_handle, A, B, C, tp_a, tp_b, weight_is_const);
  } else if (mode == kDiscPreferMKL) {
    mkl_ral_gemm(ctx, stream_handle, A, B, C, tp_a, tp_b, weight_is_const);
  } else {
    assert(mode == kDiscPreferTuningBasedSelection);
    ctx->signalError(Context::FAILURE,
                     "auto tuning mode for cpu gemm is not supported yet.");
    return;
  }

  timer.Stop();

  if (isProfilingEnabled()) {
    int64_t bytes = sizeof(Tinput) * m * k + sizeof(Tweight) * k * n +
                    sizeof(Toutput) * m * n;
    TAO_VLOG(0) << "ral_cpu_gemm:\n"
                << "\tpa = " << A.data << "\n"
                << "\tpb = " << B.data << "\n"
                << "\tpc = " << C.data << "\n"
                << "\tm = " << m << "\n"
                << "\tn = " << n << "\n"
                << "\tk = " << k << "\n"
                << "\ttp_a = " << tp_a << "\n"
                << "\ttp_b = " << tp_b << "\n"
                << "\tweight_is_const = " << weight_is_const << "\n"
                << "\tMath Ops = " << 2 * m * n * k << "\n"
                << "\tBytes = " << bytes << "\n"
                << "\tBandwidth = "
                << double(bytes) / double(timer.GetNanoSeconds()) << " GB\n"
                << "\tGFLOPS = "
                << double(2 * m * n * k) / double(timer.GetNanoSeconds())
                << "\n";
  }
}

TAO_RAL_API("ral_gemm", "cpu", ral_gemm<float>);
#if defined(TAO_AARCH64)
TAO_RAL_API("ral_gemm", "cpu", ral_gemm<bfloat16, 2, bfloat16, float>);
TAO_RAL_API("ral_gemm", "cpu", ral_gemm<float16, 2, float16, float16>);
#endif

template <typename T, int N>
int64_t GetBatchSize(MemRefType<T, N> memref) {
  int64_t batch = 1;
  for (int64_t i = 0; i < N - 2; ++i) {
    batch *= memref.sizes[i];
  }
  return batch;
}

template <typename Tinput, int N, typename Tweight = Tinput,
          typename Toutput = Tinput>
void mkl_ral_batch_gemm(ExecutionContext* ctx, void* stream_handle,
                        MemRefType<Tinput, N> A, MemRefType<Tweight, N> B,
                        MemRefType<Toutput, N> C, bool tp_a, bool tp_b,
                        bool weight_is_const) {
#if not defined(TAO_X86)
  ctx->signalError(Context::FAILURE, "mkl_ral_batch_gemm not impl");
#else
  int b = GetBatchSize(A);
  int m = tp_a ? A.sizes[N - 1] : A.sizes[N - 2];
  int n = tp_b ? B.sizes[N - 2] : B.sizes[N - 1];
  int k = tp_a ? A.sizes[N - 2] : A.sizes[N - 1];

  int ldA = A.strides[N - 2];
  int ldB = B.strides[N - 2];
  int ldC = C.strides[N - 2];

  CBLAS_TRANSPOSE ta = tp_a ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE tb = tp_b ? CblasTrans : CblasNoTrans;
  Tinput alpha = 1.0f;
  Tinput beta = 0.0f;

  cblas_sgemm_batch_strided(CblasRowMajor, ta, tb, m, n, k, alpha, A.data, ldA,
                            m * k, B.data, ldB, k * n, beta, C.data, ldC, m * n,
                            b);
#endif
}

template <typename Tinput, int N, typename Tweight = Tinput,
          typename Toutput = Tinput>
void onednn_ral_batch_gemm(ExecutionContext* ctx, void* stream_handle,
                           MemRefType<Tinput, N> A, MemRefType<Tweight, N> B,
                           MemRefType<Toutput, N> C, bool tp_a, bool tp_b,
                           bool weight_is_const) {
  int b = GetBatchSize(A);
  int m = tp_a ? A.sizes[N - 1] : A.sizes[N - 2];
  int n = tp_b ? B.sizes[N - 2] : B.sizes[N - 1];
  int k = tp_a ? A.sizes[N - 2] : A.sizes[N - 1];

  data_type input_dtype = toDataType<Tinput>();
  tensor src{dims{b, m, k}, input_dtype,
             tp_a ? format_tag::acb : format_tag::abc, A.data};
  data_type weight_dtype = toDataType<Tweight>();
  tensor weight{dims{b, k, n}, weight_dtype,
                tp_b ? format_tag::acb : format_tag::abc, B.data};
  data_type output_dtype = toDataType<Toutput>();
  tensor output{dims{b, m, n}, output_dtype, format_tag::abc, C.data};

#if defined(TAO_X86)
  // TODO(disc): oneDNN does not support pre-packing for batch gemm????
  ideep::matmul_forward::compute<true>(src, weight, output);
#elif defined(TAO_AARCH64)
  // not using pre-packing path
  if (!isWeightPrePackingForMatMulEnabled() || !weight_is_const) {
    ideep::matmul_forward::compute<true>(src, weight, output);
    return;
  }

  std::string unique_name = "tao_ral.cpu.onednn_acl_batch_gemm_" +
                            tao::ral::TaoTypeNameHelper<Tinput>::Invoke();
  auto state = ctx->getOrCreateResource<OnednnAclGemmState>(
      unique_name, []() { return new OnednnAclGemmState; });
  std::shared_ptr<MatmulPrimitive> primitive;
  {
    std::lock_guard<std::mutex> l(state->mu);
    GEMMParamsKey key{
        m, n, k, b, tp_a, tp_b, B.data, nullptr, std::this_thread::get_id()};
    primitive = getOrCreateMatmulPrimitive(state->cached_primitive, key, src,
                                           weight, output);
  }
  ideep::matmul_forward::compute(*primitive, src, weight, output);
#else
  ctx->signalError(Context::FAILURE, "onednn_ral_batch_gemm not impl");
#endif
}

template <typename Tinput, int N, typename Tweight = Tinput,
          typename Toutput = Tinput>
void ral_batch_gemm(ExecutionContext* ctx, void* stream_handle,
                    MemRefType<Tinput, N> A, MemRefType<Tweight, N> B,
                    MemRefType<Toutput, N> C, bool tp_a, bool tp_b,
                    bool weight_is_const) {
  static_assert(N > 2, "batch gemm requires operands with rank higher than 2");
  CpuTimer timer("ral_cpu_batch_gemm");
  if (isEmptyMemref(A) || isEmptyMemref(B) || isEmptyMemref(C)) {
    TAO_VLOG(1) << "ral_cpu_batch_gemm: early return for empty tensor";
    return;
  }

#if defined(TAO_AARCH64)
  applyACLThreadPoolConfigIfNotSet();
#endif

  int64_t batch_a = GetBatchSize(A);
  int64_t batch_b = GetBatchSize(B);
  int64_t batch_c = GetBatchSize(C);
  if (batch_a != batch_b || batch_a != batch_c) {
    ctx->signalError(Context::FAILURE, "mismatch batch size");
    return;
  }

  int64_t m = tp_a ? A.sizes[N - 1] : A.sizes[N - 2];
  int64_t n = tp_b ? B.sizes[N - 2] : B.sizes[N - 1];
  int64_t k = tp_a ? A.sizes[N - 2] : A.sizes[N - 1];
  int64_t kb = tp_b ? B.sizes[N - 1] : B.sizes[N - 2];
  if (C.sizes[N - 2] != m || C.sizes[N - 1] != n || kb != k) {
    ctx->signalError(Context::FAILURE, "mismatch batch gemm params");
    return;
  }

  DiscCpuMathKernelMode mode = GetDiscCpuMathKernelMode();
  if (mode == kDiscPreferOneDNN) {
    onednn_ral_batch_gemm(ctx, stream_handle, A, B, C, tp_a, tp_b,
                          weight_is_const);
  } else if (mode == kDiscPreferMKL) {
    mkl_ral_batch_gemm(ctx, stream_handle, A, B, C, tp_a, tp_b,
                       weight_is_const);
  } else {
    assert(mode == kDiscPreferTuningBasedSelection);
    ctx->signalError(
        Context::FAILURE,
        "auto tuning mode for cpu batch gemm is not supported yet.");
    return;
  }

  timer.Stop();
  if (isProfilingEnabled()) {
    int64_t bytes =
        batch_a * (sizeof(Tinput) * m * k + sizeof(Tweight) * k * n +
                   sizeof(Toutput) * m * n);
    TAO_VLOG(0) << "ral_cpu_batch_gemm:\n"
                << "\tpa = " << A.data << "\n"
                << "\tpb = " << B.data << "\n"
                << "\tpc = " << C.data << "\n"
                << "\tbatch = " << batch_a << "\n"
                << "\tm = " << m << "\n"
                << "\tn = " << n << "\n"
                << "\tk = " << k << "\n"
                << "\ttp_a = " << tp_a << "\n"
                << "\ttp_b = " << tp_b << "\n"
                << "\tweight_is_const = " << weight_is_const << "\n"
                << "\tMath Ops = " << 2 * batch_a * m * n * k << "\n"
                << "\tBytes = " << bytes << "\n"
                << "\tBandwidth = "
                << double(bytes) / double(timer.GetNanoSeconds()) << " GB\n"
                << "\tGFLOPS = "
                << double(2 * batch_a * m * n * k) /
                       double(timer.GetNanoSeconds())
                << "\n";
  }
}

TAO_RAL_API("ral_gemm", "cpu", ral_batch_gemm<float, 3>);
TAO_RAL_API("ral_gemm", "cpu", ral_batch_gemm<float, 4>);

}  // namespace ral
}  // namespace tao
#endif
