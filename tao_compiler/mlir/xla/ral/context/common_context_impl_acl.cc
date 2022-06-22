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

#if defined(TAO_CPU_ONLY) && defined(TAO_AARCH64)

#include "tensorflow/compiler/mlir/xla/ral/context/common_context_impl_acl.h"

#include <unordered_map>

#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/InfoHelpers.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/NEON/functions/NEFFTConvolutionLayer.h"
#include "src/common/utils/Log.h"
#include "src/cpu/operators/CpuConv2d.h"
#include "src/cpu/operators/CpuDepthwiseConv2d.h"

namespace arm_compute {
using namespace arm_compute::experimental;
using namespace arm_compute::misc;
using namespace arm_compute::misc::shape_calculator;

/** Memory region CPU implementation */
// TODO(disc): use ral allocator
class DISCMemoryRegion final : public IMemoryRegion {
 public:
  /** Constructor
   *
   * @param[in] size      Region size
   * @param[in] alignment Alignment in bytes of the base pointer. Defaults to 0
   */
  DISCMemoryRegion(size_t size, size_t alignment = 0)
      : IMemoryRegion(size), _mem(nullptr), _ptr(nullptr) {
    if (size != 0) {
      // Allocate backing memory
      size_t space = size + alignment;
      _mem = std::shared_ptr<uint8_t>(new uint8_t[space],
                                      [](uint8_t* ptr) { delete[] ptr; });
      _ptr = _mem.get();

      // Calculate alignment offset
      if (alignment != 0) {
        void* aligned_ptr = _mem.get();
        std::align(alignment, size, aligned_ptr, space);
        _ptr = aligned_ptr;
      }
    }
  }
  DISCMemoryRegion(void* ptr, size_t size)
      : IMemoryRegion(size), _mem(nullptr), _ptr(nullptr) {
    if (size != 0) {
      _ptr = ptr;
    }
  }
  /** Prevent instances of this class from being copied (As this class contains
   * pointers) */
  DISCMemoryRegion(const DISCMemoryRegion&) = delete;
  /** Default move constructor */
  DISCMemoryRegion(DISCMemoryRegion&&) = default;
  /** Prevent instances of this class from being copied (As this class contains
   * pointers) */
  DISCMemoryRegion& operator=(const DISCMemoryRegion&) = delete;
  /** Default move assignment operator */
  DISCMemoryRegion& operator=(DISCMemoryRegion&&) = default;

  // Inherited methods overridden :
  void* buffer() final { return _ptr; }
  const void* buffer() const final { return _ptr; }
  std::unique_ptr<IMemoryRegion> extract_subregion(size_t offset,
                                                   size_t size) final {
    if (_ptr != nullptr && (offset < _size) && (_size - offset >= size)) {
      return std::make_unique<DISCMemoryRegion>(
          static_cast<uint8_t*>(_ptr) + offset, size);
    } else {
      return nullptr;
    }
  }

 protected:
  std::shared_ptr<uint8_t> _mem;
  void* _ptr;
};

// TODO(disc): use ral allocator
class DISCAllocator final : public IAllocator {
 public:
  /** Default constructor */
  DISCAllocator() = default;

  // Inherited methods overridden:
  void* allocate(size_t size, size_t alignment) override;
  void free(void* ptr) override;
  std::unique_ptr<IMemoryRegion> make_region(size_t size,
                                             size_t alignment) override;
};

void* DISCAllocator::allocate(size_t size, size_t alignment) {
  ARM_COMPUTE_UNUSED(alignment);
  return ::operator new(size);
}

void DISCAllocator::free(void* ptr) { ::operator delete(ptr); }

std::unique_ptr<IMemoryRegion> DISCAllocator::make_region(size_t size,
                                                          size_t alignment) {
  return std::make_unique<DISCMemoryRegion>(size, alignment);
}

using DISCBuffers = std::vector<std::unique_ptr<IMemoryRegion>>;

template <typename TensorType>
WorkspaceData<TensorType> manage_runtime_workspace(
    const experimental::MemoryRequirements& mem_reqs, ITensorPack& run_pack,
    ITensorPack& persistent_pack, IAllocator& allocator, DISCBuffers& buffers) {
  WorkspaceData<TensorType> workspace_memory;
  for (const auto& req : mem_reqs) {
    if (req.size == 0 ||
        req.lifetime == experimental::MemoryLifetime::Prepare) {
      continue;
    }

    if (req.lifetime == experimental::MemoryLifetime::Persistent) {
      run_pack.add_tensor(req.slot, persistent_pack.get_tensor(req.slot));
      continue;
    }

    const auto aux_info = TensorInfo{TensorShape(req.size), 1, DataType::U8};
    workspace_memory.emplace_back(WorkspaceDataElement<TensorType>{
        req.slot, req.lifetime, std::make_unique<TensorType>()});

    auto aux_tensor = workspace_memory.back().tensor.get();
    ARM_COMPUTE_ERROR_ON_NULLPTR(aux_tensor);
    aux_tensor->allocator()->init(aux_info, req.alignment);
    run_pack.add_tensor(req.slot, aux_tensor);

    if (req.lifetime != experimental::MemoryLifetime::Prepare) {
      buffers.emplace_back(allocator.make_region(req.size, req.alignment));
      aux_tensor->allocator()->import_memory(buffers.back()->buffer());
      // aux_tensor->allocator()->allocate();
    }
  }

  return workspace_memory;
}

struct DISCNEConvolutionLayer::Impl {
  MemoryGroup memory_group{};
  std::shared_ptr<IMemoryManager> memory_manager{};
  std::unique_ptr<cpu::ICpuOperator> op{nullptr};
  experimental::MemoryRequirements aux_mem_req{};
  WorkspaceData<Tensor> workspace{};
  std::unique_ptr<IFunction> func{nullptr};
  // persistent tensors are usually those buffers that are used to store packed
  // weights.
  ITensorPack persistent_pack{};
  bool is_prepared = false;
};

DISCNEConvolutionLayer::DISCNEConvolutionLayer(
    std::shared_ptr<IMemoryManager> memory_manager)
    : _impl(std::make_unique<Impl>()) {
  _impl->memory_manager = std::move(memory_manager);
}

DISCNEConvolutionLayer::~DISCNEConvolutionLayer() = default;

void DISCNEConvolutionLayer::configure(ITensor* input, const ITensor* weights,
                                       const ITensor* biases, ITensor* output,
                                       const PadStrideInfo& conv_info,
                                       const WeightsInfo& weights_info,
                                       const Size2D& dilation,
                                       const ActivationLayerInfo& act_info,
                                       bool enable_fast_math,
                                       unsigned int num_groups) {
  // Perform validate step
  ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
  ARM_COMPUTE_UNUSED(num_groups);
  ARM_COMPUTE_ERROR_THROW_ON(DISCNEConvolutionLayer::validate(
      input->info(), weights->info(),
      ((biases != nullptr) ? biases->info() : nullptr), output->info(),
      conv_info, weights_info, dilation, act_info, enable_fast_math,
      num_groups));
  ARM_COMPUTE_LOG_PARAMS(input, weights, biases, output, conv_info,
                         weights_info, dilation, act_info, enable_fast_math,
                         num_groups);

  const Conv2dInfo info(conv_info, dilation, act_info, enable_fast_math,
                        num_groups);
  switch (cpu::CpuConv2d::get_convolution_method(
      input->info(), weights->info(), output->info(), conv_info, weights_info,
      dilation, act_info, enable_fast_math)) {
    case ConvolutionMethod::WINOGRAD:
    case ConvolutionMethod::GEMM:
    case ConvolutionMethod::GEMM_CONV2D:
    case ConvolutionMethod::DIRECT: {
      auto f = std::make_unique<cpu::CpuConv2d>();
      f->configure(input->info(), weights->info(),
                   ((biases != nullptr) ? biases->info() : nullptr),
                   output->info(), conv_info, weights_info, dilation, act_info,
                   enable_fast_math, num_groups);
      _impl->op = std::move(f);
      break;
    }
    case ConvolutionMethod::FFT: {
      auto f = std::make_unique<NEFFTConvolutionLayer>(_impl->memory_manager);
      f->configure(input, weights, biases, output, conv_info, act_info);
      _impl->func = std::move(f);
      break;
    }
    default:
      ARM_COMPUTE_ERROR("Not supported.");
      break;
  }

  if (_impl->op) {
    _impl->memory_group = MemoryGroup(std::move(_impl->memory_manager));
    _impl->aux_mem_req = _impl->op->workspace();
  }
}

Status DISCNEConvolutionLayer::validate(
    const ITensorInfo* input, const ITensorInfo* weights,
    const ITensorInfo* biases, const ITensorInfo* output,
    const PadStrideInfo& conv_info, const WeightsInfo& weights_info,
    const Size2D& dilation, const ActivationLayerInfo& act_info,
    bool enable_fast_math, unsigned int num_groups) {
  const Conv2dInfo info(conv_info, dilation, act_info, enable_fast_math,
                        num_groups);
  switch (cpu::CpuConv2d::get_convolution_method(
      input, weights, output, conv_info, weights_info, dilation, act_info,
      enable_fast_math)) {
    case ConvolutionMethod::WINOGRAD:
    case ConvolutionMethod::GEMM:
    case ConvolutionMethod::GEMM_CONV2D:
    case ConvolutionMethod::DIRECT:
      ARM_COMPUTE_RETURN_ON_ERROR(cpu::CpuConv2d::validate(
          input, weights, biases, output, conv_info, weights_info, dilation,
          act_info, enable_fast_math, num_groups));
      break;
    // TODO(disc): uncomment this block once we support FFT conv.
    // case ConvolutionMethod::FFT:
    //     ARM_COMPUTE_RETURN_ON_ERROR(NEFFTConvolutionLayer::validate(input,
    //     weights, biases, output, conv_info, act_info)); break;
    default:
      ARM_COMPUTE_ERROR("Not supported.");
      break;
  }
  return Status{};
}

ConvolutionMethod DISCNEConvolutionLayer::get_convolution_method(
    const ITensorInfo* input, const ITensorInfo* weights,
    const ITensorInfo* output, const PadStrideInfo& conv_info,
    const WeightsInfo& weights_info, const Size2D& dilation,
    const ActivationLayerInfo& act_info, bool enable_fast_math) {
  return cpu::CpuConv2d::get_convolution_method(
      input, weights, output, conv_info, weights_info, dilation, act_info,
      enable_fast_math);
}

void DISCNEConvolutionLayer::run() { ARM_COMPUTE_ERROR("Not supported."); }

void DISCNEConvolutionLayer::prepare() { ARM_COMPUTE_ERROR("Not supported."); }

void DISCNEConvolutionLayer::prepare(ITensor* input, const ITensor* weights,
                                     const ITensor* biases, ITensor* output) {
  if (_impl->func) {
    _impl->func->prepare();
  } else {
    if (_impl->is_prepared) return;

    ITensorPack run_pack = {{ACL_SRC_0, input},
                            {ACL_SRC_1, weights},
                            {ACL_SRC_2, biases},
                            {ACL_DST, output}};
    ITensorPack prep_pack = {{ACL_SRC_1, weights}, {ACL_SRC_2, biases}};
    _impl->workspace = manage_workspace<Tensor>(
        _impl->aux_mem_req, _impl->memory_group, run_pack, prep_pack);

    _impl->op->prepare(prep_pack);

    // Release temporary tensors that are only used in prepare stage
    for (auto& ws : _impl->workspace) {
      const int slot = ws.slot;
      for (auto& m : _impl->aux_mem_req) {
        if (m.slot != slot) continue;
        auto tensor = ws.tensor.get();
        if (m.lifetime != experimental::MemoryLifetime::Persistent) {
          tensor->allocator()->free();
          break;
        } else {
          _impl->persistent_pack.add_tensor(slot, tensor);
        }
      }
    }
    _impl->is_prepared = true;
  }
}

void DISCNEConvolutionLayer::run(ITensor* input, const ITensor* weights,
                                 const ITensor* biases, ITensor* output) {
  prepare(input, weights, biases, output);

  if (_impl->func) {
    _impl->func->run();
  } else {
    ITensorPack run_pack = {{ACL_SRC_0, input},
                            {ACL_SRC_1, weights},
                            {ACL_SRC_2, biases},
                            {ACL_DST, output}};
    DISCAllocator allocator;
    DISCBuffers buffers;
    auto workspace = manage_runtime_workspace<Tensor>(
        _impl->aux_mem_req, run_pack, _impl->persistent_pack, allocator,
        buffers);
    _impl->op->run(run_pack);
  }
}

DISCNEDepthwiseConvolutionLayer::~DISCNEDepthwiseConvolutionLayer() = default;

struct DISCNEDepthwiseConvolutionLayer::
    NEDepthwiseConvolutionLayerOptimizedInternal::Impl {
  Tensor packed_weights{};  // INT_4
  std::shared_ptr<cpu::CpuDepthwiseConv2d> op{nullptr};
  std::shared_ptr<cpu::CpuDepthwiseConv2dAssemblyDispatch> dwc_optimized_func{
      nullptr};
  bool is_prepared{false};
};

DISCNEDepthwiseConvolutionLayer::NEDepthwiseConvolutionLayerOptimizedInternal::
    NEDepthwiseConvolutionLayerOptimizedInternal(
        std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(memory_manager), _impl(std::make_unique<Impl>()) {}

void DISCNEDepthwiseConvolutionLayer::
    NEDepthwiseConvolutionLayerOptimizedInternal::configure(
        ITensor* input, const ITensor* weights, const ITensor* biases,
        ITensor* output, const PadStrideInfo& conv_info,
        unsigned int depth_multiplier, const ActivationLayerInfo& act_info,
        const Size2D& dilation) {
  ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);

  _impl->op = std::make_unique<cpu::CpuDepthwiseConv2d>();
  ConvolutionInfo info{conv_info, depth_multiplier, act_info, dilation};
  _impl->op->configure(input->info(), weights->info(),
                       biases == nullptr ? nullptr : biases->info(),
                       output->info(), info);

  // Configure pipeline
  ActivationLayerInfo act_info_to_use = ActivationLayerInfo();
  const bool is_relu = arm_compute::utils::info_helpers::is_relu(act_info);
  const bool is_relu6 = arm_compute::utils::info_helpers::is_relu6(act_info);
  bool is_activationlayer_enabled =
      act_info.enabled() && !(is_relu || is_relu6);

  if (!is_activationlayer_enabled) {
    act_info_to_use = act_info;
  }
  info =
      ConvolutionInfo{conv_info, depth_multiplier, act_info_to_use, dilation};

  _impl->dwc_optimized_func =
      std::make_unique<cpu::CpuDepthwiseConv2dAssemblyDispatch>();
  _impl->dwc_optimized_func->configure(
      input->info(), weights->info(),
      biases == nullptr ? nullptr : biases->info(), output->info(), info);

  // Allocate memory based on the internal memory requirements
  experimental::MemoryRequirements mem_req =
      _impl->dwc_optimized_func->workspace();
  _impl->packed_weights.allocator()->init(
      TensorInfo(TensorShape{mem_req[1].size + mem_req[1].alignment}, 1,
                 DataType::S8),
      mem_req[1].alignment);
  _impl->packed_weights.allocator()->allocate();
}

Status DISCNEDepthwiseConvolutionLayer::
    NEDepthwiseConvolutionLayerOptimizedInternal::validate(
        const ITensorInfo* input, const ITensorInfo* weights,
        const ITensorInfo* biases, const ITensorInfo* output,
        const PadStrideInfo& conv_info, unsigned int depth_multiplier,
        const ActivationLayerInfo& act_info, const Size2D& dilation) {
  ConvolutionInfo info{conv_info, depth_multiplier, act_info, dilation};
  return cpu::CpuDepthwiseConv2d::validate(input, weights, biases, output,
                                           info);
}

void DISCNEDepthwiseConvolutionLayer::
    NEDepthwiseConvolutionLayerOptimizedInternal::run() {
  ARM_COMPUTE_ERROR(
      "NEDepthwiseConvolutionLayerOptimizedInternal::run() is not "
      "implemented.");
}

void DISCNEDepthwiseConvolutionLayer::
    NEDepthwiseConvolutionLayerOptimizedInternal::prepare() {
  ARM_COMPUTE_ERROR(
      "NEDepthwiseConvolutionLayerOptimizedInternal::prepare() is not "
      "implemented.");
}

void DISCNEDepthwiseConvolutionLayer::
    NEDepthwiseConvolutionLayerOptimizedInternal::run(ITensor* input,
                                                      const ITensor* weights,
                                                      const ITensor* biases,
                                                      ITensor* output) {
  prepare(input, weights, biases, output);
  experimental::MemoryRequirements mem_req =
      _impl->dwc_optimized_func->workspace();
  DISCAllocator allocator;
  DISCBuffers buffers;
  Tensor workspace;
  buffers.emplace_back(allocator.make_region(
      mem_req[0].size + mem_req[0].alignment, mem_req[0].alignment));
  workspace.allocator()->init(
      TensorInfo(TensorShape{mem_req[0].size + mem_req[0].alignment}, 1,
                 DataType::S8),
      mem_req[0].alignment);
  workspace.allocator()->import_memory(buffers[0]->buffer());

  ITensorPack pack;
  pack.add_tensor(TensorType::ACL_SRC_0, input);
  pack.add_tensor(TensorType::ACL_SRC_1, weights);
  pack.add_tensor(TensorType::ACL_SRC_2, biases);
  pack.add_tensor(TensorType::ACL_INT_3, &workspace);
  pack.add_tensor(TensorType::ACL_INT_4, &_impl->packed_weights);
  pack.add_tensor(TensorType::ACL_DST_0, output);
  _impl->op->run(pack);
}

void DISCNEDepthwiseConvolutionLayer::
    NEDepthwiseConvolutionLayerOptimizedInternal::prepare(
        ITensor* input, const ITensor* weights, const ITensor* biases,
        ITensor* output) {
  if (!_impl->is_prepared) {
    // init run to save packed weights
    ITensorPack run_pack = {{TensorType::ACL_SRC_1, weights},
                            {TensorType::ACL_SRC_2, biases},
                            {TensorType::ACL_INT_4, &_impl->packed_weights}};
    _impl->op->prepare(run_pack);
    _impl->is_prepared = true;
  }
}

struct DISCNEDepthwiseConvolutionLayer::NEDepthwiseConvolutionLayerGeneric::
    Impl {
  std::shared_ptr<cpu::CpuDepthwiseConv2d> op{nullptr};
};

DISCNEDepthwiseConvolutionLayer::NEDepthwiseConvolutionLayerGeneric::
    NEDepthwiseConvolutionLayerGeneric()
    : _impl(std::make_unique<Impl>()) {}

void DISCNEDepthwiseConvolutionLayer::NEDepthwiseConvolutionLayerGeneric::
    configure(ITensor* input, const ITensor* weights, const ITensor* biases,
              ITensor* output, const PadStrideInfo& conv_info,
              unsigned int depth_multiplier,
              const ActivationLayerInfo& act_info, const Size2D& dilation) {
  ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
  ARM_COMPUTE_ERROR_THROW_ON(NEDepthwiseConvolutionLayer::validate(
      input->info(), weights->info(),
      (biases == nullptr) ? nullptr : biases->info(), output->info(), conv_info,
      depth_multiplier, act_info, dilation));

  const ConvolutionInfo info{conv_info, depth_multiplier, act_info, dilation};
  _impl->op = std::make_unique<cpu::CpuDepthwiseConv2d>();
  _impl->op->configure(input->info(), weights->info(),
                       biases == nullptr ? nullptr : biases->info(),
                       output->info(), info);

  auto depthwise_conv_kernel =
      std::make_unique<cpu::kernels::CpuDepthwiseConv2dNativeKernel>();
  depthwise_conv_kernel->configure(input->info(), weights->info(),
                                   biases == nullptr ? nullptr : biases->info(),
                                   output->info(), info);
}

Status
DISCNEDepthwiseConvolutionLayer::NEDepthwiseConvolutionLayerGeneric::validate(
    const ITensorInfo* input, const ITensorInfo* weights,
    const ITensorInfo* biases, const ITensorInfo* output,
    const PadStrideInfo& conv_info, unsigned int depth_multiplier,
    const ActivationLayerInfo& act_info, const Size2D& dilation) {
  ConvolutionInfo info{conv_info, depth_multiplier, act_info, dilation};
  return cpu::CpuDepthwiseConv2d::validate(input, weights, biases, output,
                                           info);
}

void DISCNEDepthwiseConvolutionLayer::NEDepthwiseConvolutionLayerGeneric::
    run() {
  ARM_COMPUTE_ERROR(
      "DISCNEDepthwiseConvolutionLayer::run() is not implemented.");
}

void DISCNEDepthwiseConvolutionLayer::NEDepthwiseConvolutionLayerGeneric::
    prepare(ITensor* input, const ITensor* weights, const ITensor* biases,
            ITensor* output) {}

void DISCNEDepthwiseConvolutionLayer::NEDepthwiseConvolutionLayerGeneric::run(
    ITensor* input, const ITensor* weights, const ITensor* biases,
    ITensor* output) {
  ITensorPack pack;
  pack.add_tensor(TensorType::ACL_SRC_0, input);
  pack.add_tensor(TensorType::ACL_SRC_1, weights);
  pack.add_tensor(TensorType::ACL_SRC_2, biases);
  pack.add_tensor(TensorType::ACL_DST_0, output);

  _impl->op->run(pack);
}

DISCNEDepthwiseConvolutionLayer::DISCNEDepthwiseConvolutionLayer(
    std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)),
      _impl(std::make_unique<Impl>()) {}

struct DISCNEDepthwiseConvolutionLayer::Impl {
  DepthwiseConvolutionFunction depth_conv_func{
      DepthwiseConvolutionFunction::OPTIMIZED};
  NEDepthwiseConvolutionLayerOptimizedInternal func_optimized{nullptr};
  NEDepthwiseConvolutionLayerGeneric func_generic{};
  std::shared_ptr<cpu::CpuDepthwiseConv2d> op{nullptr};
};

void DISCNEDepthwiseConvolutionLayer::configure(
    ITensor* input, const ITensor* weights, const ITensor* biases,
    ITensor* output, const PadStrideInfo& conv_info,
    unsigned int depth_multiplier, const ActivationLayerInfo& act_info,
    const Size2D& dilation) {
  ARM_COMPUTE_LOG_PARAMS(input, weights, output, conv_info, depth_multiplier,
                         biases, act_info, dilation);

  const ConvolutionInfo info{conv_info, depth_multiplier, act_info, dilation};
  _impl->op = std::make_shared<cpu::CpuDepthwiseConv2d>();
  _impl->depth_conv_func = _impl->op->get_depthwiseconvolution_function(
      input->info(), weights->info(),
      (biases != nullptr) ? biases->info() : nullptr, output->info(), info);
  switch (_impl->depth_conv_func) {
    case DepthwiseConvolutionFunction::OPTIMIZED:
      _impl->func_optimized.configure(input, weights, biases, output, conv_info,
                                      depth_multiplier, act_info, dilation);
      break;
    case DepthwiseConvolutionFunction::GENERIC:
      _impl->func_generic.configure(input, weights, biases, output, conv_info,
                                    depth_multiplier, act_info, dilation);
      break;
    default:
      ARM_COMPUTE_ERROR("Unsupported DepthwiseConvolutionFunction");
  }
}

Status DISCNEDepthwiseConvolutionLayer::validate(
    const ITensorInfo* input, const ITensorInfo* weights,
    const ITensorInfo* biases, const ITensorInfo* output,
    const PadStrideInfo& conv_info, unsigned int depth_multiplier,
    const ActivationLayerInfo& act_info, const Size2D& dilation) {
  ConvolutionInfo info{conv_info, depth_multiplier, act_info, dilation};
  return cpu::CpuDepthwiseConv2d::validate(input, weights, biases, output,
                                           info);
}

void DISCNEDepthwiseConvolutionLayer::run() {
  ARM_COMPUTE_ERROR("DepthwiseConvolutionFunction::run() is not implemented.");
}

void DISCNEDepthwiseConvolutionLayer::prepare() {
  ARM_COMPUTE_ERROR(
      "DepthwiseConvolutionFunction::prepare() is not implemented.");
}

void DISCNEDepthwiseConvolutionLayer::run(ITensor* input,
                                          const ITensor* weights,
                                          const ITensor* biases,
                                          ITensor* output) {
  switch (_impl->depth_conv_func) {
    case DepthwiseConvolutionFunction::OPTIMIZED:
      _impl->func_optimized.run(input, weights, biases, output);
      break;
    case DepthwiseConvolutionFunction::GENERIC:
      _impl->func_generic.run(input, weights, biases, output);
      break;
    default:
      ARM_COMPUTE_ERROR("DepthwiseConvolutionFunction not properly configured");
  }
}

void DISCNEDepthwiseConvolutionLayer::prepare(ITensor* input,
                                              const ITensor* weights,
                                              const ITensor* biases,
                                              ITensor* output) {
  switch (_impl->depth_conv_func) {
    case DepthwiseConvolutionFunction::OPTIMIZED:
      _impl->func_optimized.prepare(input, weights, biases, output);
      break;
    case DepthwiseConvolutionFunction::GENERIC:
      _impl->func_generic.prepare(input, weights, biases, output);
      break;
    default:
      ARM_COMPUTE_ERROR("DepthwiseConvolutionFunction not properly configured");
  }
}

}  // namespace arm_compute

#endif  // defined(TAO_CPU_ONLY) && defined(TAO_AARCH64)
