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

#include <cassert>
#include <mutex>
#include <random>
#include <unordered_map>
#include <vector>

#include "mlir/xla/ral/context/context_util.h"
#include "mlir/xla/ral/context/custom_library/random.h"
#include "mlir/xla/ral/device/gpu/gpu_driver.h"
#include "mlir/xla/ral/ral_context.h"
#include "mlir/xla/ral/ral_driver.h"
#include "mlir/xla/ral/ral_helper.h"
#include "mlir/xla/ral/ral_logging.h"

using tao::ral::buffer_t;
using tao::ral::Context;
using tao::ral::ExecutionContext;
using tao::ral::MemRefType;
using tao::ral::gpu::GPUDriver;
using tao::ral::gpu::stream_t;

namespace tao {
namespace ral {
namespace random {

std::mt19937_64* InitRngWithRandomSeed() {
  std::random_device device("/dev/urandom");
  return new std::mt19937_64(device());
}

uint64_t New64() {
  static std::mt19937_64* rng = InitRngWithRandomSeed();
  static std::mutex mu;
  std::lock_guard<std::mutex> l(mu);
  return (*rng)();
}

class GuardedPhiloxRandom {
 public:
  // Must call Init to finish initialization
  GuardedPhiloxRandom(int64_t seed, int64_t seed2) : initialized_(false) {
    Init(seed, seed2);
  }

  // Initialize with given seeds.
  void Init(int64_t seed, int64_t seed2) {
    assert(!initialized_);
    if (seed == 0 && seed2 == 0) {
      // If both seeds are unspecified, use completely random seeds.
      seed = New64();
      seed2 = New64();
    }
    std::lock_guard<std::mutex> l(mu_);
    generator_ = PhiloxRandom(seed, seed2);
    initialized_ = true;
  }

  // Reserve a certain number of 128-bit samples.
  // This function is thread safe.  The returned generator is valid for the
  // given number of samples, and can be used without a lock.
  PhiloxRandom ReserveSamples128(int64_t samples) {
    assert(initialized_);
    std::lock_guard<std::mutex> l(mu_);
    auto local = generator_;
    generator_.Skip(samples);
    return local;
  }

  // Reserve a certain number of 32-bit samples.
  PhiloxRandom ReserveSamples32(int64_t samples) {
    return ReserveSamples128((samples + 3) / 4);
  }

  // Reserve enough random samples in the generator for the given output count.
  PhiloxRandom ReserveRandomOutputs(int64_t output_count, int multiplier) {
    int64_t conservative_sample_count = output_count * multiplier;
    return ReserveSamples128(conservative_sample_count);
  }

 private:
  std::mutex mu_;
  PhiloxRandom generator_;
  bool initialized_;
};

struct RalRngUniformState : public Context::Resource {
  std::mutex mu;
  using GeneratorType =
      std::unordered_map<int64_t, std::unique_ptr<GuardedPhiloxRandom>>;
  GeneratorType generators;
};

template <typename T, int N, typename Tidx = int>
void ral_gpu_random_uniform(ExecutionContext* ctx, void* stream_handle,
                            MemRefType<T, 0> start, MemRefType<T, 0> limit,
                            MemRefType<Tidx, 1>, MemRefType<T, N> output,
                            int64_t id, int64_t seed, int64_t seed2) {
  T a = *start.data;
  T b = *limit.data;
  TAO_VLOG(2) << "random id#" << id << ":\n"
              << "\tseed: (" << seed << "," << seed2 << ")\n"
              << "\tstart: " << a << "\n"
              << "\tlimit: " << b;
  std::string unique_name =
      "tao_ral.gpu.rng_uniform." + tao::ral::TaoTypeNameHelper<T>::Invoke();
  auto state = ctx->getOrCreateResource<RalRngUniformState>(
      unique_name, []() { return new RalRngUniformState; });
  auto gpu_driver = ctx->getDriver<GPUDriver>(GPUDriver::name());
  auto stream = gpu_driver->asCUStream(ctx, stream_handle);

  GuardedPhiloxRandom* generator = nullptr;
  {
    std::lock_guard<std::mutex> l(state->mu);
    auto it = state->generators.find(id);
    if (it == state->generators.end()) {
      auto r = std::unique_ptr<GuardedPhiloxRandom>(
          new GuardedPhiloxRandom(seed, seed2));
      it = state->generators.emplace(id, std::move(r)).first;
    }
    generator = it->second.get();
  }
  int64_t nelem = Size(output);
  TAO_VLOG(2) << "nelem = " << nelem;
  using Distribution = UniformDistribution<PhiloxRandom, T>;
  Distribution d(a, b);
  FillPhiloxRandom<Distribution>()(
      /*key=*/nullptr, /*counter=*/nullptr,
      // Multiplier 256 is the same as in FillPhiloxRandomTask; do not change
      // it just here.
      generator->ReserveRandomOutputs(nelem, 256), output.data, nelem, d,
      stream);
}

}  // namespace random
}  // namespace ral
}  // namespace tao

namespace tao {
namespace ral {

TAO_RAL_API("ral_gpu_rng_uniform", "gpu",
            random::ral_gpu_random_uniform<float, 1>);
TAO_RAL_API("ral_gpu_rng_uniform", "gpu",
            random::ral_gpu_random_uniform<float, 2>);
TAO_RAL_API("ral_gpu_rng_uniform", "gpu",
            random::ral_gpu_random_uniform<float, 3>);
TAO_RAL_API("ral_gpu_rng_uniform", "gpu",
            random::ral_gpu_random_uniform<float, 4>);
TAO_RAL_API("ral_gpu_rng_uniform", "gpu",
            random::ral_gpu_random_uniform<float, 5>);
TAO_RAL_API("ral_gpu_rng_uniform", "gpu",
            random::ral_gpu_random_uniform<float, 6>);

}  // namespace ral
}  // namespace tao