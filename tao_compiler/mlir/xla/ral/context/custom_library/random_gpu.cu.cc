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

#include "tensorflow/compiler/mlir/xla/ral/context/custom_library/random.h"

namespace tao {
namespace ral {
namespace random {

#define __restrict__

template <class Distribution, bool VariableSamplesPerOutput>
struct FillPhiloxRandomKernel;

template <class Distribution>
struct FillPhiloxRandomKernel<Distribution, false> {
  typedef typename Distribution::ResultElementType T;
  PHILOX_DEVICE_INLINE void Run(const uint64_t *key, const uint64_t *counter,
                                random::PhiloxRandom gen, T *data, int64_t size,
                                Distribution dist);
};

template <class Distribution>
struct FillPhiloxRandomKernel<Distribution, true> {
  typedef typename Distribution::ResultElementType T;
  PHILOX_DEVICE_INLINE void Run(const uint64_t *key, const uint64_t *counter,
                                random::PhiloxRandom base_gen, T *data,
                                int64_t size, Distribution dist);
};

template <typename T, int ElementCount> class SampleCopier {
public:
  inline __device__ void operator()(T *__restrict__ buf,
                                    const Array<T, ElementCount> &array) const {
#pragma unroll
    for (int i = 0; i < ElementCount; i++) {
      buf[i] = array[i];
    }
  }
};

template <> class SampleCopier<float, 4> {
public:
  // Copies the elements from the array to buf. buf must be 128-bit aligned,
  // which is true for tensor data, and all offsets that are a multiple of the
  // vector size (because the vectors are 128 bits long).
  inline __device__ void operator()(float *__restrict__ buf,
                                    const Array<float, 4> &array) const {
    // NOTE(ringwalt): It's not safe to cast &array[0] to a float4, because they
    // have 32-bit alignment vs 128-bit alignment. There seems to be no
    // performance loss when assigning each element to a vector.
    float4 vec;
    vec.x = array[0];
    vec.y = array[1];
    vec.z = array[2];
    vec.w = array[3];
    float4 *buf_vector = reinterpret_cast<float4 *>(buf);
    *buf_vector = vec;
  }
};

template <> class SampleCopier<int32_t, 4> {
public:
  // Copies the elements from the array to buf. buf must be 128-bit aligned,
  // which is true for tensor data, and all offsets that are a multiple of the
  // vector size (because the vectors are 128 bits long).
  inline __device__ void operator()(int32_t *__restrict__ buf,
                                    const Array<int32_t, 4> &array) const {
    int4 vec;
    vec.x = array[0];
    vec.y = array[1];
    vec.z = array[2];
    vec.w = array[3];
    int4 *buf_vector = reinterpret_cast<int4 *>(buf);
    *buf_vector = vec;
  }
};

template <> class SampleCopier<double, 2> {
public:
  // Copies the elements from the array to buf. buf must be 128-bit aligned,
  // which is true for tensor data, and all offsets that are a multiple of the
  // vector size (because the vectors are 128 bits long).
  inline __device__ void operator()(double *__restrict__ buf,
                                    const Array<double, 2> &array) const {
    double2 vec;
    vec.x = array[0];
    vec.y = array[1];
    double2 *buf_vector = reinterpret_cast<double2 *>(buf);
    *buf_vector = vec;
  }
};

template <> class SampleCopier<int64_t, 2> {
public:
  // Copies the elements from the array to buf. buf must be 128-bit aligned,
  // which is true for tensor data, and all offsets that are a multiple of the
  // vector size (because the vectors are 128 bits long).
  inline __device__ void operator()(int64_t *__restrict__ buf,
                                    const Array<int64_t, 2> &array) const {
    longlong2 vec;
    vec.x = array[0];
    vec.y = array[1];
    longlong2 *buf_vector = reinterpret_cast<longlong2 *>(buf);
    *buf_vector = vec;
  }
};

// A cuda kernel to fill the data with random numbers from the specified
// distribution. Each output takes a fixed number of samples.
template <class Distribution>
PHILOX_DEVICE_INLINE void FillPhiloxRandomKernel<Distribution, false>::Run(
    const uint64_t *key, const uint64_t *counter, random::PhiloxRandom gen,
    T *data, int64_t size, Distribution dist) {
  const int kGroupSize = Distribution::kResultElementCount;

  const int32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t total_thread_count = gridDim.x * blockDim.x;
  int32_t offset = thread_id * kGroupSize;
  if (key != nullptr && counter != nullptr) {
    gen = GetPhiloxRandomFromCounterKeyMem(counter, key);
  }
  gen.Skip(thread_id);

  const SampleCopier<T, kGroupSize> copier;
  while (offset + kGroupSize <= size) {
    const typename Distribution::ResultType samples = dist(&gen);
    copier(&data[offset], samples);

    offset += total_thread_count * kGroupSize;
    gen.Skip(total_thread_count - 1);
  }

  typename Distribution::ResultType samples = dist(&gen);
  for (int i = 0; i < kGroupSize; ++i) {
    if (offset >= size) {
      return;
    }
    data[offset] = samples[i];
    ++offset;
  }
}

// A simple launch pad to call the correct function templates to fill the data
template <class Distribution>
__global__ void __launch_bounds__(256)
    FillPhiloxRandomKernelLaunch(const uint64_t *key, const uint64_t *counter,
                                 random::PhiloxRandom base_gen,
                                 typename Distribution::ResultElementType *data,
                                 int64_t size, Distribution dist) {
  FillPhiloxRandomKernel<Distribution,
                         Distribution::kVariableSamplesPerOutput>()
      .Run(key, counter, base_gen, data, size, dist);
}

template <class Distribution>
void FillPhiloxRandom<Distribution>::operator()(
    const uint64_t *key, const uint64_t *counter, PhiloxRandom gen,
    typename Distribution::ResultElementType *data, int64_t size,
    Distribution dist, void *stream) {
  if (size == 0)
    return;
  // TODO: make this tunable?
  const int32_t block_size = 256;
  const int32_t num_blocks = (size + block_size - 1) / block_size;
#if TENSORFLOW_USE_ROCM
  hipLaunchKernelGGL(FillPhiloxRandomKernelLaunch, num_blocks, block_size, 0,
                     (hipStream_t)stream, key, counter, gen, data, size, dist);
#else
  FillPhiloxRandomKernelLaunch<<<num_blocks, block_size, 0,
                                 (cudaStream_t)stream>>>(key, counter, gen,
                                                         data, size, dist);
#endif
}

template struct FillPhiloxRandom<UniformDistribution<PhiloxRandom, float>>;

} // namespace random
} // namespace ral
} // namespace tao
