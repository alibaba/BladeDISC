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

#ifndef RAL_CONTEXT_CUSTOM_LIBRARY_RANDOM_H_
#define RAL_CONTEXT_CUSTOM_LIBRARY_RANDOM_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "mlir/custom_ops/custom_library/philox_random.h"

namespace tao {
namespace ral {
namespace random {

// The following 2 functions use the contract "lower 32 bits for the first
// uint32_t, higher 32 bits for the second". Note that this is endian-neutral,
// unlike a direct memory copy `memcpy(output, &input, 8)`.
PHILOX_DEVICE_INLINE void Uint64ToUint32s(uint64_t input, uint32_t* output1,
                                          uint32_t* output2) {
  *output1 = static_cast<uint32_t>(input);
  *output2 = static_cast<uint32_t>(input >> 32);
}

PHILOX_DEVICE_INLINE uint64_t Uint32sToUint64(uint32_t input1,
                                              uint32_t input2) {
  auto u64_1 = static_cast<uint64_t>(input1);
  auto u64_2 = static_cast<uint64_t>(input2);
  return u64_1 | (u64_2 << 32);
}

PHILOX_DEVICE_INLINE PhiloxRandom::ResultType GetCounterFromMem(
    uint64_t const* ptr) {
  PhiloxRandom::ResultType counter;
  Uint64ToUint32s(ptr[0], &counter[0], &counter[1]);
  Uint64ToUint32s(ptr[1], &counter[2], &counter[3]);
  return counter;
}

PHILOX_DEVICE_INLINE void WriteCounterToMem(
    PhiloxRandom::ResultType const& counter, uint64_t* ptr) {
  ptr[0] = Uint32sToUint64(counter[0], counter[1]);
  ptr[1] = Uint32sToUint64(counter[2], counter[3]);
}

PHILOX_DEVICE_INLINE PhiloxRandom::Key GetKeyFromMem(uint64_t const* ptr) {
  PhiloxRandom::Key key;
  Uint64ToUint32s(ptr[0], &key[0], &key[1]);
  return key;
}

PHILOX_DEVICE_INLINE void WriteKeyToMem(PhiloxRandom::Key const& key,
                                        uint64_t* ptr) {
  *ptr = Uint32sToUint64(key[0], key[1]);
}

PHILOX_DEVICE_INLINE PhiloxRandom GetPhiloxRandomFromCounterKeyMem(
    uint64_t const* counter_ptr, uint64_t const* key_ptr) {
  return PhiloxRandom(GetCounterFromMem(counter_ptr), GetKeyFromMem(key_ptr));
}

// Helper function to convert a 32-bit integer to a float between [0..1).
PHILOX_DEVICE_INLINE float Uint32ToFloat(uint32_t x);

// Computes a + b. Requires that the result is representable in the destination
// type and that b is not maximal (i.e. b + 1 is not 0). Notably, the addend b
// need *not* be representable in that type. (The condition on b excludes the
// extremal case INT_MIN + UINT_MAX = INT_MAX, which this function cannot
// compute.)
template <typename Int>
PHILOX_DEVICE_INLINE Int SignedAdd(Int a,
                                   typename std::make_unsigned<Int>::type b) {
  // Implementation note: both b_div_2 and b - b_div_2 are positive and
  // representable as Int.
  auto b_div_2 = b >> 1;
  return a + static_cast<Int>(b_div_2) + static_cast<Int>(b - b_div_2);
}

// A class that generates uniform distribution random numbers from the
// underlying random integer generator.
// Arguments:
//   Generator: a generator type that returns a number of uint32_t upon each
//              invocation. It needs to define kResultElementCount for the
//              sample count for each invocation, and ResultType for the
//              actual returned sample type.
//   RealType: the data type of the real numbers that will be returned by the
//             distribution. This could be either float or double for now.
// This class is meant to be implemented through specialization. The default
// is not defined by design.
template <class Generator, typename RealType>
class UniformDistribution;

template <class Generator>
class UniformDistribution<Generator, float> {
 public:
  // The number of elements that will be returned.
  static constexpr int kResultElementCount = Generator::kResultElementCount;
  // Cost of generation of a single element (in cycles).
  static constexpr int kElementCost = 3;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static constexpr bool kVariableSamplesPerOutput = false;
  typedef Array<float, kResultElementCount> ResultType;
  typedef float ResultElementType;

  // Must have lo < hi
  UniformDistribution(float lo, float hi) : lo_(lo), range_(hi - lo) {}

  PHILOX_DEVICE_INLINE
  ResultType operator()(Generator* gen) {
    typename Generator::ResultType sample = (*gen)();
    ResultType result;
    for (int i = 0; i < kResultElementCount; ++i) {
      result[i] = range_ * Uint32ToFloat(sample[i]) + lo_;
    }
    return result;
  }

 private:
  float lo_;
  float range_;
};

// Helper function to convert an 32-bit integer to a float between [0..1).
PHILOX_DEVICE_INLINE float Uint32ToFloat(uint32_t x) {
  // IEEE754 floats are formatted as follows (MSB first):
  //    sign(1) exponent(8) mantissa(23)
  // Conceptually construct the following:
  //    sign == 0
  //    exponent == 127  -- an excess 127 representation of a zero exponent
  //    mantissa == 23 random bits
  const uint32_t man = x & 0x7fffffu;  // 23 bit mantissa
  const uint32_t exp = static_cast<uint32_t>(127);
  const uint32_t val = (exp << 23) | man;

  // Assumes that endian-ness is same for float and uint32_t.
  float result;
#if TENSORFLOW_USE_ROCM
  // ROCm compilation in Cmake only accepts memcpy without std
  memcpy(&result, &val, sizeof(val));
#else
  std::memcpy(&result, &val, sizeof(val));
#endif
  return result - 1.0f;
}

template <class Distribution>
struct FillPhiloxRandom {
  void operator()(const uint64_t* key, const uint64_t* counter,
                  PhiloxRandom gen,
                  typename Distribution::ResultElementType* data, int64_t size,
                  Distribution dist, void* stream);
};

}  // namespace random
}  // namespace ral
}  // namespace tao

#endif  // RAL_CONTEXT_CUSTOM_LIBRARY_RANDOM_H_