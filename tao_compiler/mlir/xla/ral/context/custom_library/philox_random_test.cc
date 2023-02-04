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

#include "mlir/xla/ral/context/custom_library/philox_random.h"

#include <gtest/gtest.h>
#include <math.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

namespace tao {
namespace ral {
namespace random {

class TrivialPhiloxDistribution {
 public:
  // The number of elements that will be returned.
  static constexpr int kResultElementCount = PhiloxRandom::kResultElementCount;
  typedef PhiloxRandom::ResultType ResultType;
  typedef PhiloxRandom::ResultElementType ResultElementType;

  PHILOX_DEVICE_INLINE
  ResultType operator()(PhiloxRandom* gen) { return (*gen)(); }
};

// Return a test seed.
inline uint64_t GetTestSeed() { return 1; }

// A utility function to fill the given array with samples from the given
// distribution.
template <class Distribution>
void FillRandoms(PhiloxRandom gen, typename Distribution::ResultElementType* p,
                 int64_t size) {
  const int granularity = Distribution::kResultElementCount;
  Distribution dist;
  for (int i = 0; i < size; i += granularity) {
    const auto sample = dist(&gen);
    std::copy(&sample[0], &sample[0] + granularity, &p[i]);
  }
}

// This test checks that skipping certain number of samples, is equivalent to
// generate the same number of samples without skipping.
TEST(PhiloxRandomTest, SkipMatchTest) {
  constexpr int count = 1024;
  constexpr int skip_count = 2048;

  uint64_t test_seed = GetTestSeed();
  std::vector<uint32_t> v1(count);
  {
    PhiloxRandom gen(test_seed);
    gen.Skip(skip_count / 4);
    FillRandoms<TrivialPhiloxDistribution>(gen, &v1[0], v1.size());
  }

  std::vector<uint32_t> v2(count + skip_count);
  {
    PhiloxRandom gen(test_seed);
    FillRandoms<TrivialPhiloxDistribution>(gen, &v2[0], v2.size());
  }

  for (int i = 0; i < count; ++i) {
    ASSERT_EQ(v1[i], v2[i + skip_count]);
  }
}

}  // namespace random
}  // namespace ral
}  // namespace tao